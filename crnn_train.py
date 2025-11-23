
"""
CRNN multi-task training script for speaker metadata (country, age, gender)

How to use:
1. Put this script next to your CSV (path set below) and audio files (paths in CSV).
2. Adjust parameters (SAMPLE_RATE, DURATION, device) as needed.
3. Run: python crnn_train.py --csv_path /path/to/speakers_all.csv --audio_root /path/to/audio_root
   Recommended: run on a machine with GPU for faster training.

This script:
 - Loads CSV, encodes labels (country, gender, age categorical or regression).
 - Builds a PyTorch Dataset which loads mp3/wav, computes mel-spectrograms, pads/truncates.
 - Builds a CRNN: Conv blocks -> BiGRU -> separate heads for country/gender/age.
 - Training loop with multi-task loss (cross-entropy for classification, MSE for regression).
 - Saves model and label encoders to outputs/ directory.
"""
import argparse, os, math, random, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio

# Fallback to librosa if torchaudio cannot load some formats
try:
    import librosa
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

def load_audio(path, sr):
    # try torchaudio first
    try:
        wav, orig_sr = torchaudio.load(path)
        wav = wav.mean(dim=0, keepdim=True)  # mono
        if orig_sr != sr:
            wav = torchaudio.functional.resample(wav, orig_sr, sr)
        return wav.squeeze(0).numpy()
    except Exception:
        if HAS_LIBROSA:
            wav, _ = librosa.load(path, sr=sr, mono=True)
            return wav
        raise

class SpeakerDataset(Dataset):
    def __init__(self, df, audio_root, sample_rate=16000, duration=4.0,
                 country_map=None, gender_map=None, age_map=None, transforms=None):
        self.df = df.reset_index(drop=True)
        self.audio_root = Path(audio_root)
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * duration)
        self.transforms = transforms
        self.country_map = country_map
        self.gender_map = gender_map
        self.age_map = age_map

    def __len__(self):
        return len(self.df)

    def _pad_truncate(self, wav):
        if len(wav) >= self.target_length:
            start = 0 if len(wav) == self.target_length else np.random.randint(0, len(wav)-self.target_length+1)
            return wav[start:start+self.target_length]
        else:
            pad = self.target_length - len(wav)
            left = pad // 2
            right = pad - left
            return np.pad(wav, (left, right), mode='constant')

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row.get('path') or row.get('filename') or row.get('audio_path') or row.get('file')
        if path is None or (not (self.audio_root / path).exists()):
            # try absolute path in CSV
            path = row.get('absolute_path', path)

        audio_path = Path(path) if Path(path).is_absolute() else (self.audio_root / path)

        # try adding extensions if missing
        if not audio_path.exists():
            for ext in ['.mp3', '.wav']:
                candidate = audio_path.with_suffix(ext)
                if candidate.exists():
                    audio_path = candidate
                    break
            else:
                # try glob in folder
                folder = audio_path.parent
                name = audio_path.name
                matches = list(folder.glob(f"{name}*"))
                if matches:
                    audio_path = matches[0]
                else:
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")

        wav = load_audio(str(audio_path), self.sample_rate)
        # pad/truncate
        wav = self._pad_truncate(wav)

        # mel-spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=1024, hop_length=256, n_mels=80
        )(torch.tensor(wav).float())
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

        # safe normalization
        mean = mel_spec.mean()
        std = mel_spec.std()
        if std < 1e-6:  # avoid divide by zero
            std = 1.0
        mel_spec = (mel_spec - mean) / std

        mel_spec = mel_spec.unsqueeze(0)  # (1, n_mels, time)
        sample = {'spectrogram': mel_spec}

        # labels
        if 'country' in row:
            sample['country'] = -1 if pd.isna(row['country']) else self.country_map.get(str(row['country']), -1)
        if 'gender' in row:
            sample['gender'] = -1 if pd.isna(row['gender']) else self.gender_map.get(str(row['gender']), -1)
        if 'age' in row:
            if self.age_map is not None:
                sample['age_cat'] = -1 if pd.isna(row['age']) else self.age_map.get(str(row['age']), -1)
            else:
                sample['age_reg'] = -1.0 if pd.isna(row['age']) else float(row['age'])
        return sample

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(3,3), pool=(2,2)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=(kernel[0]//2, kernel[1]//2)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel, padding=(kernel[0]//2, kernel[1]//2)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(pool)
        )
    def forward(self, x):
        return self.net(x)

class CRNN(nn.Module):
    def __init__(self, n_mels=80, rnn_hidden=128, rnn_layers=2, country_classes=None, gender_classes=None, age_classes=None, age_regression=False):
        super().__init__()
        # conv frontend
        self.conv1 = ConvBlock(1, 32, kernel=(3,3), pool=(2,2))
        self.conv2 = ConvBlock(32, 64, kernel=(3,3), pool=(2,2))
        self.conv3 = ConvBlock(64, 128, kernel=(3,3), pool=(2,2))
        # compute feature dim after convs for freq axis
        # assume input mel shape (1, n_mels, time). n_mels reduced by 2*2*2 = 8 in freq axis
        freq_after = max(1, n_mels // 8)
        # rnn input size = channels * freq_after
        rnn_input = 128 * freq_after
        self.rnn = nn.GRU(rnn_input, rnn_hidden, num_layers=rnn_layers, batch_first=True, bidirectional=True)
        # classifier heads
        hidden_fc = rnn_hidden * 2
        self.country_head = nn.Sequential(
            nn.Linear(hidden_fc, hidden_fc//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_fc//2, country_classes) if country_classes is not None else nn.Identity()
        ) if country_classes is not None else None
        self.gender_head = nn.Sequential(
            nn.Linear(hidden_fc, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, gender_classes) if gender_classes is not None else nn.Identity()
        ) if gender_classes is not None else None
        if age_regression:
            self.age_head = nn.Sequential(nn.Linear(hidden_fc, 64), nn.ReLU(), nn.Linear(64,1))
        else:
            self.age_head = nn.Sequential(nn.Linear(hidden_fc, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, age_classes)) if age_classes is not None else None

    def forward(self, x):
        # x: (B, 1, n_mels, time)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # collapse freq axis into channel for RNN across time axis
        B, C, F, T = x.shape
        x = x.permute(0,3,1,2).contiguous()  # (B, T, C, F)
        x = x.view(B, T, C*F)  # (B, T, feat)
        rnn_out, _ = self.rnn(x)  # (B, T, 2*hidden)
        # global pooling over time (mean)
        pooled = rnn_out.mean(dim=1)
        outputs = {}
        if self.country_head is not None:
            outputs['country'] = self.country_head(pooled)
        if self.gender_head is not None:
            outputs['gender'] = self.gender_head(pooled)
        if self.age_head is not None:
            outputs['age'] = self.age_head(pooled)
        return outputs

def collate_fn(batch):
    # batch is list of samples (dict)
    specs = torch.stack([b['spectrogram'] for b in batch])
    out = {'spectrogram': specs}
    # optional labels
    if 'country' in batch[0]:
        out['country'] = torch.tensor([b.get('country', -1) for b in batch], dtype=torch.long)
    if 'gender' in batch[0]:
        out['gender'] = torch.tensor([b.get('gender', -1) for b in batch], dtype=torch.long)
    if 'age_cat' in batch[0]:
        out['age_cat'] = torch.tensor([b.get('age_cat', -1) for b in batch], dtype=torch.long)
    if 'age_reg' in batch[0]:
        out['age_reg'] = torch.tensor([b.get('age_reg', -1.0) for b in batch], dtype=torch.float)
    return out

def train_epoch(model, loader, optim, device, loss_fns, lambda_age_reg=1.0):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc='train'):
        specs = batch['spectrogram'].to(device)

        targets = {}
        if 'country' in batch:
            targets['country'] = batch['country'].to(device)
        if 'gender' in batch:
            targets['gender'] = batch['gender'].to(device)
        if 'age_cat' in batch:
            targets['age_cat'] = batch['age_cat'].to(device)
        if 'age_reg' in batch:
            targets['age_reg'] = batch['age_reg'].to(device)

        optim.zero_grad()
        outputs = model(specs)
        loss = 0.0

        # country
        if 'country' in targets and model.country_head is not None:
            mask = targets['country'] != -1
            if mask.sum() > 0:
                loss += loss_fns['country'](outputs['country'][mask], targets['country'][mask])

        # gender
        if 'gender' in targets and model.gender_head is not None:
            mask = targets['gender'] != -1
            if mask.sum() > 0:
                loss += loss_fns['gender'](outputs['gender'][mask], targets['gender'][mask])

        # age categorical
        if 'age_cat' in targets and model.age_head is not None and outputs.get('age') is not None:
            mask = targets['age_cat'] != -1
            if mask.sum() > 0:
                loss += loss_fns['age'](outputs['age'][mask], targets['age_cat'][mask])

        # age regression
        if 'age_reg' in targets and model.age_head is not None and outputs.get('age') is not None:
            mask = targets['age_reg'] != -1.0
            if mask.sum() > 0:
                loss += lambda_age_reg * loss_fns['age_reg'](
                    outputs['age'].squeeze(1)[mask], targets['age_reg'][mask]
                )

        # backward with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optim.step()

        total_loss += loss.item() * specs.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    import numpy as np
    from sklearn.metrics import accuracy_score, top_k_accuracy_score, mean_squared_error
    ys = {'country': [], 'gender': [], 'age_cat': [], 'age_reg': []}
    preds = {'country': [], 'gender': [], 'age_cat': [], 'age_reg': []}
    with torch.no_grad():
        for batch in tqdm(loader, desc='eval'):
            specs = batch['spectrogram'].to(device)
            outputs = model(specs)
            b = specs.size(0)
            if 'country' in batch and outputs.get('country') is not None:
                ys['country'].extend(batch['country'].numpy().tolist())
                preds['country'].extend(torch.softmax(outputs['country'], dim=1).cpu().numpy().tolist())
            if 'gender' in batch and outputs.get('gender') is not None:
                ys['gender'].extend(batch['gender'].numpy().tolist())
                preds['gender'].extend(torch.softmax(outputs['gender'], dim=1).cpu().numpy().tolist())
            if 'age_cat' in batch and outputs.get('age') is not None:
                ys['age_cat'].extend(batch['age_cat'].numpy().tolist())
                preds['age_cat'].extend(torch.softmax(outputs['age'], dim=1).cpu().numpy().tolist())
            if 'age_reg' in batch and outputs.get('age') is not None:
                ys['age_reg'].extend(batch['age_reg'].numpy().tolist())
                preds['age_reg'].extend(outputs['age'].squeeze(1).cpu().numpy().tolist())
    results = {}
    # country metrics
    if len(ys['country'])>0:
        y_true = np.array(ys['country'])
        y_pred_probs = np.array(preds['country'])
        y_pred = y_pred_probs.argmax(axis=1)
        results['country_acc'] = accuracy_score(y_true, y_pred)
        # top-3 accuracy (if classes>=3)
        valid_mask = y_true != -1
        if valid_mask.sum() > 0:
            results['country_top3'] = top_k_accuracy_score(
                y_true[valid_mask],
                y_pred_probs[valid_mask],
                k=3,
                labels=range(y_pred_probs.shape[1])
            )
    # gender metrics
    if len(ys['gender'])>0:
        y_true = np.array(ys['gender'])
        y_pred_probs = np.array(preds['gender'])
        y_pred = y_pred_probs.argmax(axis=1)
        results['gender_acc'] = accuracy_score(y_true, y_pred)
    # age class metrics
    if len(ys['age_cat'])>0:
        y_true = np.array(ys['age_cat'])
        y_pred_probs = np.array(preds['age_cat'])
        y_pred = y_pred_probs.argmax(axis=1)
        results['age_acc'] = accuracy_score(y_true, y_pred)
    # age regression metrics
    if len(ys['age_reg'])>0:
        y_true = np.array(ys['age_reg'])
        y_pred = np.array(preds['age_reg'])
        results['age_mse'] = mean_squared_error(y_true, y_pred)
    return results

def main(args):
    df = pd.read_csv(args.csv_path)
    # require a 'path' or 'filename' column that points to audio files relative to audio_root
    # Clean dataframe and drop rows missing file
    df = df.copy()
    df = df[df["file_missing?"] == False]
    if 'path' not in df.columns and 'filename' in df.columns:
        df.rename(columns={'filename':'path'}, inplace=True)
    if 'path' not in df.columns and 'file' in df.columns:
        df.rename(columns={'file':'path'}, inplace=True)
    df = df[df['path'].notnull()].reset_index(drop=True)
    # create label maps
    country_map, gender_map, age_map = None, None, None
    if 'native_language' in df.columns:
        countries = sorted(df['native_language'].dropna().unique().astype(str).tolist())
        country_map = {c:i for i,c in enumerate(countries)}
    if 'sex' in df.columns:
        genders = sorted(df['sex'].dropna().unique().astype(str).tolist())
        gender_map = {c:i for i,c in enumerate(genders)}
    # decide age: categorical if few unique values, else regression
    age_regression = False
    if 'age' in df.columns:
        unique_ages = df['age'].dropna().unique()
        if len(unique_ages) <= 20:
            age_map = {str(a):i for i,a in enumerate(sorted(unique_ages.astype(str).tolist()))}
            age_regression = False
        else:
            age_map = None
            age_regression = True

    # split train/val (stratify by country if available)
    from sklearn.model_selection import train_test_split
    strat_col = 'native_language' if 'native_language' in df.columns else None
    if strat_col is not None:
        counts = df[strat_col].value_counts()
        valid_classes = counts[counts > 1].index
        df = df[df[strat_col].isin(valid_classes)]
        train_df, val_df = train_test_split(
            df,
            test_size=0.15,
            stratify=df[strat_col],
            random_state=42
        )

    # create datasets and loaders
    train_ds = SpeakerDataset(train_df, args.audio_root, sample_rate=args.sample_rate, duration=args.duration,
                              country_map=country_map, gender_map=gender_map, age_map=age_map)
    val_ds = SpeakerDataset(val_df, args.audio_root, sample_rate=args.sample_rate, duration=args.duration,
                              country_map=country_map, gender_map=gender_map, age_map=age_map)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    model = CRNN(n_mels=args.n_mels, rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers,
                 country_classes=(len(country_map) if country_map is not None else None),
                 gender_classes=(len(gender_map) if gender_map is not None else None),
                 age_classes=(len(age_map) if age_map is not None else None),
                 age_regression=age_regression)
    model = model.to(device)

    # Losses and optimizer
    loss_fns = {}
    if country_map is not None:
        loss_fns['country'] = nn.CrossEntropyLoss(ignore_index=-1)
    if gender_map is not None:
        loss_fns['gender'] = nn.CrossEntropyLoss(ignore_index=-1)
    if age_regression:
        loss_fns['age_reg'] = nn.MSELoss()
    else:
        if age_map is not None:
            loss_fns['age'] = nn.CrossEntropyLoss(ignore_index=-1)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = 1e9
    os.makedirs(args.output_dir, exist_ok=True)
    # save maps
    with open(os.path.join(args.output_dir, 'maps.pkl'), 'wb') as f:
        pickle.dump({'country_map': country_map, 'gender_map': gender_map, 'age_map': age_map, 'age_regression': age_regression}, f)

    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optim, device, loss_fns, lambda_age_reg=args.lambda_age)
        print("Train loss:", train_loss)
        val_stats = evaluate(model, val_loader, device)
        print("Val:", val_stats)
        # save best by country acc or loss
        # here we save every epoch
        torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_epoch{epoch}.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='speakers_all.csv')
    parser.add_argument('--audio_root', type=str, default='audio')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--duration', type=float, default=4.0)
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--rnn_hidden', type=int, default=128)
    parser.add_argument('--rnn_layers', type=int, default=2)
    parser.add_argument('--lambda_age', type=float, default=1.0)
    parser.add_argument('--force_cpu', action='store_true')
    args = parser.parse_args()
    main(args)

