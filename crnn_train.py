#!/usr/bin/env python3
"""
Stable CRNN training script for:
 - Language classification (from audio filenames like 'afrikaans1.mp3')
 - Weak age regression: normalized per-language mean age
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio

# optional librosa fallback
try:
    import librosa
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

def load_audio(path, sr):
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

def filename_to_language(fname: str):
    name = Path(fname).stem
    idx = len(name)
    for i, ch in enumerate(name):
        if ch.isdigit():
            idx = i
            break
    lang = name[:idx].replace('-', '_').replace(' ', '_').strip().lower()
    return lang

class SpeakerDataset(Dataset):
    def __init__(self, df, audio_root, sample_rate=16000, duration=4.0):
        self.df = df.reset_index(drop=True)
        self.audio_root = Path(audio_root)
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * duration)

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
        audio_path = Path(row['path'])
        if not audio_path.is_absolute():
            audio_path = self.audio_root / audio_path

        if not audio_path.exists():
            for ext in ['.mp3', '.wav', '.flac', '.ogg']:
                candidate = audio_path.with_suffix(ext)
                if candidate.exists():
                    audio_path = candidate
                    break

        wav = load_audio(str(audio_path), self.sample_rate)
        wav = self._pad_truncate(wav)

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=1024, hop_length=256, n_mels=80
        )(torch.tensor(wav).float())
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

        # normalize
        mean, std = mel_spec.mean(), mel_spec.std()
        if std < 1e-6: std = 1.0
        mel_spec = (mel_spec - mean) / std
        mel_spec = mel_spec.unsqueeze(0)  # (1, n_mels, time)

        sample = {'spectrogram': mel_spec}
        sample['language'] = -1 if pd.isna(row.get('language_idx', None)) else int(row['language_idx'])
        sample['age_reg'] = -1.0 if pd.isna(row.get('age_reg', None)) else float(row['age_reg'])
        return sample

def collate_fn(batch):
    specs = torch.stack([b['spectrogram'] for b in batch])
    out = {
        'spectrogram': specs,
        'language': torch.tensor([b['language'] for b in batch], dtype=torch.long),
        'age_reg': torch.tensor([b['age_reg'] for b in batch], dtype=torch.float)
    }
    return out

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
    def __init__(self, n_mels=80, rnn_hidden=128, rnn_layers=2, language_classes=None, age_regression=True):
        super().__init__()
        self.conv1 = ConvBlock(1, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)

        freq_after = max(1, n_mels // 8)
        rnn_input = 128 * freq_after
        self.rnn = nn.GRU(rnn_input, rnn_hidden, num_layers=rnn_layers, batch_first=True, bidirectional=True)

        hidden_fc = rnn_hidden * 2
        self.lang_head = nn.Sequential(
            nn.Linear(hidden_fc, hidden_fc//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_fc//2, language_classes)
        ) if language_classes is not None else None

        self.age_head = nn.Sequential(nn.Linear(hidden_fc,64), nn.ReLU(), nn.Linear(64,1)) if age_regression else None

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        B,C,F,T = x.shape
        x = x.permute(0,3,1,2).contiguous()
        x = x.view(B,T,C*F)
        rnn_out, _ = self.rnn(x)
        pooled = rnn_out.mean(dim=1)
        out = {}
        if self.lang_head is not None:
            out['language'] = self.lang_head(pooled)
        if self.age_head is not None:
            out['age'] = self.age_head(pooled)
        return out

def train_epoch(model, loader, optim, device, loss_fns, lambda_age=0.01):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in tqdm(loader, desc='train'):
        specs = batch['spectrogram'].to(device)
        targets_lang = batch['language'].to(device)
        targets_age = batch['age_reg'].to(device)

        optim.zero_grad()
        outputs = model(specs)
        loss = 0.0

        # language
        if 'language' in outputs:
            mask = targets_lang != -1
            if mask.sum() > 0:
                loss += loss_fns['language'](outputs['language'][mask], targets_lang[mask])

        # age regression
        if 'age' in outputs and 'age' in loss_fns:
            mask_age = targets_age != -1.0
            if mask_age.sum() > 0:
                preds = outputs['age'].squeeze(1)[mask_age]
                loss += lambda_age * loss_fns['age'](preds, targets_age[mask_age])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optim.step()

        total_loss += loss.item() * specs.size(0)
        total_samples += specs.size(0)
    return total_loss / (total_samples + 1e-12)

def evaluate(model, loader, device):
    from sklearn.metrics import accuracy_score, mean_squared_error
    model.eval()
    ys_lang, preds_lang = [], []
    ys_age, preds_age = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='eval'):
            specs = batch['spectrogram'].to(device)
            outputs = model(specs)

            if 'language' in outputs:
                mask = batch['language'] != -1
                if mask.sum() > 0:
                    ys_lang.extend(batch['language'][mask].numpy())
                    probs = torch.softmax(outputs['language'][mask], dim=1).cpu().numpy()
                    preds_lang.extend(probs)

            if 'age' in outputs:
                mask_age = batch['age_reg'] != -1.0
                if mask_age.sum() > 0:
                    ys_age.extend(batch['age_reg'][mask_age].numpy())
                    preds_age.extend(outputs['age'].squeeze(1)[mask_age].cpu().numpy())

    results = {}
    if ys_lang:
        y_true = np.array(ys_lang)
        y_pred = np.array(preds_lang).argmax(axis=1)
        results['language_acc'] = accuracy_score(y_true, y_pred)
    if ys_age:
        results['age_mse'] = mean_squared_error(np.array(ys_age), np.array(preds_age))
    return results

def build_metadata_from_files(csv_path, audio_root):
    audio_root = Path(audio_root)
    files = sorted([p for p in audio_root.iterdir() if p.suffix.lower() in ['.mp3','.wav','.flac','.ogg']])
    if not files:
        raise RuntimeError(f"No audio files found in {audio_root}")

    df_csv = pd.read_csv(csv_path)
    if 'native_language' in df_csv.columns:
        df_csv['native_language_norm'] = df_csv['native_language'].astype(str).str.strip().str.lower()
    else:
        df_csv['native_language_norm'] = None

    # mean age per language
    age_mean = {}
    if 'age' in df_csv.columns:
        tmp = df_csv[df_csv['age'].notna()].copy()
        if 'native_language_norm' in tmp.columns:
            age_mean = tmp.groupby('native_language_norm')['age'].mean().to_dict()

    # normalize ages [0,1]
    all_ages = np.array(list(age_mean.values()))
    age_min, age_max = all_ages.min(), all_ages.max()

    rows = []
    for p in files:
        lang = filename_to_language(p.name)
        age_label = age_mean.get(lang, np.nan)
        if not np.isnan(age_label):
            age_norm = (age_label - age_min) / (age_max - age_min)
        else:
            age_norm = np.nan
        rows.append({'path': p.name, 'language': lang, 'age_reg': float(age_norm) if not np.isnan(age_norm) else np.nan})
    return pd.DataFrame(rows)

def main(args):
    meta = build_metadata_from_files(args.csv_path, args.audio_root)
    print(f"Found {len(meta)} audio files and {meta['language'].nunique()} languages.")

    languages = sorted(meta['language'].dropna().unique())
    lang_map = {c:i for i,c in enumerate(languages)}
    meta['language_idx'] = meta['language'].map(lang_map)

    # remove languages with fewer than 2 samples
    counts = meta['language'].value_counts()
    meta = meta[meta['language'].isin(counts[counts >= 2].index)]


    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(meta, test_size=args.test_size, stratify=meta['language'], random_state=42)

    train_ds = SpeakerDataset(train_df, args.audio_root, sample_rate=args.sample_rate, duration=args.duration)
    val_ds = SpeakerDataset(val_df, args.audio_root, sample_rate=args.sample_rate, duration=args.duration)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=max(1,args.num_workers//2))

    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    model = CRNN(n_mels=args.n_mels, rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers,
                 language_classes=len(lang_map), age_regression=True).to(device)

    loss_fns = {'language': nn.CrossEntropyLoss(ignore_index=-1), 'age': nn.MSELoss()}
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'maps.pkl'),'wb') as f:
        pickle.dump({'language_map': lang_map}, f)

    for epoch in range(1,args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optim, device, loss_fns, lambda_age=args.lambda_age)
        print("Train loss:", train_loss)
        val_stats = evaluate(model, val_loader, device)
        print("Val:", val_stats)
        torch.save(model.state_dict(), os.path.join(args.output_dir,f'model_epoch{epoch}.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='/mnt/data/speakers_all.csv')
    parser.add_argument('--audio_root', type=str, default='voices/recordings/recordings')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--duration', type=float, default=4.0)
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--rnn_hidden', type=int, default=128)
    parser.add_argument('--rnn_layers', type=int, default=2)
    parser.add_argument('--lambda_age', type=float, default=0.01, help='weight for age regression relative to language')
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--force_cpu', action='store_true')
    args = parser.parse_args()
    main(args)
