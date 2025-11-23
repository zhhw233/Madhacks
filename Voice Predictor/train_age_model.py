import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from joblib import Parallel, delayed
import xgboost as xgb
import pickle

# -----------------------------
# Parameters
# -----------------------------
SPEAKERS_CSV = "Voices/speakers_all.csv"
AUDIO_PATH = "/Users/anuj/Downloads/Voices/recordings/recordings"  # Path to audio files
SR = 22050
DURATION = 4        # seconds
N_MFCC = 13
FEATURES_FILE = "age_features.npy"
LABELS_FILE = "age_labels.npy"
MODEL_FILE = "age_model.pkl"
LABEL_ENCODER_FILE = "age_label_encoder.pkl"

# -----------------------------
# Feature extraction function
# -----------------------------
def extract_features(file_path):
    """Extract MFCC features from audio file"""
    try:
        audio, _ = librosa.load(file_path, sr=SR, duration=DURATION)
        if len(audio) < SR * DURATION:
            audio = np.pad(audio, (0, SR*DURATION - len(audio)))
        else:
            audio = audio[:SR*DURATION]

        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
        delta = librosa.feature.delta(mfcc)

        # Aggregate features: mean + std for MFCC and delta
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.std(delta, axis=1)
        ])
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# -----------------------------
# Load speakers CSV and prepare data
# -----------------------------
print("Loading speakers CSV...")
speakers_df = pd.read_csv(SPEAKERS_CSV)

# Filter out rows without age or filename information
speakers_df = speakers_df.dropna(subset=['age', 'filename'])

# Convert age to numeric
speakers_df['age'] = pd.to_numeric(speakers_df['age'], errors='coerce')
speakers_df = speakers_df.dropna(subset=['age'])

# Create age ranges for classification
def age_to_range(age):
    if age <= 25:
        return "18-25"
    elif age <= 35:
        return "26-35"
    elif age <= 45:
        return "36-45"
    elif age <= 60:
        return "46-60"
    else:
        return "60+"

speakers_df['age_range'] = speakers_df['age'].apply(age_to_range)

print(f"Total speakers with age labels: {len(speakers_df)}")
print(f"\nAge distribution:")
print(speakers_df['age_range'].value_counts().sort_index())

# -----------------------------
# Load or compute features
# -----------------------------
if os.path.exists(FEATURES_FILE) and os.path.exists(LABELS_FILE):
    print("\nLoading cached features...")
    X = np.load(FEATURES_FILE)
    y = np.load(LABELS_FILE, allow_pickle=True)
    print(f"Loaded {len(X)} cached feature vectors")
else:
    print("\nExtracting features from audio files...")

    features_list = []
    labels_list = []

    # Process each speaker
    for idx, row in speakers_df.iterrows():
        filename = row['filename']
        age_range = row['age_range']

        # Construct audio file path
        audio_file = os.path.join(AUDIO_PATH, filename)

        if not os.path.exists(audio_file):
            # Try with .mp3 extension if not found
            audio_file = os.path.join(AUDIO_PATH, filename + '.mp3')

        if os.path.exists(audio_file):
            feat = extract_features(audio_file)
            if feat is not None:
                features_list.append(feat)
                labels_list.append(age_range)

        # Progress update
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(speakers_df)} files...")

    X = np.array(features_list)
    y = np.array(labels_list)

    # Save for caching
    np.save(FEATURES_FILE, X)
    np.save(LABELS_FILE, y)
    print(f"Features saved. Total samples: {len(X)}")

# -----------------------------
# Encode labels
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nLabel encoding:")
print(f"Classes: {le.classes_}")
print(f"Total samples: {len(y_encoded)}")

# Save label encoder
with open(LABEL_ENCODER_FILE, "wb") as f:
    pickle.dump(le, f)
print(f"Label encoder saved to {LABEL_ENCODER_FILE}")

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# -----------------------------
# XGBoost Classifier for Age Range Classification
# -----------------------------
print("\nTraining XGBoost model for age prediction...")

num_classes = len(le.classes_)

model = xgb.XGBClassifier(
    n_estimators=400,           # Increased from 200
    max_depth=8,                # Increased from 6
    learning_rate=0.05,         # Decreased for better convergence
    subsample=0.8,              # Sample 80% of data for each tree
    colsample_bytree=0.8,       # Sample 80% of features for each tree
    min_child_weight=3,         # Minimum samples in leaf nodes
    gamma=0.1,                  # Regularization parameter
    reg_alpha=0.1,              # L1 regularization
    reg_lambda=1.0,             # L2 regularization
    objective='multi:softprob',
    num_class=num_classes,
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Calculate mean absolute error in terms of age ranges
# Convert predictions back to age ranges for analysis
y_test_ranges = le.inverse_transform(y_test)
y_pred_ranges = le.inverse_transform(y_pred)

# -----------------------------
# Save the trained model
# -----------------------------
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)
print(f"\nModel saved as {MODEL_FILE}")

print("\n" + "="*60)
print("Age prediction model training complete!")
print("="*60)
