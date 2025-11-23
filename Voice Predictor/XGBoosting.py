import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import Parallel, delayed
import xgboost as xgb
from collections import Counter
import pickle

# -----------------------------
# Parameters
# -----------------------------
audio_path = "Voices/recordings"  # CHANGE THIS to your audio folder path
sr = 22050
duration = 4        # seconds
n_mfcc = 13
features_file = "features.npy"
labels_file = "labels.npy"
min_samples_per_class = 11   # Filter rare classes

# -----------------------------
# Feature extraction function
# -----------------------------
def extract_features(file_path):
    audio, _ = librosa.load(file_path, sr=sr, duration=duration)
    if len(audio) < sr * duration:
        audio = np.pad(audio, (0, sr*duration - len(audio)))
    else:
        audio = audio[:sr*duration]

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)

    # Aggregate features: mean + std for MFCC and delta
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.std(delta, axis=1)
    ])
    return features

# -----------------------------
# Load or compute features
# -----------------------------
if os.path.exists(features_file) and os.path.exists(labels_file):
    X = np.load(features_file)
    y = np.load(labels_file)
    print("Loaded cached features")
else:
    files = [f for f in os.listdir(audio_path) if f.lower().endswith(".mp3")]
    labels = [f.split('n')[0] for f in files]  # assuming filename: "countryn.mp3"
    le = LabelEncoder()
    y = le.fit_transform(labels)

    print("Extracting features in parallel...")
    X = Parallel(n_jobs=-1)(delayed(extract_features)(os.path.join(audio_path, f)) for f in files)
    X = np.array(X)

    # Save for caching
    np.save(features_file, X)
    np.save(labels_file, y)

    # Save label encoder
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("Features saved")

# -----------------------------
# Filter rare classes
# -----------------------------
counts = Counter(y)
valid_classes = [k for k, v in counts.items() if v >= min_samples_per_class]
mask = np.isin(y, valid_classes)
X = X[mask]
y = y[mask]

print(f"Number of samples after filtering rare classes: {len(y)}")
print(f"Number of classes after filtering: {len(valid_classes)}")

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Remap labels to consecutive integers based on training set
# -----------------------------
unique_train_labels = np.unique(y_train)
label_map = {old: new for new, old in enumerate(unique_train_labels)}

y_train = np.array([label_map[yi] for yi in y_train])
mask_test = np.isin(y_test, unique_train_labels)
y_test = np.array([label_map[yi] for yi in y_test[mask_test]])
X_test = X_test[mask_test]

num_class = len(unique_train_labels)
print(f"Number of classes used in training: {num_class}")

# -----------------------------
# XGBoost Classifier
# -----------------------------
model = xgb.XGBClassifier(
    n_estimators=400,           # Increased from 200
    max_depth=8,                # Increased from 5
    learning_rate=0.05,         # Decreased for better convergence
    subsample=0.8,              # Sample 80% of data for each tree
    colsample_bytree=0.8,       # Sample 80% of features for each tree
    min_child_weight=3,         # Minimum samples in leaf nodes
    gamma=0.1,                  # Regularization parameter
    reg_alpha=0.1,              # L1 regularization
    reg_lambda=1.0,             # L2 regularization
    objective='multi:softprob',
    num_class=num_class,
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
print(f"Test Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save the trained model
# -----------------------------
with open("xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as xgboost_model.pkl")
