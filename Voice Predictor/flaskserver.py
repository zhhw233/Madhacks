from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import librosa
import xgboost as xgb
import pickle
import os
from werkzeug.utils import secure_filename
import subprocess
import tempfile
import logging

app = Flask(__name__)
CORS(app)

CORS(app)
app = Flask(__name__, static_folder='.', static_url_path='')
logging.basicConfig(level=logging.INFO)

# Config
MODEL_PATH = "xgboost_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
SEX_MODEL_PATH = "sex_model.pkl"
SEX_LABEL_ENCODER_PATH = "sex_label_encoder.pkl"
AGE_MODEL_PATH = "age_model.pkl"
AGE_LABEL_ENCODER_PATH = "age_label_encoder.pkl"
SPEAKERS_CSV = "Voices/speakers_all.csv"
UPLOAD_FOLDER = "temp_uploads"
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'webm', 'm4a', 'flac'}

# Audio params
SR = 22050
# How long we expect incoming recordings (max). The client may record up to this many seconds.
RECORDING_MAX_DURATION = 30
# Window length (seconds) used to compute features — keep this equal to training duration (4s)
FEATURE_DURATION = 4
N_MFCC = 13

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
model = None
label_encoder = None
sex_model = None
sex_label_encoder = None
age_model = None
age_label_encoder = None
speakers_df = None

def load_model():
    global model, label_encoder, sex_model, sex_label_encoder, age_model, age_label_encoder, speakers_df
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f: model = pickle.load(f)
            logging.info(f"Loaded main model: {type(model)}")
        if os.path.exists(LABEL_ENCODER_PATH):
            with open(LABEL_ENCODER_PATH, 'rb') as f: label_encoder = pickle.load(f)
            logging.info(f"Loaded label encoder: {getattr(label_encoder,'classes_',None)}")
        if os.path.exists(SEX_MODEL_PATH):
            with open(SEX_MODEL_PATH, 'rb') as f: sex_model = pickle.load(f)
            logging.info(f"Loaded sex model: {type(sex_model)}")
        if os.path.exists(SEX_LABEL_ENCODER_PATH):
            with open(SEX_LABEL_ENCODER_PATH, 'rb') as f: sex_label_encoder = pickle.load(f)
            logging.info(f"Loaded sex label encoder: {getattr(sex_label_encoder,'classes_',None)}")
        if os.path.exists(AGE_MODEL_PATH):
            with open(AGE_MODEL_PATH, 'rb') as f: age_model = pickle.load(f)
            logging.info(f"Loaded age model: {type(age_model)}")
        if os.path.exists(AGE_LABEL_ENCODER_PATH):
            with open(AGE_LABEL_ENCODER_PATH, 'rb') as f: age_label_encoder = pickle.load(f)
            logging.info(f"Loaded age label encoder: {getattr(age_label_encoder,'classes_',None)}")
        if os.path.exists(SPEAKERS_CSV):
            speakers_df = pd.read_csv(SPEAKERS_CSV)
            logging.info(f"Loaded speakers dataframe: {speakers_df.shape}")
    except Exception as e:
        logging.exception(f"Error loading models: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(file_path):
    try:
        # Load up to the maximum recording duration (so we can accept longer recordings)
        audio, _ = librosa.load(file_path, sr=SR, duration=RECORDING_MAX_DURATION)
        # Select a fixed-length window for feature extraction (keep consistent with training)
        target_len = SR * FEATURE_DURATION
        if len(audio) < target_len:
            # pad to target length
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            # choose a centered window of FEATURE_DURATION seconds
            start = max(0, (len(audio) - target_len) // 2)
            audio = audio[start:start + target_len]

        # Match the feature vector used during training: MFCC mean/std and delta mean/std
        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
        delta = librosa.feature.delta(mfcc)

        # Compute f0 separately for heuristics (do not include in model features)
        f0_mean = np.nan
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=50, fmax=500, sr=SR)
            if f0 is not None:
                f0_mean = float(np.nanmedian(f0))
        except Exception:
            try:
                f0 = librosa.yin(audio, fmin=50, fmax=500)
                if f0 is not None and len(f0) > 0:
                    f0_mean = float(np.nanmedian(f0))
            except Exception:
                f0_mean = 0.0
        if np.isnan(f0_mean):
            f0_mean = 0.0

        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.std(delta, axis=1)
        ])
        # Return (features, f0_mean) so heuristics can use pitch without changing model input dims
        return features, float(f0_mean)
    except Exception as e:
        logging.exception(f"Error extracting features: {e}")
        return None


def convert_to_wav(in_path):
    """Convert an uploaded audio file to a mono WAV at `SR` using ffmpeg if available.
    Returns path to WAV file or original path if conversion not needed/failed."""
    ext = os.path.splitext(in_path)[1].lower()
    if ext == '.wav':
        return in_path
    out_path = in_path + '_conv.wav'
    try:
        cmd = ['ffmpeg', '-y', '-i', in_path, '-ar', str(SR), '-ac', '1', out_path]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logging.info(f"Converted {in_path} -> {out_path} using ffmpeg")
        return out_path
    except Exception as e:
        logging.warning(f"ffmpeg conversion failed ({e}), attempting to use original file")
        return in_path

def get_accent_info(native_language):
    if pd.isna(native_language): return "Unknown"
    accent = ''.join([c for c in str(native_language) if not c.isdigit()]).capitalize()
    accent_map = {
        'Mandarin': 'Mandarin Chinese',
        'Cantonese': 'Cantonese Chinese',
        'Farsi': 'Persian (Farsi)',
        'Kiswahili': 'Swahili',
        'Haitian': 'Haitian Creole',
        'Filipino': 'Filipino (Tagalog)',
    }
    return accent_map.get(accent, accent)

def estimate_age(features, f0=None):
    """Returns (age_range, confidence). `f0` is the pitch (Hz) used for a simple heuristic."""
    global age_model, age_label_encoder
    if features is None: return "Unknown", 0.0

    heuristic_pitch = f0 if (f0 is not None) else 0.0
    heuristic_age = "36-45"
    if heuristic_pitch > 220:
        heuristic_age = "18-25"
    elif heuristic_pitch > 180:
        heuristic_age = "26-35"
    elif heuristic_pitch > 140:
        heuristic_age = "36-45"
    else:
        heuristic_age = "46+"
    heuristic_conf = 50.0

    if age_model is not None and age_label_encoder is not None:
        pred = age_model.predict(features.reshape(1, -1))
        pred_proba = age_model.predict_proba(features.reshape(1, -1))
        model_age = age_label_encoder.inverse_transform(pred)[0]
        model_conf = float(np.max(pred_proba) * 100)
        # Blend heuristic with model
        final_age = model_age if model_conf > 60 else heuristic_age
        final_conf = max(model_conf, heuristic_conf)
        return final_age, final_conf
    return heuristic_age, heuristic_conf

def estimate_sex(features, f0=None):
    """Returns (sex, confidence). `f0` is the pitch (Hz) used for a simple heuristic."""
    global sex_model, sex_label_encoder
    if features is None: return "Unknown", 0.0

    heuristic_pitch = f0 if (f0 is not None) else 0.0
    heuristic_sex = "Female" if heuristic_pitch > 180 else "Male"
    heuristic_conf = 50.0

    if sex_model is not None and sex_label_encoder is not None:
        pred = sex_model.predict(features.reshape(1, -1))
        pred_proba = sex_model.predict_proba(features.reshape(1, -1))
        model_sex = sex_label_encoder.inverse_transform(pred)[0]
        model_conf = float(np.max(pred_proba) * 100)
        final_sex = model_sex.capitalize() if model_conf > 60 else heuristic_sex
        final_conf = max(model_conf, heuristic_conf)
        return final_sex, final_conf
    return heuristic_sex, heuristic_conf

def get_speaker_info(pred_class):
    if speakers_df is None or label_encoder is None: return None
    try:
        native_language = label_encoder.inverse_transform([pred_class])[0]
        accent = get_accent_info(native_language)
        country = "Unknown"
        if not speakers_df.empty:
            matches = speakers_df[speakers_df['native_language']==native_language]
            if not matches.empty:
                country = matches['country'].mode()[0]
        return {'accent': accent, 'country': country, 'native_language': native_language}
    except: return None

@app.route('/')
def index(): return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None: 
        return jsonify({'error':'Model not loaded','accent':'Unknown','age':'Unknown','sex':'Unknown'}),500
    if 'audio' not in request.files: return jsonify({'error':'No audio file'}),400
    file = request.files['audio']
    if file.filename=='': return jsonify({'error':'No file selected'}),400

    filename = secure_filename(file.filename)
    temp_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(temp_path)

    # Convert to WAV for more consistent feature extraction when possible
    wav_path = convert_to_wav(temp_path)
    extracted = extract_features(wav_path)
    if isinstance(extracted, tuple):
        features, f0_mean = extracted
    else:
        features = extracted
        f0_mean = None
    try: os.remove(temp_path)
    except: pass
    # Remove converted file if different
    if 'wav_path' in locals() and wav_path != temp_path:
        try: os.remove(wav_path)
        except: pass
    if features is None: return jsonify({'error':'Failed to extract features'}),500

    logging.info(f"Extracted features shape: {features.shape} values: {np.round(features[:6],3)}...{np.round(features[-6:],3)} f0={f0_mean}")

    features_reshaped = features.reshape(1, -1)
    pred = model.predict(features_reshaped)
    pred_proba = model.predict_proba(features_reshaped)
    pred_class = int(pred[0])
    accent_conf = float(np.max(pred_proba)*100)

    logging.info(f"Accent prediction: class={pred_class} proba={pred_proba.tolist()}")

    speaker_info = get_speaker_info(pred_class)

    age, age_conf = estimate_age(features, f0=f0_mean)
    sex, sex_conf = estimate_sex(features, f0=f0_mean)
    accent = speaker_info['accent'] if speaker_info else "Unknown"
    country = speaker_info['country'] if speaker_info else "Unknown"

    response = {
        'accent': accent,
        'age': age,
        'sex': sex,
        'confidence': round(accent_conf,2),
        'accent_confidence': round(accent_conf,2),
        'age_confidence': round(age_conf,2),
        'sex_confidence': round(sex_conf,2),
        'country': country,
        'predicted_class': pred_class,
        'native_language': speaker_info['native_language'] if speaker_info else None
    }
    return jsonify(response),200


@app.route('/debug_features', methods=['POST'])
def debug_features():
    if 'audio' not in request.files:
        return jsonify({'error':'No audio file'}),400
    file = request.files['audio']
    if file.filename=='': return jsonify({'error':'No file selected'}),400
    filename = secure_filename(file.filename)
    temp_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(temp_path)
    wav_path = convert_to_wav(temp_path)
    extracted = extract_features(wav_path)
    if isinstance(extracted, tuple):
        features, f0_mean = extracted
    else:
        features = extracted
        f0_mean = None
    try: os.remove(temp_path)
    except: pass
    if wav_path != temp_path:
        try: os.remove(wav_path)
        except: pass
    if features is None:
        return jsonify({'error':'Failed to extract features'}),500
    # Return basic info for debugging (include f0)
    return jsonify({'shape': int(features.shape[0]), 'features': np.round(features,4).tolist(), 'f0': f0_mean}),200

@app.route('/test', methods=['GET'])
def test(): return jsonify({'message':'Backend working!'}),200

@app.route('/accents', methods=['GET'])
def get_accents():
    if speakers_df is None: return jsonify({'error':'Speakers data not loaded'}),500
    accents=[]
    for lang,count in speakers_df['native_language'].value_counts().items():
        accents.append({'name':get_accent_info(lang),'count':int(count),'code':lang})
    return jsonify({'accents':accents}),200

if __name__=='__main__':
    load_model()
    print("Server ready at http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)