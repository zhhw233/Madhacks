# Fish Audio Integration (removed)

This project previously included optional integration with Fish Audio for voice cloning and synthesis. That integration has been removed — the app now focuses on local audio feature extraction and accent/age/sex prediction using the included models.

What remains in this repository
- Local prediction endpoints: `/predict` and `/debug_features` (these use local ML models and feature extraction).
- The frontend live transcription uses the browser Web Speech API (client-side).

If you previously followed instructions to configure Fish Audio, you can ignore them — voice cloning and TTS endpoints are no longer available in this server.

How to run the server and try predictions
1. Start the server:
```bash
cd "Voice Predictor"
python3 -m venv venv  # optional
source venv/bin/activate
pip install -r requirements.txt 2>/dev/null || true
pip install flask flask-cors librosa numpy pandas xgboost requests
python3 flaskserver.py
```

2. Test endpoints:
```bash
curl http://localhost:5001/test
curl -X POST -F "audio=@/path/to/sample.wav" http://localhost:5001/debug_features
curl -X POST -F "audio=@/path/to/sample.wav" http://localhost:5001/predict
```

If you want to re-enable Fish Audio features later, see the Fish Audio docs:
https://docs.fish.audio/developer-guide/getting-started/introduction

For now the repository focuses on improving local accent detection and UI experience.
