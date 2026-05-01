# Indian Sign Language → Text & Audio

> AI-powered gesture recognition system using deep learning and text-to-speech.

## Features
- 🤟 **Real-time ISL gesture recognition** (A–Z, 0–9) via webcam
- 🧠 **MLP deep learning model** trained on 63-dim MediaPipe hand landmarks
- 📝 **Sentence builder** with hold-to-confirm logic
- 🔊 **Text-to-Speech** output via gTTS
- 🎨 **Premium dark Streamlit UI**

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python model/train_model.py
```

### 3. Run the app
```bash
streamlit run app.py
```

## Project Structure
```
├── app.py                  # Main Streamlit application
├── requirements.txt
├── utils/
│   ├── landmark_utils.py   # MediaPipe hand landmark extraction
│   ├── gesture_buffer.py   # Temporal smoothing & sentence builder
│   └── tts_utils.py        # Text-to-speech helpers
├── data/
│   ├── generate_dataset.py # Synthetic ISL dataset generator
│   └── isl_keypoints.csv   # Generated dataset (after running train)
└── model/
    ├── train_model.py      # Model training script
    ├── isl_model.pkl       # Trained MLP model
    ├── label_encoder.pkl   # Class label encoder
    └── scaler.pkl          # Feature scaler
```

## How It Works
1. **Camera captures** a frame of your hand sign
2. **MediaPipe** detects 21 hand landmarks (63 features)
3. **MLP classifier** predicts the ISL letter/digit
4. **Hold-to-confirm**: hold the same sign for N frames to add it to the sentence
5. **gTTS** converts the accumulated sentence to spoken audio

## Tech Stack
| Component | Technology |
|---|---|
| UI | Streamlit |
| Hand Detection | MediaPipe Hands |
| ML Model | scikit-learn MLPClassifier |
| TTS | gTTS (Google Text-to-Speech) |
| Vision | OpenCV |
