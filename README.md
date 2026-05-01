# Indian-Sign-Language-Text-Audio
Deep learning-based system for converting Indian Sign Language gestures into text and audio using computer vision and CNN models.
# 🤟 Indian Sign Language to Text & Audio Conversion using Deep Learning

## 📌 Overview

This project focuses on converting Indian Sign Language (ISL) gestures into readable text and corresponding audio output using deep learning techniques. The system aims to bridge the communication gap between hearing/speech-impaired individuals and others by providing a real-time interpretation solution.

## 🚀 Features

* Real-time gesture recognition from image/video input
* Conversion of hand gestures into text
* Text-to-speech (TTS) audio output
* Deep learning-based classification model
* End-to-end pipeline from input to output

## 🧠 Methodology

* Collected and used ISL image dataset for training
* Preprocessed images (resizing, normalization)
* Built a Convolutional Neural Network (CNN) for gesture classification
* Mapped predicted classes to alphabets/words
* Integrated Text-to-Speech module for audio output

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy, Pandas
* Matplotlib
* pyttsx3 / gTTS

## 📂 Project Structure

├── dataset/
├── models/
├── notebooks/
├── app.py
├── requirements.txt
└── README.md

## 📊 Model Performance

* Achieved ~85–92% accuracy on ISL dataset
* Optimized using data augmentation and tuning

## ▶️ How to Run

1. Clone the repository
   git clone https://github.com/your-username/isl-project

2. Install dependencies
   pip install -r requirements.txt

3. Run the application
   python app.py

## 🔍 Use Case

* Assistive technology for hearing and speech-impaired individuals
* Real-time communication support system
* Educational tools for learning sign language

## 🚀 Future Improvements

* Real-time video streaming support
* Sentence-level recognition
* Mobile application deployment
* Integration with chat systems

## 👨‍💻 Author

Padamati Tarun Krishna
AI/ML Engineer (Aspiring)

## 📌 Note

This project was independently developed as part of academic and research work focusing on AI-based accessibility solutions.
