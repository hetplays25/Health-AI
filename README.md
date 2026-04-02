# 🏥 Health AI: Disease Prediction System

A machine learning-powered web application that predicts potential health conditions based on user-inputted symptoms. Built with **Flask** and **Scikit-Learn**, this project demonstrates a full end-to-end ML pipeline from data processing to cloud deployment.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Render](https://img.shields.io/badge/Render-%46E3B7.svg?style=for-the-badge&logo=render&logoColor=white)

## 🚀 Features
- **Predictive Modeling:** Uses a Random Forest classifier (or your specific model) to analyze health data.
- **Modern UI:** Responsive, glassmorphism-inspired frontend built with Tailwind CSS.
- **Instant Results:** Real-time prediction feedback via a Flask backend.
- **Scalable Deployment:** Hosted on Render with automated CI/CD via GitHub.

## 🛠️ Project Structure
```text
health-ai/
├── app.py              # Flask Application (Backend)
├── model.pkl           # Trained ML Model
├── label_encoder.pkl   # Data Preprocessing Encoder
├── health_dataset.csv  # Training Data
├── requirements.txt    # Python Dependencies
├── templates/          # Frontend HTML files
│   └── index.html      # UI Template
└── train_model.py      # Model training script
