# Spam Classifier API

This project is a **Spam Classifier API** built with **FastAPI**.  
It uses a trained machine learning model (scikit-learn) to classify text messages as **spam** or **ham**.

## Features
- REST API built with FastAPI 🚀
- Preprocessing with NLTK (stopwords removal, stemming, tokenization)
- Machine Learning model trained and loaded via pickle
- HTML frontend (Jinja2 templates)
- Deployed on Render (Free Hosting)

## Hosted URL
👉 The project is live at: [https://spam-classifier-s5p5.onrender.com/](https://spam-classifier-s5p5.onrender.com/)

## Project Structure
```
.
├── main.py                # FastAPI app
├── vectorizer.pkl         # Pickle file for vectorizer
├── spam_classifier.pkl    # Pickle file for ML model
├── templates/
│   └── index.html         # Frontend HTML page
├── requirements.txt       # Dependencies
├── Procfile               # Deployment start command
└── README.txt             # Project documentation
```

## Local Setup
1. Clone the repo or download the files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run locally:
   ```bash
   uvicorn main:app --reload
   ```
4. Visit: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Deployment (Render)
1. Push code to GitHub
2. Connect repo to [Render](https://render.com/)
3. Set **Start Command** as:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 10000
   ```
4. Deploy 🚀

---
Made with ❤️ using FastAPI & scikit-learn
