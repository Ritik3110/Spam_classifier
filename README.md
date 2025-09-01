# Spam Classifier API 🚀

A FastAPI-based web app that classifies text messages as **spam** or **ham** using an NLP pipeline with NLTK, TF-IDF vectorization, and a trained ML model.

---

## Features
- REST API with FastAPI
- Preprocessing using NLTK (tokenization, stopwords, stemming)
- ML model for spam classification
- HTML frontend with Jinja2 templates
- Ready for deployment on **Render** or similar platforms

---

## Requirements

Create a virtual environment (recommended) and install:

```bash
pip install -r requirements.txt
```

If you don’t have `requirements.txt`, here’s a working one:

```
fastapi==0.110.0
uvicorn==0.29.0
pydantic==1.10.14
scikit-learn==1.3.2
numpy==1.26.4
nltk==3.8.1
jinja2==3.1.3
```

---

## Run Locally

1. Clone the repo:
```bash
git clone https://github.com/yourusername/spam-classifier.git
cd spam-classifier
```

2. Run with Uvicorn:
```bash
uvicorn app:app --reload
```
> ⚠️ Replace `app:app` with `main:app` if your entry file is named `main.py`.

3. Visit in browser:
```
http://127.0.0.1:8000
```

---

## API Endpoints

### **GET /**
Returns the homepage (`index.html`).

### **POST /predict**
Classify input text.

**Request body:**
```json
{
  "text": "Congratulations! You have won a lottery."
}
```

**Response:**
```json
{
  "prediction": "spam"
}
```

---

## Deploy on Render

1. Push this repo to GitHub.
2. Create a new **Web Service** on [Render](https://render.com).
3. Fill in settings:
   - **Environment:** `Python 3`
   - **Build Command:**  
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command:**  
     ```bash
     uvicorn app:app --host 0.0.0.0 --port 10000
     ```
     > ⚠️ Change `app:app` to `main:app` if your main file is named `main.py`.

4. Make sure your repo has this structure:

```
.
├── app.py (or main.py)        # FastAPI app
├── vectorizer.pkl             # Saved TF-IDF vectorizer
├── spam_classifier.pkl        # Trained model
├── templates/
│   └── index.html             # Frontend template
├── requirements.txt
└── README.md
```
