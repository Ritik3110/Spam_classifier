import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import os

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)

# Ensure nltk resources are available
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)

# Preload stopwords & stemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Load vectorizer & model once at startup
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('spam_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Input schema
class TextInput(BaseModel):
    text: str

# Preprocessing function
def transform_text(text: str) -> str:
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [i for i in text if i not in stop_words and i not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

templates = Jinja2Templates(directory="Templates")

@app.get("/predict")
async def predict_get(text: str):
    try:
        text_transformed = transform_text(text)
        text_vectorized = vectorizer.transform([text_transformed])
        prediction = model.predict(text_vectorized)[0]
        result = "spam" if prediction == 1 else "ham"
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}


# Home route -> HTML
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Health-check -> for JS status function
@app.get("/api")
async def health_check():
    return {"status": "ok", "message": "API is running"}

# Predict endpoint
@app.post("/api/predict")
async def predict(input_data: TextInput):
    try:
        # Transform text (assuming you have transform_text function)
        text_transformed = transform_text(input_data.text)
        text_vectorized = vectorizer.transform([text_transformed])

        # Get probabilities
        probabilities = model.predict_proba(text_vectorized)[0]
        spam_prob = float(probabilities[1])
        not_spam_prob = float(probabilities[0])

        is_spam = spam_prob > 0.5
        confidence = spam_prob if is_spam else not_spam_prob

        return {
            "is_spam": is_spam,
            "confidence": confidence,
            "probabilities": {
                "not_spam": not_spam_prob,
                "spam": spam_prob
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

 