import pickle
import nltk
import string
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
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

# GET route (health check / welcome)
@app.get("/")
async def home():
    return {"message": "Spam Classifier API is running 🚀. Use POST /predict with {text: 'your message'}"}

# POST route (prediction)
@app.post("/predict")
async def predict(input_data: TextInput):
    try:
        # Transform & vectorize
        text_transformed = transform_text(input_data.text)
        text_vectorized = vectorizer.transform([text_transformed])

        # Predict
        prediction = model.predict(text_vectorized)[0]
        result = "spam" if prediction == 1 else "ham"

        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}
