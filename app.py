import streamlit as st
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn
import joblib

tfidf_vectorizer = joblib.load("tfidf.pkl")
label_encoder = joblib.load("label_encoder.pkl")

class FeedforwardNN(nn.Module):
    def __init__(self, input_dim):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 2)  

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)  
        return x

input_dim = len(tfidf_vectorizer.get_feature_names_out())
model = FeedforwardNN(input_dim)
model.load_state_dict(torch.load("pytorch_model.pth", map_location=torch.device('cpu')))
model.eval()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub("<.*?>", "", text)  
    text = re.sub(r'[^\w\s]', '', text)  
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def predict_sentiment(review):
    processed_review = preprocess_text(review)
    tfidf_vector = tfidf_vectorizer.transform([processed_review])
    tensor_input = torch.tensor(tfidf_vector.toarray(), dtype=torch.float32)
    with torch.no_grad():
        output = model(tensor_input)
        predicted_class = torch.argmax(output, dim=1).item()
    sentiment_label = label_encoder.inverse_transform([predicted_class])[0]
    return sentiment_label

st.title("Sentiment Analysis (PyTorch)")
review_input = st.text_area("Enter your movie review:")

if st.button("Predict Sentiment"):
    if review_input.strip():
        sentiment = predict_sentiment(review_input)
        st.write("Predicted Sentiment:", sentiment)
    else:
        st.write("Please enter a review for prediction.")
