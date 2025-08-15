import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("IMDB Dataset.csv")

df['review'] = df['review'].apply(lambda x: re.sub("<.*?>", "", x))
df['review'] = df['review'].apply(lambda x: re.sub(r'[^\w\s]', "", x))

stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

df['review'] = df['review'].apply(word_tokenize)
df['review'] = df['review'].apply(lambda x: [word for word in x if word.lower() not in stop_words])
df['review'] = df['review'].apply(lambda x: [lemma.lemmatize(word) for word in x])

X = df['review'].apply(lambda x: " ".join(x))
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#saving RAM
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

class SparseDataset(Dataset):
    def __init__(self, X_sparse, y):
        self.X = X_sparse
        self.y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        row = self.X[idx].toarray().squeeze(0)
        return torch.tensor(row, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

train_dataset = SparseDataset(X_train_tfidf, y_train_enc)
test_dataset = SparseDataset(X_test_tfidf, y_test_enc)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

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


input_dim = X_train_tfidf.shape[1]
model = FeedforwardNN(input_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 4
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")


from sklearn.metrics import classification_report

model.eval()
y_pred = []
with torch.no_grad():
    for xb, _ in test_loader:
        outputs = model(xb)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.numpy())

acc = accuracy_score(y_test_enc, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%\n")


print("Classification Report:")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

torch.save(model.state_dict(), "pytorch_model.pth")
joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(le, "label_encoder.pkl")
