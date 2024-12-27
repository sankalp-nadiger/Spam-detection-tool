#reading and labelling data
import pandas as pd
df=pd.read_csv(r'C:\Users\ADMIN\Downloads\spam.csv',encoding='latin-1')
df=df[['v1','v2']]
df.columns=['category','text']
df['category'] = df['category'].map({'ham': 0, 'spam': 1})

#data cleaning
pip install nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()             # Convert to lowercase
    text = text.split()             # Tokenize
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

df['processed_text'] = df['text'].apply(preprocess_text)

#feature extraction
pip install scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=3000)  # Use the top 3000 features
X = tfidf.fit_transform(df['processed_text']).toarray()
y = df['category']

#training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import pickle
with open('spam_detector.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('spam_detector.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    loaded_tfidf = pickle.load(f)

sample_text = input("Enter text: ");
sample_text = preprocess_text(sample_text)
sample_vector = loaded_tfidf.transform([sample_text]).toarray()
prediction = loaded_model.predict(sample_vector)
print("Spam" if prediction[0] == 1 else "Not Spam")
