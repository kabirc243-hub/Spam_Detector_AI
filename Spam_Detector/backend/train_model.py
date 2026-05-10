import re
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)



df = pd.read_csv(
    "SMSSpamCollection",
    sep='\t',
    header=None,
    names=['label', 'message']
)



def clean_text(text):

  
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' URL ', text)

    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)


    text = re.sub(r'[^\w\s]', '', text)


    text = re.sub(r'\s+', ' ', text).strip()

    return text


df['message'] = df['message'].apply(clean_text)



df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})



X = df['message']
y = df['label']



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 3),
    max_features=15000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = LogisticRegression(
    max_iter=2000,
    class_weight='balanced'
)

model.fit(X_train_vec, y_train)



y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)

print("\n=========================")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("=========================\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))



pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully.")