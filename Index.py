import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

dataset = pd.read_csv("emails.csv")

email_column = 'text'
label_column = 'spam'

dataset.dropna(subset=[email_column, label_column], inplace=True)

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(dataset[email_column]).toarray()
y = dataset[label_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

new_email = ["Congrats, You won Lottery!"]
new_email_tfidf = tfidf.transform(new_email)
prediction = model.predict(new_email_tfidf)
print("Prediction:", "Spam" if prediction[0] == 1 else "Not Spam")

# Output:
