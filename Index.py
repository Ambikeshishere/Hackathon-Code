import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
dataset = pd.read_csv("emails.csv")

email_column = 'text'
label_column = 'spam'

# Drop rows with missing values
dataset.dropna(subset=[email_column, label_column], inplace=True)

# Initialize the TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(dataset[email_column]).toarray()
y = dataset[label_column]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# New email to be classified
new_email = ["Congrats, You won Lottery!"]
new_email_tfidf = tfidf.transform(new_email)
prediction = model.predict(new_email_tfidf)

# Print prediction for the new email
prediction_label = "Spam" if prediction[0] == 1 else "Not Spam"
print(f"Prediction: {prediction_label}")

# User feedback: Ask if the new email is spam or not
user_feedback = input(f"Is the email '{new_email[0]}' correctly classified as '{prediction_label}'? (y/n): ")

if user_feedback.lower() == 'n':
    correct_label = input("Please provide the correct label for this email (Spam/Not Spam): ").lower()
    # If the new email doesn't exist in the dataset, add it
    if new_email[0] not in dataset[email_column].values:
        new_data = pd.DataFrame({email_column: [new_email[0]], label_column: [1 if correct_label == 'spam' else 0]})
        dataset = pd.concat([dataset, new_data], ignore_index=True)
        dataset.to_csv("emails.csv", index=False)  # Save the updated dataset
        print("New email added to dataset.")
    else:
        print("Email already exists in the dataset.")

# Retrain the model with the updated dataset (if any new email was added)
X = tfidf.fit_transform(dataset[email_column]).toarray()
y = dataset[label_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
