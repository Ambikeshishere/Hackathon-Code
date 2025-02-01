import imaplib
import email
import re
import pandas as pd
from email.header import decode_header
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# Email Credentials (use environment variables for security)
EMAIL = "abhay2004raj15@gmail.com"
PASSWORD = "uvgc jdra muxw bjdo"  # Use an App Password if 2FA is enabled
IMAP_SERVER = "imap.gmail.com" # Change for other providers (e.g., Outlook, Yahoo)

# Load dataset (ensure emails.csv is in the correct location)
dataset = pd.read_csv("emails.csv")

email_column = 'text'
label_column = 'spam'
dataset.dropna(subset=[email_column, label_column], inplace=True)

# List of trusted email providers (removing duplicates for simplicity)
trusted_providers = set([
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com",
    "pintrest.com", "linkedin.com", "protonmail.com", "zoho.com", "aol.com",
    "gmx.com", "mail.com", "yandex.com", "fastmail.com", "tutanota.com", "google.com"
])

# Function to extract sender's email
def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(dataset[email_column]).toarray()
y = dataset[label_column]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# --- Connect to Email Server ---
try:
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL, PASSWORD)
    mail.select("inbox")

    # Search for all emails (you could refine this to filter based on other criteria)
    status, messages = mail.search(None, "ALL")
    if status != "OK":
        print("No emails found.")
        exit()

    messages = messages[0].split()

    # Fetch only the last 30 emails
    recent_messages = messages[-30:]  # Last 30 emails

    for msg_num in recent_messages:
        try:
            status, msg_data = mail.fetch(msg_num, "(RFC822)")
            if status != "OK":
                print(f"Failed to fetch email {msg_num}.")
                continue

            raw_email = msg_data[0][1]

            # Parse email content
            msg = email.message_from_bytes(raw_email)
            sender = msg["From"]
            subject = decode_header(msg["Subject"])[0][0]
            date_str = msg["Date"]

            if isinstance(subject, bytes):
                subject = subject.decode(errors="ignore")

            # Extract sender email
            sender_email = extract_email(sender)

            print(f"\n📩 New Email from: {sender_email}")
            print(f"📌 Subject: {subject}")

            # Extract email body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    if "attachment" not in content_disposition and content_type == "text/plain":
                        try:
                            body = part.get_payload(decode=True).decode(errors="ignore")
                            break
                        except Exception as e:
                            print(f"⚠️ Error decoding email body: {e}")
            else:
                try:
                    body = msg.get_payload(decode=True).decode(errors="ignore")
                except Exception as e:
                    print(f"⚠️ Error decoding email body: {e}")

            print(f"📄 Email Content: {body[:200]}...")  # Show first 200 characters

            # Predict spam
            email_tfidf = tfidf.transform([body])
            prediction = model.predict(email_tfidf)
            prediction_label = "Spam" if prediction[0] == 1 else "Not Spam"
            print(f"🔍 Prediction: {prediction_label}")

            # --- Block Untrusted Email IDs ---
            if sender_email:
                domain = sender_email.split("@")[-1]

                if prediction_label == "Spam" and domain not in trusted_providers:
                    # Save to blocklist
                    with open("blocked_emails.txt", "a") as file:
                        file.write(sender_email + "\n")
                    print(f"🚫 Blocked Email ID: {sender_email} (Untrusted domain: {domain})")

            # Mark email as read if spam
            if prediction_label == "Spam":
                mail.store(msg_num, "+FLAGS", "\\Seen")

        except Exception as e:
            print(f"⚠️ Error processing email {msg_num}: {e}")

finally:
    # Safely close connection
    try:
        mail.logout()
    except Exception as e:
        print(f"⚠️ Error logging out: {e}")
