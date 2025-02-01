import imaplib
import email
import re
import pandas as pd
from datetime import datetime, timedelta, timezone
from email.header import decode_header
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# Email Credentials (use environment variables for security)
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("EMAIL_PASSWORD")
IMAP_SERVER = "imap.gmail.com"

# Load dataset (ensure emails.csv is in the correct location)
dataset = pd.read_csv("emails.csv")

email_column = 'text'
label_column = 'spam'
dataset.dropna(subset=[email_column, label_column], inplace=True)

# List of trusted email providers (removing duplicates for simplicity)
trusted_providers = set([
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com",
    "pintrest.com", "linkedin.com", "protonmail.com", "zoho.com", "aol.com",
    "gmx.com", "mail.com", "yandex.com", "fastmail.com", "tutanota.com"
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

    # Get today's date in IMAP format
    today = datetime.now().strftime("%d-%b-%Y")

    # Search for emails received today
    status, messages = mail.search(None, f'(SINCE {today})')

    messages = messages[0].split()

    # Get current time as timezone-aware datetime
    now = datetime.now(timezone.utc)  # Ensure it's timezone-aware
    one_hour_ago = now - timedelta(hours=1)

    for msg_num in messages:
        status, msg_data = mail.fetch(msg_num, "(RFC822)")
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

        # Convert email date to timezone-aware datetime
        email_date = email.utils.parsedate_to_datetime(date_str)

        if email_date.tzinfo is None:  # Convert to UTC if naive
            email_date = email_date.replace(tzinfo=timezone.utc)

        # Check if the email is from the last 1 hour
        if email_date < one_hour_ago:
            continue  # Skip this email
        
        print(f"\n📩 New Email from: {sender_email} (Received: {email_date})")
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

finally:
    # Close connection safely
    mail.logout()
