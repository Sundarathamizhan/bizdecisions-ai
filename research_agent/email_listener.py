import imaplib
import email
import time
import os
import subprocess
from dotenv import load_dotenv
from email_sender import send_email
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

IMAP_SERVER = "imap.gmail.com"

executor = ThreadPoolExecutor(max_workers=3)

print("🤖 Email Research Agent Started...")
print("Waiting for research requests...\n")


def process_request(topic, sender):

    print(f"🚀 Processing request from {sender}")
    print(f"📚 Topic: {topic}")

    try:

        subprocess.run(
            [".venv\\Scripts\\python.exe", "research_agent.py", topic, sender],
            check=True
        )

        send_email(
            sender,
            "Your AI Research Report",
            f"Here is the research report for: {topic}",
            "outputs/research_report.pdf"
        )

        print(f"✅ Report sent to {sender}\n")

    except Exception as e:
        print("❌ Error processing request:", e)


def check_inbox():

    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    mail.select("inbox")

    status, messages = mail.search(None, '(UNSEEN SUBJECT "Research:")')

    email_ids = messages[0].split()

    for mail_id in email_ids:

        status, msg_data = mail.fetch(mail_id, "(RFC822)")
        raw_email = msg_data[0][1]

        msg = email.message_from_bytes(raw_email)

        subject = msg["subject"]
        sender = email.utils.parseaddr(msg["from"])[1]

        if not subject:
            continue

        if not subject.lower().startswith("research:"):
            continue

        topic = subject.replace("Research:", "").strip()

        print(f"📩 New research request from {sender}")
        print(f"🔎 Topic: {topic}\n")

        # submit task to thread pool
        executor.submit(process_request, topic, sender)

    mail.logout()


# Continuous listener
while True:

    try:
        check_inbox()
    except Exception as e:
        print("⚠ Error checking inbox:", e)

    time.sleep(30)