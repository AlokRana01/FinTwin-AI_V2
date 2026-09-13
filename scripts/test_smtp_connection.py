"""
scripts/test_smtp_connection.py
===============================
Diagnostic script to test SMTP authentication with the credentials in .env.
"""

import os
import sys
import smtplib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from database.connection import _load_dotenv
_load_dotenv()

from utils.email_service import get_smtp_config

def run_smtp_check():
    cfg = get_smtp_config()
    print("Testing SMTP Connection...")
    print(f"Host     : {cfg['host']}")
    print(f"Port     : {cfg['port']}")
    print(f"User     : {cfg['username']}")
    print(f"From     : {cfg['from_email']}")
    
    password = cfg['password']
    # Gmail app passwords can contain spaces, smtplib accepts them or stripped
    
    try:
        if cfg['port'] == 465:
            server = smtplib.SMTP_SSL(cfg['host'], cfg['port'], timeout=10)
        else:
            server = smtplib.SMTP(cfg['host'], cfg['port'], timeout=10)
            server.ehlo()
            server.starttls()
            server.ehlo()
        
        server.login(cfg['username'], password)
        print("[SUCCESS] SMTP Authentication Successful!")
        server.quit()
        return True
    except smtplib.SMTPAuthenticationError as e:
        print(f"[AUTH ERROR] Authentication failed: {e}")
        print("Note: For Gmail, ensure you are using a 16-character Google App Password.")
        return False
    except Exception as e:
        print(f"[ERROR] Connection error: {e}")
        return False

if __name__ == "__main__":
    run_smtp_check()
