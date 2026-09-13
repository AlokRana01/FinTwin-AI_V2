"""
utils/email_service.py
======================
Secure, professional SMTP email service for FinTwin AI.

Handles:
  - Account registration confirmation & email verification
  - Secure password reset requests
  - Resending verification emails

Security & Design:
  - Responsive, fintech dark-themed HTML email templates matching FinTwin AI UI
  - Inline CSS for maximum email client compatibility (Gmail, Outlook, Apple Mail)
  - SMTP credentials and URLs loaded strictly from environment variables
  - Zero sensitive information exposure in logs or user-facing messages
"""

import os
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from html import escape as _he
from typing import Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Base Paths & Config Defaults
_BASE_DIR = Path(__file__).resolve().parent.parent


def get_smtp_config() -> dict:
    """Loads and returns current SMTP configuration from environment."""
    return {
        "host": os.environ.get("SMTP_HOST", "").strip(),
        "port": int(os.environ.get("SMTP_PORT", "587")),
        "username": os.environ.get("SMTP_USERNAME", "").strip(),
        "password": os.environ.get("SMTP_PASSWORD", "").strip(),
        "from_email": os.environ.get("SMTP_FROM_EMAIL", "noreply@fintwin.app").strip(),
        "from_name": os.environ.get("SMTP_FROM_NAME", "FinTwin AI").strip(),
        "base_url": os.environ.get("APP_BASE_URL", "http://localhost:8501").rstrip("/"),
    }


def is_smtp_configured() -> bool:
    """Checks whether SMTP delivery has been configured with host and username."""
    cfg = get_smtp_config()
    return bool(cfg["host"] and cfg["username"] and cfg["password"])


def _build_html_email(
    title: str,
    preheader: str,
    content_html: str,
    action_url: Optional[str] = None,
    action_text: Optional[str] = None,
    security_notice: Optional[str] = None,
) -> str:
    """
    Renders a responsive, fintech-styled dark HTML email matching FinTwin AI's design language.
    Colors: Background #0B1220, Card #111827, Accent #4F8CFF / #00D4FF, Text #F8FAFC / #94A3B8.
    """
    action_button_html = ""
    if action_url and action_text:
        action_button_html = f"""
        <table role="presentation" border="0" cellpadding="0" cellspacing="0" style="margin: 28px 0;">
            <tr>
                <td align="center" style="border-radius: 8px; background: linear-gradient(135deg, #4F8CFF 0%, #00D4FF 100%);">
                    <a href="{action_url}" target="fintwin_app" style="font-family: 'Plus Jakarta Sans', 'Segoe UI', Arial, sans-serif; font-size: 15px; font-weight: 700; color: #FFFFFF; text-decoration: none; padding: 14px 32px; display: inline-block; border-radius: 8px; letter-spacing: 0.01em;">
                        {_he(action_text)} &rarr;
                    </a>
                </td>
            </tr>
        </table>
        <p style="margin: 0 0 16px 0; font-size: 12px; color: #64748B; word-break: break-all; line-height: 1.5;">
            Or copy and paste this link into your browser:<br>
            <a href="{action_url}" target="fintwin_app" style="color: #4F8CFF; text-decoration: underline;">{action_url}</a>
        </p>
        """

    notice_box_html = ""
    if security_notice:
        notice_box_html = f"""
        <div style="background: rgba(79, 140, 255, 0.08); border-left: 3px solid #4F8CFF; border-radius: 4px; padding: 14px 16px; margin: 24px 0 12px 0;">
            <p style="margin: 0; font-size: 13px; line-height: 1.5; color: #CBD5E1;">
                <strong style="color: #4F8CFF;">Security Notice:</strong> {security_notice}
            </p>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_he(title)}</title>
    <!--[if mso]>
    <noscript>
        <xml>
            <o:OfficeDocumentSettings>
                <o:PixelsPerInch>96</o:PixelsPerInch>
            </o:OfficeDocumentSettings>
        </xml>
    </noscript>
    <![endif]-->
</head>
<body style="margin: 0; padding: 0; background-color: #0B1220; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #F8FAFC; -webkit-font-smoothing: antialiased;">
    <div style="display: none; font-size: 1px; color: #0B1220; line-height: 1px; max-height: 0px; max-width: 0px; opacity: 0; overflow: hidden;">
        {_he(preheader)}
    </div>
    <table role="presentation" width="100%" border="0" cellpadding="0" cellspacing="0" style="background-color: #0B1220; width: 100%; padding: 40px 10px;">
        <tr>
            <td align="center">
                <!-- Main Container -->
                <table role="presentation" width="100%" border="0" cellpadding="0" cellspacing="0" style="max-width: 600px; background-color: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; overflow: hidden; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);">
                    
                    <!-- Header with FinTwin AI Branding -->
                    <tr>
                        <td style="padding: 32px 36px 24px 36px; background: linear-gradient(180deg, rgba(79, 140, 255, 0.12) 0%, rgba(17, 24, 39, 0) 100%); border-bottom: 1px solid rgba(255, 255, 255, 0.06);">
                            <table role="presentation" width="100%" border="0" cellpadding="0" cellspacing="0">
                                <tr>
                                    <td>
                                        <div style="font-family: 'Plus Jakarta Sans', 'Segoe UI', Arial, sans-serif; font-size: 26px; font-weight: 800; letter-spacing: -0.02em; color: #F8FAFC;">
                                            Fin<span style="color: #4F8CFF;">Twin</span> <span style="color: #00D4FF; font-weight: 300;">AI</span>
                                        </div>
                                        <div style="display: inline-block; background: rgba(79, 140, 255, 0.15); border: 1px solid rgba(79, 140, 255, 0.3); border-radius: 4px; padding: 2px 8px; margin-top: 6px; font-size: 10px; font-weight: 700; color: #4F8CFF; letter-spacing: 0.1em; text-transform: uppercase;">
                                            AI-Powered Financial Health & Planning
                                        </div>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- Body Content -->
                    <tr>
                        <td style="padding: 36px 36px 28px 36px;">
                            <h1 style="margin: 0 0 16px 0; font-family: 'Plus Jakarta Sans', 'Segoe UI', Arial, sans-serif; font-size: 22px; font-weight: 700; color: #F8FAFC; letter-spacing: -0.02em; line-height: 1.3;">
                                {_he(title)}
                            </h1>
                            
                            <div style="font-size: 15px; line-height: 1.6; color: #CBD5E1;">
                                {content_html}
                            </div>

                            {action_button_html}
                            {notice_box_html}
                        </td>
                    </tr>

                    <!-- About & Mission Section -->
                    <tr>
                        <td style="padding: 20px 36px; background-color: #0E1626; border-top: 1px solid rgba(255, 255, 255, 0.04); font-size: 12px; line-height: 1.6; color: #94A3B8;">
                            <strong style="color: #CBD5E1;">About FinTwin AI:</strong> FinTwin AI is an AI-powered financial health platform designed to help users understand their financial health, analyze financial behavior, plan financial goals, simulate financial scenarios, forecast financial outcomes, and receive personalized financial guidance.
                        </td>
                    </tr>

                    <!-- Footer -->
                    <tr>
                        <td style="padding: 24px 36px; background-color: #080D17; border-top: 1px solid rgba(255, 255, 255, 0.04); text-align: center; font-size: 11px; line-height: 1.6; color: #64748B;">
                            <p style="margin: 0 0 6px 0;">
                                &copy; FinTwin AI &bull; AI-Powered Financial Health & Planning
                            </p>
                            <p style="margin: 0;">
                                Please keep your account credentials secure. Never share your password with anyone. FinTwin AI will never ask you to send your password by email.
                            </p>
                        </td>
                    </tr>

                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""


def _send_smtp_email(to_email: str, subject: str, html_body: str, text_body: str) -> Tuple[bool, str]:
    """
    Sends an email via SMTP.
    Returns (success: bool, safe_message: str).
    Guarantees zero leakage of SMTP passwords or internal exceptions to the user.
    """
    cfg = get_smtp_config()

    if not is_smtp_configured():
        logger.warning(
            f"SMTP not configured (host='{cfg['host']}', user='{cfg['username']}'). "
            f"Simulating email delivery for: {to_email}"
        )
        # Development mode simulation: return success so local devs can test flows
        return True, "Email queued for delivery (Development Simulation Mode)."

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{cfg['from_name']} <{cfg['from_email']}>"
    msg["To"] = to_email

    # Attach plain text and HTML alternatives
    part_text = MIMEText(text_body, "plain", "utf-8")
    part_html = MIMEText(html_body, "html", "utf-8")
    msg.attach(part_text)
    msg.attach(part_html)

    try:
        if cfg["port"] == 465:
            # SSL
            server = smtplib.SMTP_SSL(cfg["host"], cfg["port"], timeout=10)
        else:
            # STARTTLS (e.g. port 587 or 25)
            server = smtplib.SMTP(cfg["host"], cfg["port"], timeout=10)
            server.ehlo()
            server.starttls()
            server.ehlo()

        if cfg["username"] and cfg["password"]:
            server.login(cfg["username"], cfg["password"])

        server.sendmail(cfg["from_email"], [to_email], msg.as_string())
        server.quit()
        logger.info(f"Email sent successfully to {to_email} (Subject: '{subject}')")
        return True, "Email sent successfully."

    except Exception as e:
        # Secure logging: Log technical error without logging passwords/tokens
        logger.error(f"SMTP delivery failure to {to_email}: {type(e).__name__} - {e}")
        return False, "Could not send email due to an SMTP service issue. Please try again later."


def send_verification_email(name: str, email: str, user_id: str, raw_token: str) -> Tuple[bool, str]:
    """
    Generates and sends the official FinTwin AI Registration Confirmation & Email Verification email.
    """
    cfg = get_smtp_config()
    base_url = cfg["base_url"]
    verification_url = f"{base_url}/?verify_email_token={raw_token}"

    title = "Congratulations! Your FinTwin AI account has been created"
    preheader = f"Verify your email to activate your FinTwin AI digital twin, {name}."

    content_html = f"""
    <p style="margin: 0 0 16px 0;">Hi <strong>{_he(name)}</strong>,</p>
    <p style="margin: 0 0 16px 0;">
        Congratulations! Your FinTwin AI account has been successfully created. You are one step away from unlocking personalized financial health insights, AI-powered goal planning, and real-time scenario simulation.
    </p>
    
    <div style="background-color: #1F2937; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 16px 20px; margin: 20px 0;">
        <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8; margin-bottom: 8px; font-weight: 700;">Account Information</div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="color: #94A3B8;">Full Name:</span>
            <strong style="color: #F8FAFC;">{_he(name)}</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="color: #94A3B8;">User ID:</span>
            <code style="color: #00D4FF; background: #111827; padding: 2px 6px; border-radius: 4px;">{_he(user_id)}</code>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #94A3B8;">Registered Email:</span>
            <strong style="color: #F8FAFC;">{_he(email)}</strong>
        </div>
    </div>

    <p style="margin: 0 0 12px 0;">
        Please verify your email address to activate your account and access your Financial Digital Twin. This verification link will expire in <strong>30 minutes</strong>.
    </p>
    """

    text_body = f"""Hi {name},

Congratulations! Your FinTwin AI account has been successfully created.

Account Information:
- Full Name: {name}
- User ID: {user_id}
- Registered Email: {email}

Please verify your email address by visiting the following link (expires in 30 minutes):
{verification_url}

About FinTwin AI:
FinTwin AI is an AI-powered financial health platform designed to help users understand their financial health, analyze financial behavior, plan financial goals, simulate financial scenarios, forecast financial outcomes, and receive personalized financial guidance.

Security Reminder:
Please keep your account credentials secure. Never share your password with anyone. FinTwin AI will never ask you to send your password by email.
"""

    security_notice = "This verification link is single-use and will expire in 30 minutes. If you did not create this account, please ignore this email."

    html_body = _build_html_email(
        title=title,
        preheader=preheader,
        content_html=content_html,
        action_url=verification_url,
        action_text="Verify My Email",
        security_notice=security_notice,
    )

    return _send_smtp_email(
        to_email=email,
        subject="Verify your FinTwin AI Account",
        html_body=html_body,
        text_body=text_body,
    )


def send_password_reset_email(name: str, email: str, raw_token: str) -> Tuple[bool, str]:
    """
    Generates and sends the official FinTwin AI Password Reset email.
    """
    cfg = get_smtp_config()
    base_url = cfg["base_url"]
    reset_url = f"{base_url}/?reset_password_token={raw_token}"

    title = "Password Reset Request"
    preheader = f"Reset the password for your FinTwin AI account."

    content_html = f"""
    <p style="margin: 0 0 16px 0;">Hi <strong>{_he(name)}</strong>,</p>
    <p style="margin: 0 0 16px 0;">
        We received a request to reset the password for your FinTwin AI account associated with <strong>{_he(email)}</strong>.
    </p>
    <p style="margin: 0 0 16px 0;">
        Click the button below to choose a new password. This password reset link is single-use and will expire in <strong>30 minutes</strong>.
    </p>
    """

    text_body = f"""Hi {name},

We received a request to reset the password for your FinTwin AI account ({email}).

To reset your password, visit the following link within 30 minutes:
{reset_url}

If you did not request this password reset, you can safely ignore this email. Your password will not be changed.

Security Reminder:
Please keep your account credentials secure. Never share your password with anyone. FinTwin AI will never ask you to send your password by email.
"""

    security_notice = "This password reset link will expire in 30 minutes. If you did not request this password reset, you can safely ignore this email. Your password will remain unchanged."

    html_body = _build_html_email(
        title=title,
        preheader=preheader,
        content_html=content_html,
        action_url=reset_url,
        action_text="Reset My Password",
        security_notice=security_notice,
    )

    return _send_smtp_email(
        to_email=email,
        subject="Reset your FinTwin AI Password",
        html_body=html_body,
        text_body=text_body,
    )


def send_account_deletion_email(name: str, email: str) -> Tuple[bool, str]:
    """
    Sends an account deletion confirmation email confirming that all user data
    has been permanently removed.
    """
    title = "Account Deleted"
    preheader = "Your FinTwin AI account and personal data have been permanently removed."

    content_html = f"""
    <p style="margin: 0 0 16px 0;">Hi <strong>{_he(name)}</strong>,</p>
    <p style="margin: 0 0 16px 0;">
        This email confirms that your FinTwin AI account and all associated personal financial data, transactions, goals, and digital twin models have been permanently deleted from our systems.
    </p>
    <p style="margin: 0 0 16px 0;">
        We are sorry to see you go! If you ever wish to return and plan your financial future with FinTwin AI, you are always welcome to create a new account.
    </p>
    """

    text_body = f"""Hi {name},

This email confirms that your FinTwin AI account ({email}) and all associated personal financial records, goals, transactions, and digital twin state have been permanently deleted from our systems.

We are sorry to see you go! If you ever wish to return to FinTwin AI, you are always welcome to create a fresh account.
"""

    security_notice = "This is a confirmation notice. If you did not authorize this deletion, please contact support immediately."

    html_body = _build_html_email(
        title=title,
        preheader=preheader,
        content_html=content_html,
        security_notice=security_notice,
    )

    return _send_smtp_email(
        to_email=email,
        subject="Your FinTwin AI Account Has Been Deleted",
        html_body=html_body,
        text_body=text_body,
    )
