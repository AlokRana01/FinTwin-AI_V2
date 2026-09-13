"""
Settings & Privacy Page
=======================
Provides authenticated users with complete control over account security,
password management, secure personal data export (JSON), and permanent atomic account deletion.
"""

import streamlit as st
import json
import os
import datetime
from html import escape as _he
from utils.session import render_sidebar_user_selector, invalidate_session_twin
from database.db_manager import DBManager
from utils import auth
from utils.avatar import render_avatar_html, render_avatar_selector
from utils.email_service import send_account_deletion_email

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Settings & Privacy — FinTwin AI",
    page_icon="assets/logos/favicon.svg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Render Sidebar Navigation and retrieve active Digital Twin
twin = render_sidebar_user_selector()

# Custom Styling for Settings & Privacy
st.markdown("""
<style>
.settings-header-box {
    margin-bottom: 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.settings-section-card {
    background-color: #111827;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}
.settings-section-title {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.35rem;
}
.settings-section-title h4 {
    margin: 0;
    font-size: 1.02rem;
    font-weight: 700;
    color: #F8FAFC;
}
.settings-section-desc {
    color: #94A3B8;
    font-size: 0.82rem;
    margin-bottom: 0.85rem;
    line-height: 1.4;
}
.danger-zone-card {
    background: linear-gradient(180deg, rgba(239, 68, 68, 0.03) 0%, rgba(17, 24, 39, 0.95) 100%);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    margin-top: 1.25rem;
    margin-bottom: 1rem;
}
.danger-badge {
    background: rgba(239, 68, 68, 0.15);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
    font-size: 0.68rem;
    font-weight: 700;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    display: inline-block;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}
.info-row:last-child {
    border-bottom: none;
}
.info-label {
    color: #94A3B8;
    font-size: 0.84rem;
    font-weight: 500;
}
.info-value {
    color: #F8FAFC;
    font-size: 0.84rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ── Authentication Barrier ───────────────────────────────────────────────────
if not twin:
    st.markdown("""
    <div class="settings-header-box">
        <h2 style="margin: 0 0 4px 0; font-size: 1.6rem; font-weight: 800; color: #F8FAFC;">Settings & Privacy</h2>
        <p style="color: #94A3B8; font-size: 0.88rem; margin: 0;">
            Please log in or register via the sidebar to manage your account security, download data, or configure preferences.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.info("Authentication required to access Settings & Privacy.")
    st.stop()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="settings-header-box">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
        <div>
            <h2 style="margin: 0 0 2px 0; font-size: 1.6rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">Settings & Privacy</h2>
            <p style="color: #94A3B8; font-size: 0.86rem; margin: 0;">Manage your account credentials, security preferences, and personal financial data.</p>
        </div>
        <div style="background: rgba(79, 140, 255, 0.1); color: #4F8CFF; border: 1px solid rgba(79, 140, 255, 0.2); font-size: 0.72rem; font-weight: 600; padding: 0.25rem 0.65rem; border-radius: 6px;">
            Account ID: <code>{_he(twin.user_id)}</code>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Retrieve account details from database
user_profile = DBManager.get_user_profile(twin.user_id) or {}
user_email = user_profile.get("email", "")
created_at_str = str(user_profile.get("created_at", "Active Session"))

# ── Main Tabs: Privacy & Security | Profile Details ───────────────────────────
tab_security, tab_profile = st.tabs([
    ":material/shield: Privacy & Security",
    ":material/person: Account Overview",
])

with tab_security:
    # ── 1. ACCOUNT SECURITY OVERVIEW ─────────────────────────────────────────
    st.markdown("""
    <div class="settings-section-card">
        <div class="settings-section-title">
            <i class="fa-solid fa-lock" style="color: #4F8CFF; font-size: 0.95rem;"></i>
            <h4>Account & Security Overview</h4>
        </div>
        <div class="settings-section-desc">
            Your account is secured with Argon2id cryptographic password hashing and automated session protection.
        </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="info-row"><span class="info-label">Registered Email</span><span class="info-value">{_he(user_email)}</span></div>
        <div class="info-row"><span class="info-label">Account ID</span><span class="info-value"><code>{_he(twin.user_id)}</code></span></div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="info-row"><span class="info-label">Email Status</span><span class="info-value" style="color: #22C55E;">Verified</span></div>
        <div class="info-row"><span class="info-label">Member Since</span><span class="info-value">{_he(created_at_str[:10])}</span></div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── 2. CHANGE PASSWORD ───────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top: 1.25rem; margin-bottom: 0.5rem;">
        <div class="settings-section-title">
            <i class="fa-solid fa-key" style="color: #00D4FF; font-size: 0.95rem;"></i>
            <h4>Change Password</h4>
        </div>
        <div class="settings-section-desc">
            Update your account password (at least 8 characters, with uppercase, lowercase, and numbers or symbols).
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("change_password_form", clear_on_submit=True):
        col_pw1, col_pw2, col_pw3 = st.columns(3)
        with col_pw1:
            curr_pw = st.text_input("Current Password", type="password", max_chars=64)
        with col_pw2:
            new_pw = st.text_input("New Password", type="password", max_chars=64)
        with col_pw3:
            confirm_new_pw = st.text_input("Confirm New Password", type="password", max_chars=64)

        submit_pw = st.form_submit_button("Update Password", type="primary", use_container_width=False)
        if submit_pw:
            ok, msg = auth.change_user_password(
                user_id=twin.user_id,
                current_password=curr_pw,
                new_password=new_pw,
                confirm_password=confirm_new_pw,
            )
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    # ── 3. YOUR DATA & PORTABILITY (DOWNLOAD MY DATA) ────────────────────────
    st.markdown("""
    <div style="margin-top: 1.5rem; margin-bottom: 0.5rem;">
        <div class="settings-section-title">
            <i class="fa-solid fa-file-arrow-down" style="color: #22C55E; font-size: 0.95rem;"></i>
            <h4>Download My Data</h4>
        </div>
        <div class="settings-section-desc">
            Download a portable, structured JSON export of all personal and financial data associated with your account.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Export generation
    user_export = DBManager.export_user_data(twin.user_id)
    if user_export:
        tx_count = len(user_export.get("transactions", []))
        goal_count = len(user_export.get("goals", []))
        chat_days = len(user_export.get("ai_coach_usage", []))

        col_d1, col_d2 = st.columns([1, 2.5])
        with col_d1:
            st.download_button(
                label="Download JSON Export",
                icon=":material/download:",
                data=json.dumps(user_export, indent=2),
                file_name=f"fintwin_ai_data_{twin.user_id}.json",
                mime="application/json",
                use_container_width=True,
                type="secondary",
            )
        with col_d2:
            with st.expander(f"Export Summary ({tx_count} transactions, {goal_count} goals)"):
                st.json({
                    "metadata": user_export.get("metadata"),
                    "account": user_export.get("account"),
                    "digital_twin_summary": {
                        "net_worth": user_export.get("digital_twin", {}).get("net_worth"),
                        "bank_savings": user_export.get("digital_twin", {}).get("bank_savings"),
                        "loan_amount": user_export.get("digital_twin", {}).get("loan_amount"),
                    },
                    "records_count": {
                        "transactions": tx_count,
                        "goals": goal_count,
                    }
                })
    else:
        st.warning("Could not prepare export at this moment. Please try again.")

    # ── 4. DANGER ZONE: DELETE ACCOUNT & DATA ────────────────────────────────
    st.markdown("""
    <div class="danger-zone-card">
        <div class="danger-badge">Danger Zone</div>
        <div class="settings-section-title">
            <i class="fa-solid fa-triangle-exclamation" style="color: #EF4444; font-size: 0.95rem;"></i>
            <h4 style="color: #EF4444;">Delete Account & Data</h4>
        </div>
        <div class="settings-section-desc" style="margin-bottom: 0.65rem;">
            Permanently delete your FinTwin AI account and all personal financial records. This action cannot be undone.
        </div>
    """, unsafe_allow_html=True)

    # Show initial button or expanded confirmation view
    if "show_delete_confirmation" not in st.session_state:
        st.session_state["show_delete_confirmation"] = False

    if not st.session_state["show_delete_confirmation"]:
        if st.button("Delete Account...", type="secondary", key="initiate_delete_btn"):
            st.session_state["show_delete_confirmation"] = True
            st.rerun()
    else:
        st.markdown("""
        <div style="background-color: rgba(239, 68, 68, 0.08); border-left: 3px solid #EF4444; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px;">
            <strong style="color: #EF4444; font-size: 0.86rem;">Permanent Deletion Warning:</strong>
            <p style="color: #CBD5E1; font-size: 0.80rem; margin: 4px 0 0 0; line-height: 1.4;">
                All personal demographics, digital twin financial models, transaction records, goals, and AI history will be permanently deleted.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if user_export:
            col_pre_dl, _ = st.columns([1.5, 2.5])
            with col_pre_dl:
                st.download_button(
                    label="Download Data Backup",
                    icon=":material/download:",
                    data=json.dumps(user_export, indent=2),
                    file_name=f"fintwin_ai_backup_{twin.user_id}.json",
                    mime="application/json",
                    use_container_width=True,
                )

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        del_pw = st.text_input(
            "Enter current password to authorize deletion",
            type="password",
            max_chars=64,
            key="del_pw_input",
        )
        confirm_check = st.checkbox(
            "I understand that my FinTwin AI account and all financial data will be permanently deleted.",
            key="del_confirm_checkbox",
        )

        is_deletion_authorized = bool(confirm_check and del_pw.strip())

        col_del_act1, col_del_act2 = st.columns([1, 1])
        with col_del_act1:
            confirm_delete = st.button(
                "Permanently Delete Account",
                type="primary",
                disabled=not is_deletion_authorized,
                use_container_width=True,
                key="confirm_permanent_delete_btn",
            )
        with col_del_act2:
            cancel_delete = st.button(
                "Cancel",
                type="secondary",
                use_container_width=True,
                key="cancel_permanent_delete_btn",
            )

        if cancel_delete:
            st.session_state["show_delete_confirmation"] = False
            st.rerun()

        if confirm_delete:
            if not auth.verify_user_credentials_by_id(twin.user_id, del_pw.strip()):
                st.error("Incorrect password. Account deletion was rejected.")
            else:
                email_to_notify = user_email
                user_name = twin.name
                user_id_to_delete = twin.user_id

                ok, del_msg = DBManager.delete_user_account_atomic(user_id_to_delete)
                if ok:
                    if email_to_notify:
                        send_account_deletion_email(name=user_name, email=email_to_notify)

                    st.session_state.pop("user_id", None)
                    st.session_state.pop("ftw_active_twin", None)
                    st.session_state.pop("_session_last_active", None)
                    st.session_state.pop("_master_pdf_bytes", None)
                    st.session_state.pop("_master_pdf_uid", None)
                    st.session_state.pop("show_delete_confirmation", None)
                    st.session_state["_auth_success_msg"] = "Your FinTwin AI account and associated personal data have been permanently deleted."
                    if "uid" in st.query_params:
                        del st.query_params["uid"]
                    st.rerun()
                else:
                    st.error(del_msg)

    st.markdown("</div>", unsafe_allow_html=True)


with tab_profile:
    st.markdown("""
    <div class="settings-section-card">
        <div class="settings-section-title">
            <i class="fa-solid fa-user-circle" style="color: #4F8CFF; font-size: 0.95rem;"></i>
            <h4>Financial Profile Summary</h4>
        </div>
        <div class="settings-section-desc">
            Active demographics and income parameters configured in your Financial Digital Twin.
        </div>
    """, unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown(f"""
        <div class="info-row"><span class="info-label">Full Name</span><span class="info-value">{_he(twin.name)}</span></div>
        <div class="info-row"><span class="info-label">Age</span><span class="info-value">{twin.age} years</span></div>
        <div class="info-row"><span class="info-label">City / Region</span><span class="info-value">{_he(twin.city or 'Not set')}</span></div>
        <div class="info-row"><span class="info-label">Occupation</span><span class="info-value">{_he(twin.occupation or 'Salaried Professional')}</span></div>
        """, unsafe_allow_html=True)
    with col_p2:
        st.markdown(f"""
        <div class="info-row"><span class="info-label">Monthly Income</span><span class="info-value">₹{twin.monthly_income:,.0f}</span></div>
        <div class="info-row"><span class="info-label">Annual Bonus</span><span class="info-value">₹{twin.bonus:,.0f}</span></div>
        <div class="info-row"><span class="info-label">Additional Income</span><span class="info-value">₹{twin.additional_income:,.0f}</span></div>
        <div class="info-row"><span class="info-label">Total Monthly Net</span><span class="info-value" style="color: #00D4FF;">₹{twin.total_income:,.0f}</span></div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── 2. PROFILE AVATAR MANAGEMENT ─────────────────────────────────────────
    current_avatar_id = getattr(twin, "avatar_id", "avatar_01")
    current_avatar_html = render_avatar_html(current_avatar_id, size=56, class_name="settings-current-avatar-img")

    st.markdown(f"""
    <div class="settings-section-card" style="margin-top: 1.25rem;">
        <div class="settings-section-title">
            <i class="fa-solid fa-circle-user" style="color: #4F8CFF; font-size: 0.95rem;"></i>
            <h4>Profile Avatar</h4>
        </div>
        <div class="settings-section-desc">
            Personalize your account appearance across FinTwin AI.
        </div>
        <div style="display: flex; align-items: center; gap: 16px; margin: 12px 0 16px 0; padding: 12px 16px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px;">
            {current_avatar_html}
            <div>
                <div style="font-size: 0.88rem; font-weight: 700; color: #F8FAFC;">Current Profile Avatar</div>
                <div style="font-size: 0.78rem; color: #94A3B8;">Selected: <code>{_he(current_avatar_id)}</code></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    new_avatar = render_avatar_selector(
        key="settings_avatar_selector",
        selected_id=current_avatar_id,
        label="Choose your profile avatar",
    )

    col_sav1, _ = st.columns([1.5, 3.5])
    with col_sav1:
        if st.button("Save Avatar", type="primary", key="save_avatar_btn", use_container_width=True):
            if new_avatar:
                if new_avatar == current_avatar_id:
                    st.info("You already have this avatar selected.")
                else:
                    ok = DBManager.update_user_avatar(twin.user_id, new_avatar)
                    if ok:
                        twin.avatar_id = new_avatar
                        invalidate_session_twin()
                        st.success("Your profile avatar has been updated successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to update profile avatar.")

    st.markdown("</div>", unsafe_allow_html=True)
