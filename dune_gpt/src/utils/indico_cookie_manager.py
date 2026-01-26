# indico_cookie_manager.py
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
from datetime import datetime, timedelta

import requests

url = "https://indico.fnal.gov/export/categ/455.json"
resp = requests.get(url)
print(resp.status_code)
print(resp.json().get("results", [])[:5])  # just the first 5 events

COOKIE_FILE = os.path.expanduser("~/.indico_cookie.json")
COOKIE_LIFETIME = timedelta(hours=24)  # assume FNAL cookie lifetime ~12h

def _save_cookie(cookies: dict):
    data = {
        "cookies": cookies,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(COOKIE_FILE, "w") as f:
        json.dump(data, f)

def _load_raw_cookie():
    if not os.path.exists(COOKIE_FILE):
        return None
    with open(COOKIE_FILE) as f:
        return json.load(f)

def login_and_save_cookie(username: str, password: str, base_url="https://indico.fnal.gov"):
    session = requests.Session()

    # Step 1: Go to login page
    resp = session.get(f"{base_url}/login/")
    soup = BeautifulSoup(resp.text, "html.parser")
    form = soup.find("form")
    if not form:
        raise RuntimeError("Could not find login form on Indico")

    login_url = urljoin(base_url, form["action"])

    # Step 2: Submit login credentials
    login_data = {
        "pf.username": username,
        "pf.pass": password,
        "pf.ok": "clicked",
        "pf.adapterId": "FormBased"
    }
    resp = session.post(login_url, data=login_data)

    # Step 3: Handle SAML redirect if present
    soup = BeautifulSoup(resp.text, "html.parser")
    form = soup.find("form")
    if form and "SAMLResponse" in resp.text:
        saml_url = form["action"]
        saml_data = {
            "RelayState": form.find("input", {"name": "RelayState"})["value"],
            "SAMLResponse": form.find("input", {"name": "SAMLResponse"})["value"],
        }
        resp = session.post(saml_url, data=saml_data)

    # Save cookies + timestamp
    cookies = session.cookies.get_dict()
    _save_cookie(cookies)
    print(f"[INFO] Logged in and saved Indico cookie to {COOKIE_FILE}")
    return cookies

def load_cookie(auto_refresh=True):
    data = _load_raw_cookie()
    if not data:
        print("[WARN] No cookie found, logging in fresh...")
        return _do_refresh()

    timestamp = datetime.fromisoformat(data["timestamp"])
    if auto_refresh and datetime.utcnow() - timestamp > COOKIE_LIFETIME:
        print("[INFO] Cookie expired, refreshing...")
        return _do_refresh()
    return data["cookies"]

def _do_refresh():
    username = os.getenv("FERMI_SSO_USERNAME")
    password = os.getenv("FERMI_SSO_PASSWORD")
    if not username or not password:
        raise ValueError("FERMI_SSO_USERNAME and FERMI_SSO_PASSWORD must be set in environment.")
    return login_and_save_cookie(username, password)