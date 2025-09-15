# bootstrap_cookies.py
import os
import json
import sys

from playwright.sync_api import sync_playwright

BASE_URL = os.environ.get("INDICO_BASE_URL", "https://indico.fnal.gov").rstrip("/")
COOKIES_FILE = os.environ.get("INDICO_COOKIES_FILE", os.path.expanduser("~/.indico_cookies.json"))

def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        ctx = browser.new_context()
        page = ctx.new_page()

        page.goto(BASE_URL, wait_until="domcontentloaded")
        page.wait_for_timeout(7000)  # let Cloudflare complete
        # If you need SSO, navigate and login here:
        # page.goto(f"{BASE_URL}/login", wait_until="domcontentloaded")
        # complete login manually; wait if needed:
        # page.wait_for_timeout(20000)

        cookies = ctx.cookies()
        ua = page.evaluate("() => navigator.userAgent")

        # Store cookies + UA to file
        data = []
        for c in cookies:
            data.append({
                "name": c["name"],
                "value": c["value"],
                "domain": c.get("domain"),
                "path": c.get("path", "/"),
                "user_agent": ua
            })
        with open(COOKIES_FILE, "w") as f:
            json.dump(data, f)
        print(f"Saved cookies to {COOKIES_FILE}")
        browser.close()

if __name__ == "__main__":
    sys.exit(main())