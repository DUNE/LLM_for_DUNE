# playwright_login_full.py
import asyncio
import os
import json
from playwright.async_api import async_playwright

COOKIE_FILE = "indico_cookies.json"
INDICO_URL = "https://indico.fnal.gov/login/"

# SSO credentials from environment
username = os.getenv("FERMI_SSO_USERNAME")
password = os.getenv("FERMI_SSO_PASSWORD")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # headless=True for no UI
        context = await browser.new_context()
        page = await context.new_page()

        # Step 1: Go to Indico login page
        await page.goto(INDICO_URL)

        # Step 2: Click "Institutional SSO"
        await page.click('text=Institutional SSO')

        # Step 3: Wait for Fermilab SSO page
        await page.wait_for_url("https://pingprod.fnal.gov/*")

        # Step 4: Fill in username/password
        await page.fill('input[name="username"]', username)
        await page.fill('input[name="password"]', password)
        await page.click('button[type="submit"]')

        # Step 5: Handle "Continue" page
        # Sometimes it’s inside an iframe or requires a small delay
        try:
            # Wait for a frame that contains Continue button
            await page.wait_for_timeout(2000)  # wait for JS to load
            for frame in page.frames:
                continue_btn = await frame.query_selector('text=Continue')
                if continue_btn:
                    await continue_btn.click()
                    break
            else:
                # Fallback: try clicking on main page
                await page.click('text=Continue')
        except Exception as e:
            print("Continue button not found automatically:", e)

        # Step 6: Wait for redirect back to Indico
        await page.wait_for_url("https://indico.fnal.gov/*", timeout=60000)

        # Step 7: Save cookies
        cookies = await context.cookies()
        with open(COOKIE_FILE, "w") as f:
            json.dump(cookies, f, indent=2)
        print(f"Cookies saved to {COOKIE_FILE}")

        await browser.close()

asyncio.run(main())