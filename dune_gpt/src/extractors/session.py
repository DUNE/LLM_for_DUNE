
from abc import ABC, abstractmethod
import requests
import json
from src.utils.logger import get_logger
import os
logger = get_logger(__name__)
try:
    import cloudscraper  # pip install cloudscraper
    HAS_CLOUDSCRAPER = True
except Exception:
    HAS_CLOUDSCRAPER = False


from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)
DEFAULT_COOKIES_PATH = os.path.expanduser("~/.indico_cookies.json")


class Session(ABC):
    def __init__(self):
        super().__init__()
        print("session called")
        self.session = self._build_session()

    @abstractmethod
    def _build_session(self) -> requests.Session:
        pass

    def _load_cookies(self) -> None:
        try:
            if os.path.exists(self.cookies_file):
                with open(self.cookies_file, "r") as f:
                    cookies = json.load(f)

                ua = None
                for c in cookies:
                    self.session.cookies.set(
                        c["name"], c["value"],
                        domain=c.get("domain"),
                        path=c.get("path", "/")
                    )
                    if not ua and c.get("user_agent"):
                        ua = c["user_agent"]
                if ua:
                    self.session.headers["User-Agent"] = ua
                logger.info(f"Loaded cookies from {self.cookies_file}")
        except Exception as e:
            logger.warning(f"Failed to load cookies: {e}")

    def _save_cookies(self) -> None:
        try:
            cookies = []
            for c in self.session.cookies:
                cookies.append({
                    "name": c.name,
                    "value": c.value,
                    "domain": c.domain,
                    "path": c.path,
                    "user_agent": self.session.headers.get("User-Agent", DEFAULT_UA),
                })
            with open(self.cookies_file, "w") as f:
                json.dump(cookies, f)
            logger.info(f"Saved cookies to {self.cookies_file}")
        except Exception as e:
            logger.warning(f"Failed to save cookies: {e}")

    def _is_cloudflare_challenge(self, resp: requests.Response) -> bool:
        try:
            text = resp.text.lower()
        except Exception:
            text = ""
        if resp.status_code in (403, 503):
            if "just a moment" in text:
                return True
            if "cf-ray" in {k.lower() for k in resp.headers.keys()}:
                return True
            if "cf-mitigated" in {k.lower() for k in resp.headers.keys()}:
                return True
            if any("cloudflare" in str(v).lower() for v in resp.headers.values()):
                return True
        return False

    def _upgrade_to_cloudscraper(self) -> bool:
        if not HAS_CLOUDSCRAPER:
            return False
        try:
            cs = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "mobile": False}
            )
            cs.headers.update(self.session.headers)
            for c in self.session.cookies:
                cs.cookies.set(c.name, c.value, domain=c.domain, path=c.path)
            if self.api_token:
                cs.headers.update({"Authorization": f"Bearer {self.api_token}"})
            self.session = cs
            logger.info("Switched to cloudscraper session.")
            return True
        except Exception as e:
            logger.warning(f"cloudscraper init failed: {e}")
            return False

    def _browser_cookie_bootstrap(self) -> bool:
        """
        Use Playwright to load the site in a real browser, pass Cloudflare and optionally SSO,
        then copy cookies and UA into the requests session.
        Requires:
          pip install playwright
          playwright install chromium
        """
        try:
            from playwright.sync_api import sync_playwright
        except Exception:
            logger.error("Playwright not available. Install with: pip install playwright && playwright install chromium")
            return False

        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)  # headful helps with MFA/SSO/CF
                ctx = browser.new_context()
                page = ctx.new_page()
                logger.warning(f"In cookies 189, page={page}")

                # Load base to satisfy Cloudflare; wait a bit for challenge completion
                page.goto(self.base_url, wait_until="domcontentloaded")
                page.wait_for_timeout(6000)

                # Optionally visit export endpoint to ensure cookies apply there too
                page.goto(f"{self.base_url}/export/categ/{self.category_id}.json", wait_until="domcontentloaded")
                page.wait_for_timeout(3000)

                # If SSO login is required for some content, user can navigate to /login and complete it
                # page.goto(f"{self.base_url}/login", wait_until="domcontentloaded")

                cookies = ctx.cookies()
                ua = page.evaluate("() => navigator.userAgent") or DEFAULT_UA

                self.session.headers["User-Agent"] = ua
                self.session.cookies.clear()
                for c in cookies:
                    self.session.cookies.set(c["name"], c["value"], domain=c.get("domain"), path=c.get("path", "/"))

                browser.close()
                logger.info("Captured browser cookies and user agent.")
                self._save_cookies()
                return True
        except Exception as e:
            logger.error(f"Browser bootstrap failed: {e}")
            return False

    def _ensure_access(self) -> None:
        if self._auth_initialized:
            return

        test_url = f"{self.base_url}/categ/{self.category_id}"
        r = self.session.get(test_url, timeout=30)
        logger.error(f"esure access {r}")

        if r.status_code == 200:
            self._auth_initialized = True
            logger.info("Access OK (API token/cookies).")
            return

        if self._is_cloudflare_challenge(r):
            logger.warning("Cloudflare challenge detected on export API.")
            # Trpython3 cli.py index --docdb-limit  cloudscraper first
            if self._upgrade_to_cloudscraper():
                r = self.session.get(test_url, timeout=30)
                if r.status_code == 200:
                    self._auth_initialized = True
                    logger.info("Access OK via cloudscraper.")
                    return

        # Try browser cookie bootstrap
        if self.use_browser_login and self._browser_cookie_bootstrap():
            r = self.session.get(test_url, timeout=30)
            if r.status_code == 200:
                self._auth_initialized = True
                logger.info("Access OK via browser cookies.")
                return

        # Final failure
        snippet = (r.text or "")[:300].replace("\n", " ")
        logger.error(f"Cannot access {test_url} (status {r.status_code}). Snippet: {snippet}")
        raise RuntimeError(
            "Blocked by Cloudflare/SSO. Set INDICO_API_TOKEN and either "
            "install cloudscraper or enable INDICO_USE_BROWSER_LOGIN to capture cookies."
        )
