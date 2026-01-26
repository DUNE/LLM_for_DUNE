# indico_extractor.py
import os
import json
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional: cloudscraper can help with basic Cloudflare challenges
try:
    import cloudscraper  # pip install cloudscraper
    HAS_CLOUDSCRAPER = True
except Exception:
    HAS_CLOUDSCRAPER = False

from .base import BaseExtractor
from config import (
    INDICO_BASE_URL,
    INDICO_CATEGORY_ID,
    INDICO_API_TOKEN,       # Personal Access Token (create in Indico UI)
    INDICO_COOKIES_FILE,    # optional path to persist browser cookies
    INDICO_USE_BROWSER_LOGIN  # bool: use Playwright to fetch cookies on first run
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)
DEFAULT_COOKIES_PATH = os.path.expanduser("~/.indico_cookies.json")


class IndicoExtractor(BaseExtractor):
    """
    Indico extractor resilient to Cloudflare + SSO:
    - Uses export API with Personal Access Token (Bearer).
    - Detects Cloudflare 403/503 challenge; tries cloudscraper or browser cookies.
    - Reuses consistent User-Agent + cookies for subsequent requests.
    - Avoids scraping HTML.
    """

    def __init__(self,faiss):
        super().__init__()
        self.base_url = INDICO_BASE_URL.rstrip("/")
        self.category_id = INDICO_CATEGORY_ID
        self.api_token = INDICO_API_TOKEN or os.getenv("INDICO_API_TOKEN")
        self.cookies_file = (INDICO_COOKIES_FILE
                             or os.getenv("INDICO_COOKIES_FILE")
                             or DEFAULT_COOKIES_PATH)
        self.use_browser_login = bool(
            INDICO_USE_BROWSER_LOGIN or os.getenv("INDICO_USE_BROWSER_LOGIN")
        )
        self.max_file_bytes =  50 * 1024 * 1024
        self.faiss=faiss
        self.session = self._build_session()
        self._auth_initialized = False
        self.total_fetched =0
        # Load any previously captured cookies (e.g., via Playwright)
        self._load_cookies()

    # ------------------------ Session and Auth ------------------------

    def _build_session(self) -> requests.Session:
        s = requests.Session()

        retry = Retry(
            total=5,
            read=5,
            connect=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "HEAD"])
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
        s.mount("http://", adapter)
        s.mount("https://", adapter)

        s.headers.update({
            "User-Agent": DEFAULT_UA,
            "Accept": "application/json,text/html;q=0.8,*/*;q=0.5",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        })

        if self.api_token:
            s.headers.update({"Authorization": f"Bearer {self.api_token}"})

        return s

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
                browser = pw.chromium.launch(headless=False)  # headful helps with MFA/SSO/CF
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
                self.session.get(f"{self.base_url}/categ/{self.category_id}", timeout=30)
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


    def _fetch_category_events(self, categ_id:int, limit: int) -> List[Dict[str, Any]]:
        """
        Fetch recent events in the category via the export API.
        """
        if self.total_fetched >= limit: return []
        url = f"{self.base_url}/export/categ/{categ_id}.json"
        #url = 'https://indico.fnal.gov/event/10575/'
        params = {
            "limit": max(1, limit),
            # You can also add time filters if needed:
            # "from": "today-365", "to": "today"
        }
        r = self.session.get(url, timeout=45)
        # If Cloudflare blocks here, _ensure_access should have handled it,
        # but we re-check for safety.


        # gets all details in the header
        if hasattr(self, "_is_cloudflare_challenge") and self._is_cloudflare_challenge(r):
            raise RuntimeError("Cloudflare blocked category export; ensure token/cookies configured.")
        r.raise_for_status()
        data = r.json()
        self.total_fetched += min(len(data.get("results",[])), limit)
        return data.get("results", [])[:limit]

    def _fetch_event_details(self, event_id: int) -> Dict[str, Any]:
        """
        Fetch event details via export API. Try multiple parameter combos that
        different Indico versions use to expose contributions/material/attachments.
        """
        
        url = f"{self.base_url}/export/event/{event_id}.json"

        # Try a sequence of parameter sets
        param_candidates = [
            {"attachments": "1", "contributions": "all"},
            {"attachments": "yes", "contributions": "all"},
            {"attachments": "1", "detail": "contributions"},
            {"attachments": "yes", "detail": "contributions"},
            # Some setups require 'material'
            {"attachments": "1", "contributions": "all", "material": "1"},
            {"attachments": "yes", "contributions": "all", "material": "1"},
        ]

        last_resp = None
        for params in param_candidates:
            r = self.session.get(url, params=params, timeout=45)
            last_resp = r
            if getattr(self, "_is_cloudflare_challenge")(r):
                raise RuntimeError(f"Cloudflare blocked event export {event_id}.")

            if r.status_code != 200:
                continue
            results = r.json().get("results", [])
            if not results:
                continue
            details = results[0] if isinstance(results, list) else results
            # Heuristics: if this payload includes either contributions or attachments/material
            contribs = details.get("contributions") or []
            top_attachments = details.get("attachments") or []
            materials = details.get("material") or details.get("materials") or []

            if contribs or top_attachments or materials:
                #logger.info(f"Event {event_id}: export params {params} yielded "
                            #f"{len(contribs)} contributions, "
                            #f"{len(top_attachments)} top attachments, "
                            #f"{len(materials)} materials")
                return details
            

        # If none of the combos returned anything useful, return an empty details dict
        if last_resp is not None and last_resp.status_code != 200:
            last_resp.raise_for_status()
        logger.warning(f"Event {event_id}: export API did not yield attachments; will try HTML fallback.")
        return {}

    def _collect_attachments(self, obj: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect attachments from several common export shapes:
        - obj['folders']-> list of attachment dicts
        - obj['material'][i]['attachments'] -> list of attachment dicts
        - obj['materials'][i]['attachments'] -> list of attachment dicts
        - attachment entries can carry direct 'download_url' or a 'files' list
        """
        attachments: List[Dict[str, Any]] = []
        # Direct attachments list
        if isinstance(obj.get("folders"), list):
            for attachment_metadata in obj['folders']:
                attachments.extend(attachment_metadata["attachments"])
        else:
            logger.error(f"DID NOT FIND AN ATTACHMENT IN OBJECT")
        

        # Materials may be under 'material' or 'materials'
        assert 'material' in obj.keys() or 'materials' in obj.keys(), 'neither matieral nor materials in contirb'
        for key in ("material", "materials"):
            mats = obj.get(key) or []
            if isinstance(mats, list):
                for mat in mats:
                    ats = mat.get("attachments") or []
                    if isinstance(ats, list):
                        attachments.extend(ats)

        return attachments

    def _normalize_attachment(self, att: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Normalize fields for an attachment entry from export JSON.
        Supports different variants:
        - att['download_url'] or att['url'] or att['link']
        - att['files'][0]['download_url'] / ['url'] / ['filename']
        """
        title = att.get("title") or att.get("name") or att.get("filename") or ""
        url = att.get("download_url") or att.get("url") or att.get("link")

        if not url:
            files = att.get("files") or []
            if files and isinstance(files, list):
                url = files[0].get("download_url") or files[0].get("url")
                title = files[0].get("filename") or title

        if not url:
            return None

        if url.startswith("/"):
            url = urljoin(self.base_url, url)

        # Prefer filename from URL, else from title
        filename = url.split("/")[-1]
        if "." in filename:
            file_type = filename.split(".")[-1].lower()
        else:
            # Try to infer from title
            file_type = (title.split(".")[-1].lower() if "." in title else "")

        return {"title": title, "url": url, "filename": filename, "file_type": file_type}

    def get_soup_parser(self,url):
        import bs4 
        r = self.session.get(url, timeout=60)
        if r.status_code != 200:
            logger.warning(f"HTML fallback: event page {url} returned {r.status_code}")
            return []

        soup = bs4.BeautifulSoup(r.text, "html.parser")
        return soup
    
    def _scrape_event_attachments_html(self, event_id: int) -> List[Dict[str, Any]]:
        """
        HTML fallback: fetch the event page and scrape attachment links.
        We search for anchors whose href contains '/attachments/' and ends with .pdf/.pptx.
        We also try to pair them with a contribution title if present in the same timetable item.

        ie if u just need the header
        """
         # BeautifulSoup already in your deps
        url = f"{self.base_url}/event/{event_id}/"
        
        soup = self.get_soup_parser(url)
        found: List[Dict[str, Any]] = []

        # Strategy 1: attachments within timetable contributions
        for li in soup.select("li.timetable-item"):
            # Contribution title (if any)
            contrib_title = None
            t = li.select_one(".timetable-title")
            if t:
                contrib_title = t.get_text(strip=True)

            # Speakers (if any)
            contrib_author = None
            s = li.select_one(".speaker-list")
            if s:
                txt = s.get_text(separator=" ", strip=True)
                contrib_author = re.sub(r"(?i)^speakers?\s*:\s*", "", txt)
                contrib_author = re.sub(r"\s*\([^)]*\)", "", contrib_author).strip()

            # Attachment anchors
            for a in li.select('a[href*="/attachments/"]'):
                href = a.get("href") or ""
                # accept .pdf or .pptx (case-insensitive)
                if not re.search(r"\.(pdf|pptx)(?:\?|$)", href, flags=re.I):
                    continue
                abs_url = href if href.startswith("http") else urljoin(self.base_url, href)
                filename = abs_url.split("/")[-1]
                ext = filename.split(".")[-1].lower() if "." in filename else ""
                found.append({
                    "title": contrib_title or filename,
                    "author": contrib_author or "Unknown Author",
                    "url": abs_url,
                    "filename": filename,
                    "file_type": ext
                })

        # Strategy 2: catch any remaining attachments anywhere on the page
        if not found:
            for a in soup.select('a[href*="/attachments/"]'):
                href = a.get("href") or ""
                if not re.search(r"\.(pdf|pptx)(?:\?|$)", href, flags=re.I):
                    continue
                abs_url = href if href.startswith("http") else urljoin(self.base_url, href)
                filename = abs_url.split("/")[-1]
                ext = filename.split(".")[-1].lower() if "." in filename else ""
                found.append({
                    "title": filename,
                    "author": "Unknown Author",
                    "url": abs_url,
                    "filename": filename,
                    "file_type": ext
                })

        logger.info(f"HTML fallback: event {event_id} yielded {len(found)} attachments")
        return found

    def _extract_speakers(self, contrib: Dict[str, Any]) -> str:
        """
        Extract speaker/presenter names from a contribution JSON object.
        Handles various Indico export schemas:
        - speakers: [{fullName|full_name|name|display_name}]
        - personLinks: [{fullName|name}]
        - primaryAuthors / coAuthors: [{fullName|name}]
        - presenter / speaker: dict or string
        Returns a comma-separated string or 'Unknown Author'.
        """
        names = []

        def _add_from_list(lst):
            if not isinstance(lst, list):
                return
            for p in lst:
                if not isinstance(p, dict):
                    continue
                name = (
                    p.get("fullName")
                    or p.get("full_name")
                    or p.get("name")
                    or p.get("display_name")
                )
                if name:
                    # Remove affiliations in parentheses
                    name = re.sub(r"\s*\([^)]*\)", "", name).strip()
                    if name:
                        names.append(name)

        # Common fields
        _add_from_list(contrib.get("speakers"))
        _add_from_list(contrib.get("personLinks"))
        _add_from_list(contrib.get("primaryAuthors"))
        _add_from_list(contrib.get("coAuthors"))

        # Fallback: single presenter/speaker
        for key in ("presenter", "speaker"):
            val = contrib.get(key)
            if isinstance(val, dict):
                name = val.get("fullName") or val.get("name")
                if name:
                    name = re.sub(r"\s*\([^)]*\)", "", name).strip()
                    if name:
                        names.append(name)
            elif isinstance(val, str):
                name = re.sub(r"\s*\([^)]*\)", "", val).strip()
                if name:
                    names.append(name)

        # De-duplicate while preserving order
        if names:
            unique = list(dict.fromkeys(names))
            return ", ".join(unique)
        return "Unknown Author"

    # ------------------------ Public API ------------------------
    def get_authors(self, event):
        authors=set()

        author_keys =['creator', 'chairs']
        for k in author_keys:
            if k in event:
                if isinstance(event[k], list):
                    for person in event[k]:
                        authors.add(person['fullName'])
                elif isinstance(event[k], dict):
                    authors.add(event[k]['fullName'])
        
        return ', '.join(list(authors)) if authors else 'Unknown Author'
    

    def get_attachments(self, event_id, details,attachments_total):
        docs=[]
        for a in self._collect_attachments(details):
            doc={}
            doc['document_id']= f"{event_id}_{attachments_total}"
            download_url = a['download_url']
            content_type = a.get('content_type','')
            if not content_type: continue
            filename = a['filename']
            doc['download_url'] = download_url
            doc['filename'] = filename
            doc['abstract'] = 'none'
            
            #if there is an abstract store that 
            
            content, _ = self._download_file(download_url, self.session, self.max_file_bytes )
            
            if content:
                
                raw_text = self.get_raw_text(content_type,content)
                cleaned_text = self.clean_text(raw_text) if raw_text else ''
               
                #doc['raw_text'] = raw_text
                attachments_total += 1
                doc['cleaned_text'] = cleaned_text
            docs.append(doc)

        return docs, attachments_total
    def collect_subcategories(self):
        ids=[]
        url= f"{self.base_url}/category/{self.category_id}"
        soup=self.get_soup_parser(url)
        sub_urls = soup.find("ul", class_='category-list')
        for a in sub_urls.find_all('a'):
            id_ = a.get('href').split("/")[-2]
            ids.append(id_)
        return ids

    def extract_documents(self, limit: int = 50) -> List[Dict[str, Any]]:
        logger.info(f"Extracting documents from Indico category {self.category_id} (limit: {limit})")
        self._ensure_access()

        events=[]
        subcategs=self.collect_subcategories()
        for categ_id in subcategs:
            events.extend(self._fetch_category_events(categ_id, limit))
        

        dataset: List[Dict[str, Any]] = []

        current_versions=self.faiss.get_indico_ids()
        for ct, event in enumerate(events):
            doc = {}
            event_id = event.get("id")

            if current_versions.get(int(event_id)):
                logger.warning(f"Already logged {event_id}, moving on")
                continue
            
          
            title = event.get("title", "Unknown Title")
            author = self.get_authors(event)#event.get("author", "Unknown Author")
            start_date = event.get("startDate").get('date', "Unknown Date")
            start_time = event.get("startDate").get('time', "Unknown Time")
 
            doc['document_id'] = event_id
            doc['meeting_name'] = title #meeting name
            doc['conveners'] = author  #conveners
            doc['start_date'] = start_date
            doc['start_time'] = start_time
            doc['location'] = event.get("location", '')
            doc['event_description']=event.get("description",'')
            doc["source"]= "indico"
            doc['event_url']=event.get('url','')
            dataset.append(doc)

            try:
                logger.info(f"Processing event: {title} ({event_id})")
                
                details = self._fetch_event_details(event_id)
                
                #logger.error("GOT DETAILS FROM WEBPAGE AS {details}")
                attachments_total = 0
                

                # Top-level attachments from export
                top_attachments = details.get("attachments", []) if details else [] #for 10575 there are no attachments
                
                for a in top_attachments:
                    doc={}
                    download_url = a.get('download_url','')
                    
                    content, _ = self._download_file(download_url, self.session, self.max_file_bytes )
                    if content:
                        raw_text = self.get_raw_text(a['content_type'],content)
                        #cleaned_text = self.clean_text(raw_text) if raw_text else ''
                        doc['raw_text'] = raw_text
                        doc["document_id"]=f"{event_id}_{attachments_total}"
                        dataset.append(doc)

                        attachments_total += 1

                # Contribution-level attachments from export
                contribs = details.get("contributions", []) if details else []
                
                if not contribs:
                    
                    docs ,attachments_total= self.get_attachments(event_id, details, attachments_total)
                    
                    #only put one, BASED ON EVALUATION
                    #TO DO START EVALUATION 
                    if docs: #only if theres an attachment store it
                        for doc in docs:
                            dataset.append(doc)
                    
                else:
                    for  contrib in  contribs:
                        doc={}
                        c_title = contrib.get("title") or title
                        c_author = self._extract_speakers(contrib) or author #here gets the sub titles and sub authors
                    
                        doc['presentation_title'] = c_title #presentation_title
                        doc['speaker_name'] = c_author #speaker_name

                        docs , attachments_total= self.get_attachments(event_id, contrib, attachments_total)
                        if docs:
                            doc.update(docs[0])
                        else:
                            doc['document_id'] = f"{event_id}_{attachments_total}" 
                            attachments_total += 1 
                        dataset.append(doc)

                # Fallback to HTML scrape if export yielded nothing
                if attachments_total == 0:
                    logger.warning("Failback to HTML")
                    html_atts = self._scrape_event_attachments_html(event_id)
                    for a in html_atts:
                        if a["file_type"] not in ("pdf", "pptx"):
                            continue
                        # Download and process by raw URL
                        try:
                            resp = self.session.get(a["url"], timeout=90)
                            if self._is_cloudflare_challenge(resp):
                                logger.warning(f"Cloudflare blocked attachment download: {a['url']}")
                                continue
                            resp.raise_for_status()
                            content = resp.content
                            if a["file_type"] == "pdf":
                                raw_text = self.extract_text_from_pdf(content)
                            else:
                                raw_text = self.extract_text_from_pptx(content)
                            if raw_text and raw_text.strip():
                                parsed_path = urlparse(a["url"]).path
                                doc_id = "/".join([p for p in parsed_path.split("/") if p][-3:])
                                cleaned_text = self.clean_text(raw_text)
                                dataset.append({
                                    "document_id": f"{doc_id}_{attachments_total}",
                                    #"raw_text": raw_text,
                                    "cleaned_text": cleaned_text,
                                    "content_type": "application/pdf" if a['file_type']=='pdf' else "presentationml.presentation" 
                                })
                                attachments_total += 1
                        except Exception as e:
                            logger.error(f"HTML fallback: error processing {a['url']}: {e}")

                logger.info(f"Event {event_id}: collected {attachments_total} attachments")
                
                #if enumerate get to 50: yield dataset then reset dataset 
                #evert 40 mins save dataset

            except Exception as e:
                logger.error(f"Error processing event {event_id}: {e}") #22667

        logger.info(f"Extracted {len(dataset)} Indico documents")
        
        return dataset