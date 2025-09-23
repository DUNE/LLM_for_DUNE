import functools
import json
import mimetypes
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Set
from urllib.parse import parse_qs, urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from src.extractors.indico_extractor import DEFAULT_COOKIES_PATH, DEFAULT_UA, HAS_CLOUDSCRAPER
from urllib3.util.retry import Retry

from .base import BaseExtractor
from config import DUNE_DOCDB_USERNAME, DUNE_DOCDB_PASSWORD, DOCDB_BASE_URL, INDICO_COOKIES_FILE
from src.utils.logger import get_logger

logger = get_logger(__name__)
try:
    import cloudscraper  # pip install cloudscraper
    HAS_CLOUDSCRAPER = True
except Exception:
    HAS_CLOUDSCRAPER = False

# Matches "1 Aug 2025" or "18 August 2025"
DATE_PAT = re.compile(r'(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})')
#USING THE INDICO AS AN EXAMPLE MAKE THE DOCDB EXTRACTION WORK THE SAME WAY WITH SINGLE SIGN ON. WE MIGHT ACTUALLY 
#ABSTRACT OUT THE CODE IN INDICO_EXTRACTOR
def parse_date(date_str: str) -> str:
    """Convert '18 Aug 2025' or '18 August 2025' -> '2025-08-18' (ISO format)"""
    for fmt in ("%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(date_str, fmt).date().isoformat()
        except ValueError:
            continue
    return "Unknown Date"


class DocDBExtractor(BaseExtractor):
    """Extractor for DUNE DocDB documents (latest-first, backfill + weekly incrementals).
    - Downloads files (in memory), size-capped, and extracts text.
    - Uses ShowDocument (latest revision) pages only.
    """

    def __init__(
        self,
        faiss,
        max_retries: int = 5,
        timeout_sec: int = 20,
        max_pool: int = 20,
        max_file_bytes: int = 50 * 1024 * 1024,  # 50 MB
        max_workers_pages: int = 6,
        max_workers_files: int = 4,
    ):
        super().__init__()
        self.username = DUNE_DOCDB_USERNAME
        self.password = DUNE_DOCDB_PASSWORD
        if not self.username or not self.password:
            raise ValueError("DocDB credentials not provided")
        self.faiss=faiss

        # Example base: "https://docs.dunescience.org/cgi-bin/ShowDocument?docid="
        self.base_url = DOCDB_BASE_URL.rstrip("?&")
        self.session = self._build_session(max_retries, timeout_sec, max_pool)
        self.max_file_bytes = max_file_bytes
        self.max_workers_pages = max_workers_pages
        self.max_workers_files = max_workers_files

        parsed = urlparse(self.base_url)
        self.root_base = f"{parsed.scheme}://{parsed.netloc}"
        self.cgi_base = urljoin(self.root_base, "/cgi-bin/")

        # Early auth sanity check (GET to avoid friendly 200 pages)
        try:
            r = self.session.get(self.base_url + "1", allow_redirects=True)
            if r.status_code in (401, 403):
                raise PermissionError("Unauthorized to access DocDB (check credentials or ACLs)")
        except requests.RequestException as e:
            logger.warning(f"Could not perform initial DocDB auth check: {e}")



    def _build_session(self, max_retries: int, timeout: int, max_pool: int) -> requests.Session:
        s = requests.Session()
        s.auth = HTTPBasicAuth(self.username, self.password)
        s.headers.update({"User-Agent": "DUNE-DocDB-Extractor/1.2 (+python-requests)"})
        retries = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=max_pool, pool_maxsize=max_pool)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.request = functools.partial(s.request, timeout=timeout)  # default timeout
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
                # If SSO login is required for some content, user can navigate to /login and complete it
                page.goto(f"{self.base_url}/login", wait_until="domcontentloaded")

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

        test_url = f"{self.base_url}"
        r = self.session.get(test_url, timeout=30)
        logger.error(f"esure access {r}")

        if r.status_code == 200:
            self._auth_initialized = True
            logger.info("Access OK (API token/cookies).")
            return

        if self._is_cloudflare_challenge(r):
            logger.warning("Cloudflare challenge detected on export API.")
            # Try cloudscraper first
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

    def _detect_type(self, url: str, headers: dict) -> str:
        ct = headers.get("content-type", "").split(";")[0].strip().lower()
        if ct in ("", "application/octet-stream"):
            guess, _ = mimetypes.guess_type(url)
            return (guess or ct or "")
        return ct

    def check_document_page(self, docid: int) -> str:
        """Return 'exists', 'missing', 'unauthorized', or 'error' for a ShowDocument page.
        Use GET + HTML structure and negative markers to avoid false 'missing'."""
        url = f"{self.base_url}{docid}"
        try:
            g = self.session.get(url, allow_redirects=True)
            if g.status_code in (401, 403):
                return "unauthorized"
            if g.status_code == 404:
                return "missing"
            if g.status_code != 200:
                return "error"

            soup = BeautifulSoup(g.text, "html.parser")

            # Positive markers
            title_div = soup.find("div", id="DocTitle")
            has_retrieve = soup.find("a", href=re.compile(r"RetrieveFile\?docid=\d+"))

            # Negative markers typical of DocDB "not found" pages
            not_found = soup.find(string=re.compile(r"(no such document|document not found|unknown document|invalid docid)", re.I))

            if not_found:
                return "missing"
            if title_div or has_retrieve:
                return "exists"

            # If 200 but inconclusive, lean 'exists' so we don't cap too low
            return "exists"
        except requests.RequestException as e:
            logger.warning(f"Error checking doc {docid}: {e}")
            return "error"
    

    def document_page_exists(self, docid: int) -> bool:
        return self.check_document_page(docid) == "exists"

    def get_latest_docid(self, start: int = 1, max_cap: int = 2_000_000) -> int:
        """
        Find the highest available DocDB ID via exponential + binary search,
        but begin probing at `start` (your hint) rather than at 1.
        """
        # Initialize low bound to your hint
        low = start

        # Check the status at 'low'
        status = self.check_document_page(low)
        if status == "unauthorized":
            raise PermissionError("Unauthorized to access DocDB (check credentials or ACLs)")

        # If the hinted page doesn't exist, exponentially grow 'low' until we find an existing page
        if status != "exists":
            while low < max_cap and self.check_document_page(low) != "exists":
                low = min(low * 2, max_cap)
            if low >= max_cap:
                raise RuntimeError("Could not find any existing DocDB IDs up to cap")
            high = low
        else:
            # Hint exists; now find an upper bound 'high' by doubling until a missing page
            high = low
            while high < max_cap and self.document_page_exists(high):
                low, high = high, min(high * 2, max_cap)

        # Binary search the last-existing ID in (low, high]
        while low + 1 < high:
            mid = (low + high) // 2
            status = self.check_document_page(mid)
            if status == "exists":
                low = mid
            elif status == "missing":
                high = mid
            elif status == "unauthorized":
                raise PermissionError("Unauthorized to access DocDB (check credentials or ACLs)")
            else:
                # treat any other error as "missing" to make progress
                high = mid

        return low

    def extract_document_links_and_metadata(self, webpage_url: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Extract RetrieveFile links and per-link metadata from a ShowDocument page (latest revision)."""
        try:
            logger.info(f"Parsing {webpage_url}")

            response = self.session.get(webpage_url)
            if not response.ok:
                return [], []

            soup = BeautifulSoup(response.text, 'html.parser')
            document_links: List[str] = []
            metadata_list: List[Dict[str, Any]] = []

            # Title
            title_div = soup.find('div', id='DocTitle')
            title = title_div.h1.get_text(strip=True) if title_div and title_div.h1 else 'Unknown Title'

            # Authors (join if multiple)
            authors_div = soup.find('div', id='Authors')
            authors = []
            if authors_div:
                for a in authors_div.find_all('a', href=re.compile(r'ListBy\?authorid=\d+')):
                    txt = a.get_text(strip=True)
                    if txt:
                        authors.append(txt)
            author = ", ".join(authors) if authors else "Unknown Author"

            #Topics
            topics_div = soup.find("div", id='Topics')
            topics = []
            if topics_div:
                for t in topics_div.find_all('a', href=re.compile(r'ListBy\?topicid=\d+')):
                    topic_text = t.get_text(strip=True)
                    if topic_text:
                        topics.append(topic_text)
            topic = ', '.join(topics) if topics else 'Unspecified Topic'

            #Abstract
            abstract_div = soup.find('div', id='Abstract')
            abstract = abstract_div.dd.get_text(strip=True) if abstract_div.dd else 'No Recorded Abstract'

            # Creation date
            created_date = "Unknown Date"
            dt_created = soup.find('dt', string=re.compile(r'Document Created', re.I))
            if dt_created and dt_created.find_next_sibling('dd'):
                full_created = dt_created.find_next_sibling('dd').get_text(" ", strip=True)
                m = DATE_PAT.search(full_created)
                if m:
                    created_date = parse_date(m.group(1))

            # Last modified date
            content_last_modified_date =  str(datetime.min)
            dt_mod = soup.find('dt', string=re.compile(r'Contents Revised', re.I))
            if dt_mod and dt_mod.find_next_sibling('dd'):
                full_mod = dt_mod.find_next_sibling('dd').get_text(" ", strip=True)
                m = DATE_PAT.search(full_mod)
                if m:
                    content_last_modified_date = parse_date(m.group(1))

            # Submission author
            submission_author = "Submission Author"
            dt_mod = soup.find('dt', string=re.compile(r'Submitted by:', re.I))
            if dt_mod and dt_mod.find_next_sibling('dd'):
                name = dt_mod.find_next_sibling('dd').get_text(" ", strip=True)
                if name:
                    submission_author = name

            # Revsior
            revisor = "Revisor "
            dt_mod = soup.find('dt', string=re.compile(r'Updated by:', re.I))
            if dt_mod and dt_mod.find_next_sibling('dd'):
                name = dt_mod.find_next_sibling('dd').get_text(" ", strip=True)
                if name:
                    revisor = name

            metadata_last_modified_date = str(datetime.min)
            dt_mod = soup.find('dt', string=re.compile(r'Metadata Revised', re.I))
            if dt_mod and dt_mod.find_next_sibling('dd'):
                full_mod = dt_mod.find_next_sibling('dd').get_text(" ", strip=True)
                m = DATE_PAT.search(full_mod)
                if m:
                    metadata_last_modified_date = parse_date(m.group(1))

            all_keywords = []
            keywords = soup.find('div', id_ = 'Keywords')
            if keywords and keywords.find_next_sibling('dd'):
                all_keywords.append(keywords.find_next_sibling('dd').get_text(" ", strip=True))
            all_keywords = ', '.join(all_keywords) if all_keywords else ''
            


            # Latest revision files (links on this page)
            for a in soup.find_all('a', href=True):
                href = a['href']
                if 'RetrieveFile?docid=' not in href:
                    continue
                full_url = href if href.startswith('http') else urljoin(self.cgi_base, href)
                qs = parse_qs(urlparse(full_url).query)
                doc_id = qs.get("docid", ["unknown"])[0]
                version = qs.get("version", qs.get("ver", [""]))[0]
                filename = qs.get("filename", [""])[0]

                document_links.append(full_url)
                metadata_list.append({
                    'url': full_url,
                    'title': title,
                    'abstract': abstract,
                    'author': author,
                    'submitted_by': submission_author,
                    'updated_by': revisor,
                    'topic': topic,
                    'created_date': created_date,
                    'content_last_modified_date': content_last_modified_date,
                    'metadata_last_modified_date': metadata_last_modified_date,
                    'source': 'docdb',
                    'doc_id': doc_id,
                    'docdb_version': version,
                    'filename': filename,
                    'keywords': all_keywords,
                })

            return document_links, metadata_list

        except Exception as e:
            logger.error(f"Error extracting from {webpage_url}: {e}")
            return [], []


    def _enumerate_pages_latest_first(
        self,
        limit_pages: Optional[int],
        indexed_doc_ids: Optional[Set[int]],
        stop_after_seen: int,
        max_missing: int,
        mode: str,
    ) -> List[str]:
        """
        Walk backwards from the latest DocDB ID, skipping any IDs
        in `indexed_doc_ids`, until we've either collected `limit_pages`
        *pages that actually have attachments*, hit too many missing IDs,
        or seen enough already‐indexed in a row (incremental mode).
        """

        indexed_doc_ids = indexed_doc_ids or set()

        existing_pages: List[str] = []
        docid = 0
        consecutive_missing = 0
        consecutive_seen_indexed = 0

        def should_continue() -> bool:
            return (
                docid < limit_pages
                and consecutive_missing < max_missing
            )

        while should_continue():
            status = self.check_document_page(docid)

            if status == "exists":
                # reset missing counter
                consecutive_missing = 0

                if docid in indexed_doc_ids:
                    # already indexed → skip
                    consecutive_seen_indexed += 1
                    logger.debug(f"Skipping DocDB {docid} (already indexed)")
                    # in incremental mode, stop if we've seen enough in a row
                    if mode == "incremental" and consecutive_seen_indexed >= stop_after_seen:
                        logger.info(
                            f"Stopped after {consecutive_seen_indexed} "
                            "already‐indexed pages in a row."
                        )
                        break

                else:
                    # **INLINE ATTACHMENT CHECK**
                    page_url = f"{self.base_url}{docid}"
                    resp = self.session.get(page_url)
                    # if there is no RetrieveFile link on that page, treat as "missing"
                    if "RetrieveFile?docid=" not in resp.text:
                        consecutive_missing += 1
                        logger.debug(f"DocDB {docid} has no attachments, skipping")
                        # do not reset consecutive_seen_indexed
                        docid += 1
                        continue

                    # otherwise it's truly a page with attachments → queue it
                    existing_pages.append(page_url)
                  
                    consecutive_seen_indexed = 0

            elif status == "missing":
                # track missing pages
                consecutive_missing += 1
                consecutive_seen_indexed = 0

            elif status == "unauthorized":
                raise PermissionError("Unauthorized to access DocDB")

            else:
                # other errors → just reset seen counter
                consecutive_seen_indexed = 0

            docid += 1

        logger.info(f"Found {len(existing_pages)} DocDB pages with attachments to process")
        return existing_pages

   

    def extract_documents(
        self,
        limit: int = 50,
        indexed_doc_ids: Optional[Set[int]] = None,
        mode: str = "incremental",
        stop_after_seen: int = 100,
        max_missing: int = 1000,
        latest_hint: Optional[int] = None,  # NEW: optional manual starting hint (e.g., 34626)
    ) -> List[Dict[str, Any]]:
        logger.info(f"Extracting events from DocDB (mode: {mode}, limit: {limit}, latest_hint={latest_hint})")
        #self._ensure_access()
        indexed_versions: Dict[int,int] = {}
        for rec in self.faiss.metadata_store.values():
            try:
                did = int(rec['doc_id'])
                
                v   = int(rec.get('docdb_version', '0'))
            except (KeyError, ValueError):
                continue
            indexed_versions[did] = max(indexed_versions.get(did, 0), v)
        
        pages = self._enumerate_pages_latest_first(
            limit_pages=limit,
            indexed_doc_ids=indexed_doc_ids,
            #indexed_doc_versions=indexed_versions,
            stop_after_seen=stop_after_seen,
            max_missing=max_missing,
            mode=mode,
        )

        # Extract links and metadata from pages (parallel)
        all_links: List[str] = []
        all_metadata: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers_pages) as pool:
            future_to_page = {pool.submit(self.extract_document_links_and_metadata, page_url): page_url for page_url in pages}
            for fut in as_completed(future_to_page):
                page_url = future_to_page[fut]
                try:
                    links, metadata = fut.result()
                    all_links.extend(links)
                    all_metadata.extend(metadata)
                except Exception as e:
                    logger.error(f"Error processing page {page_url}: {e}")

        # Download files (parallel)
        documents: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers_files) as pool:
            futures = {pool.submit(self._download_file, link, self.session, self.max_file_bytes): (link, metadata) for link, metadata in zip(all_links, all_metadata)}
            for fut in as_completed(futures):
                link, metadata = futures[fut]
                try:
                    result = fut.result()
                    if not result:
                        continue
                    content, headers = result
                    ct = self._detect_type(link, headers)
                    qs = parse_qs(urlparse(link).query)
                    doc_id = qs.get("docid", ["unknown"])[0]
                    documents.append({
                        'doc_id': doc_id,
                        'content': content,
                        'content_type': ct,
                        'metadata': metadata
                    })
                except Exception as e:
                    logger.error(f"Error downloading document {link}: {e}")

        # Process content to text
        dataset: List[Dict[str, Any]] = []

        for ct, doc in enumerate(documents):
            
            try:
                raw_text = ""
                content_type = doc['content_type']
                metadata = doc['metadata']
                content = doc['content']

                raw_text = self.get_raw_text(content_type,content)
                if doc['doc_id']==126:print(metadata.get('filename','NONAME')) #DEBG
                

                # Always include a record; raw_text may be empty if unsupported
                cleaned_text = self.clean_text(raw_text) if raw_text else ""
                # give each vector a unique "document_id" – e.g. include filename
                unique_id = f"docdb_{doc['doc_id']}_{metadata.get('filename','')}"

                #PUT IN DOCUMENTATION THAT THESE ARE THE FIELDS
                dataset.append({
                    'document_id': doc['doc_id'],
                    'vector_id': unique_id,
                    'raw_text': raw_text,
                    'cleaned_text': cleaned_text,
                    'url': metadata['url'],
                    'title': metadata['title'],
                    'author': metadata['author'],
                    'submitted_by': metadata['submitted_by'],
                    'updated_by': metadata['updated_by'],
                    'content_last_modified_date': metadata['content_last_modified_date'],
                    'metadata_last_modified_date': metadata['metadata_last_modified_date'],
                    'abstract': metadata['abstract'],
                    'topic': metadata['topic'],
                    'keywords': metadata['keywords'],
                    #STORE KEYWORDS
                    'created_date': metadata['created_date'],
                    'source': metadata['source'],
                    'docdb_version': metadata.get('docdb_version', ''),
                    #HWHEN UPDATING THE VERISONS ALSO MAKE SURE YOU UPDATE THE ABSTRACT IF THE META DATA IS CHANGED
                    'filename': metadata.get('filename', ''),
                    'content_type': content_type,

                })
                if raw_text:
                    logger.info(f"Processed DocDB event: {doc['doc_id']} -  Title: {metadata['title']}")
                else:
                    logger.info(f"Added metadata-only (no text) for DocDB {doc['doc_id']} - {metadata['title']}")

                # Log attachment details: doc_id, filename, version, title, URL
                fname = metadata.get('filename', '<no filename>')
                version = metadata.get('docdb_version', '<no version>')
                url = metadata.get('url', '')

                if raw_text:
                    logger.info(
                        f"Processed attachment: doc={doc['doc_id']} "
                        f"file={fname} version={version} "
                        f"title=\"{metadata['title']}\"\n"
                        #f"    URL={url}"
                    )
                else:
                    logger.info(
                        f"Skipped text-extraction (metadata-only): doc={doc['doc_id']} "
                        f"file={fname} version={version} "
                        f"title=\"{metadata['title']}\"\n"
                        f"    URL={url}"
                    )


            except Exception as e:
                logger.error(f"Error processing document {doc.get('doc_id','unknown')}: {e}")

        logger.info(f"Extracted {len(dataset)} DocDB events")
        return dataset