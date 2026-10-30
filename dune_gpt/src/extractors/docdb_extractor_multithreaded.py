import collections
import functools
import json
import mimetypes
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Set, Iterator
from urllib.parse import parse_qs, urlparse, urljoin
import time
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from src.extractors.indico_extractor import DEFAULT_COOKIES_PATH, DEFAULT_UA, HAS_CLOUDSCRAPER
from urllib3.util.retry import Retry
from curl_cffi import requests

from .base import BaseExtractor
from .session import Session
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
# USING THE INDICO AS AN EXAMPLE MAKE THE DOCDB EXTRACTION WORK THE SAME WAY WITH SINGLE SIGN ON. WE MIGHT ACTUALLY
# ABSTRACT OUT THE CODE IN INDICO_EXTRACTOR
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
        faiss= None,
        max_retries: int = 5,
        timeout_sec: int = 30,
        max_pool: int = 20,
        max_file_bytes: int = 50 * 1024 * 1024,  # 50 MB
        max_workers_pages: int = 6,
        max_workers_files: int = 4,
    ):

        self.username = DUNE_DOCDB_USERNAME
        self.password = DUNE_DOCDB_PASSWORD
        if not self.username or not self.password:
            raise ValueError("DocDB credentials not provided")
        self.faiss=faiss

        # Example base: "https://docs.dunescience.org/cgi-bin/sso/"
        self.base_url = DOCDB_BASE_URL
        self.doc_url = urljoin(self.base_url, "ShowDocument?docid=")
        self.timeout_sec = timeout_sec
        self.session = self._build_session()
        self.max_file_bytes = max_file_bytes
        self.max_workers_pages = max_workers_pages
        self.max_workers_files = max_workers_files
        self.start_time = time.time()

        # Lightweight counters for the current DocDB indexing run.
        self.run_stats = {
            "pages_checked": 0,
            "pages_missing": 0,
            "pages_unchanged": 0,
            "pages_failed": 0,
            "pages_prepared": 0,
        }

        super().__init__()

    def _build_session(self) -> requests.Session:
        """Configures HTTP retries, timeouts, and handles SSO login."""
        s = requests.Session(impersonate="chrome120", timeout=self.timeout_sec)

        self._sso_login(s)

        return s

    def _sso_login(self, session: requests.session) -> None:
        "Logs into DocDB using SAML SSO."
        try:
            homepage_url = urljoin(self.base_url, "DocumentDatabase")

            # Step 1: Hit DocDB to trigger the initial SSO redirect
            response = session.get(homepage_url)
            response.raise_for_status()

            # Step 2: Select username/password as the login method
            soup = BeautifulSoup(response.text, "html.parser")
            auth_form = soup.find("form")
            if not auth_form:
                raise RuntimeError("Initial SSO redirect form not found.")

            auth_url = urljoin(response.url, auth_form.get("action"))
            response = session.get(
                auth_url,
                params={"pfidpadapterid": "ad..FormBased", "rememberChoice": "false"},
            )
            response.raise_for_status()

            # Step 3: Submit credentials to the login form
            soup = BeautifulSoup(response.text, "html.parser")
            login_form = soup.find("form")
            if not login_form:
                raise RuntimeError("Login credentials form not found.")

            login_url = urljoin(response.url, login_form.get("action"))
            login_data = {
                "pf.username": self.username,
                "pf.pass": self.password,
                "pf.ok": "clicked",
                "pf.adapterId": "FormBased",
                "pf.cancel": "",
            }
            response = session.post(login_url, data=login_data)
            response.raise_for_status()

            # Step 4: Extract SAML token and POST it back to DocDB
            soup = BeautifulSoup(response.text, "html.parser")
            saml_form = soup.find("form")
            saml_response = soup.select_one('input[name="SAMLResponse"]')
            relay_state = soup.select_one('input[name="RelayState"]')
            if not (saml_form and saml_response and relay_state):
                raise PermissionError(
                    "SAML attributes missing. Credentials may be incorrect."
                )

            saml_data = {
                "RelayState": relay_state.get("value", ""),
                "SAMLResponse": saml_response.get("value", ""),
            }

            post_url = urljoin(response.url, saml_form.get("action"))
            auth_response = session.post(post_url, data=saml_data)
            auth_response.raise_for_status()

            # Step 5: Verify we reached DocDB instead of getting stuck on an SSO page
            test_response = session.get(homepage_url)
            if ("pingprod" in test_response.url):
                raise PermissionError("SSO login failed.")

        except Exception:
            # Close session on failure to prevent memory leaks in multithreaded runs
            session.close()
            raise

    def _detect_type(self, url: str, headers: dict) -> str:
        ct = headers.get("content-type", "").split(";")[0].strip().lower()
        if ct in ("", "application/octet-stream"):
            guess, _ = mimetypes.guess_type(url)
            return (guess or ct or "")
        return ct

    def check_document_page(self, docid: int, return_html: bool = False):
        """Return 'exists', 'missing', 'unauthorized', or 'error' for a ShowDocument page.
        Use GET + HTML structure and negative markers to avoid false 'missing'."""
        url = f"{self.doc_url}{docid}"
        try:
            g = self.session.get(url, allow_redirects=True)

            def result(status):
                return (status, g.text) if return_html and status in ("exists", "inconclusive") else (
                    (status, None) if return_html else status
                )
            if g.status_code in (401, 403):
                return result("unauthorized")
            if g.status_code == 404:
                return result("missing")
            if g.status_code != 200:
                return result("error")

            soup = BeautifulSoup(g.text, "html.parser")

            # Positive markers
            title_div = soup.find("div", id="DocTitle")
            has_retrieve = soup.find("a", href=re.compile(r"RetrieveFile\?docid=\d+"))
            # Negative markers typical of DocDB "not found" pages
            not_found = soup.find(string=re.compile(r"(no such document|document not found|unknown document|invalid docid)", re.I))

            if not_found:
                return result("missing")
            if title_div or has_retrieve:
                return result("exists")

            # If 200 but inconclusive, lean 'exists' so we don't cap too low
            return result("inconclusive")
        except requests.RequestException as e:
            logger.exception(f"Error checking doc {docid}: {e}")
            return ("error", None) if return_html else "error"

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

    def extract_document_links_and_metadata(self, webpage_url: str, html: Optional[str] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Extract RetrieveFile links and per-link metadata from a ShowDocument page (latest revision)."""
        try:
            logger.debug(f"Parsing {webpage_url}")

            if html is None:
                response = self.session.get(webpage_url)
                if not response.ok:
                    return [], []
                html = response.text

            soup = BeautifulSoup(html, 'html.parser')
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

            # Topics
            topics_div = soup.find("div", id='Topics')
            topics = []
            if topics_div:
                for t in topics_div.find_all('a', href=re.compile(r'ListBy\?topicid=\d+')):
                    topic_text = t.get_text(strip=True)
                    if topic_text:
                        topics.append(topic_text)
            topic = ', '.join(topics) if topics else 'Unspecified Topic'

            # Abstract
            abstract_div = soup.find('div', id='Abstract')
            abstract = abstract_div.dd.get_text(strip=True) if abstract_div and abstract_div.dd else 'No Recorded Abstract'

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
                full_url = href if href.startswith('http') else urljoin(self.base_url, href)
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

    def iter_changed_docdb_pages(
        self,
        start: int,
        limit: int,
        existing_versions: Dict[str, int],
        max_missing: int = 1000,
    ) -> Iterator[Tuple[int, List[str], List[Dict[str, Any]]]]:
        """Scan ascending DocDB IDs and yield new or updated documents."""
        self.run_stats = dict.fromkeys(self.run_stats, 0)
        docid = start
        consecutive_missing = 0

        while (limit == -1 or docid < start + limit) and consecutive_missing < max_missing:
            if docid % 4 == 0:
                time.sleep(3)

            status, html = self.check_document_page(docid, return_html=True)
            self.run_stats["pages_checked"] += 1

            if status == "unauthorized":
                raise PermissionError("Unauthorized to access DocDB")

            if status == "missing":
                self.run_stats["pages_missing"] += 1
                consecutive_missing += 1
                docid += 1
                continue

            if status == "error":
                self.run_stats["pages_failed"] += 1
                logger.error(
                    f"DocDB {docid}: page check failed. "
                    f"Skipping and continuing."
                )
                docid += 1
                continue

            if status not in ("exists", "inconclusive") or html is None:
                raise RuntimeError(f"Unexpected DocDB status for {docid}: {status}")

            consecutive_missing = 0
            page_url = f"{self.doc_url}{docid}"
            try:
                links, metadata = self.extract_document_links_and_metadata(
                    page_url, html=html
                )

                if not links and not metadata:
                    self.run_stats["pages_failed"] += 1
                    logger.warning(
                        f"DocDB {docid} has no attachments or metadata "
                        f"could not be extracted. Skipping."
                    )
                elif len(links) != len(metadata):
                    self.run_stats["pages_failed"] += 1
                    logger.error(
                        f"DocDB {docid}: attachment/metadata count mismatch. "
                        f"Skipping."
                    )
                else:
                    versions = set()
                    for item in metadata:
                        version = str(item.get("docdb_version", ""))
                        if not version.isdigit():
                            raise ValueError(
                                f"Invalid version for DocDB {docid}: {version}"
                            )
                        versions.add(int(version))

                    if len(versions) != 1:
                        raise ValueError(
                            f"Inconsistent attachment versions for "
                            f"DocDB {docid}: {versions}"
                        )

                    current_version = versions.pop()
                    indexed_version = existing_versions.get(str(docid))

                    if indexed_version is None or current_version > int(indexed_version):
                        yield docid, links, metadata
                    else:
                        self.run_stats["pages_unchanged"] += 1
                        logger.debug(
                            f"DocDB {docid} version {current_version} "
                            f"already indexed. Skipping."
                        )

            except PermissionError:
                raise
            except Exception as e:
                self.run_stats["pages_failed"] += 1
                logger.error(
                    f"DocDB {docid}: metadata processing failed: {e}. "
                    f"Skipping and continuing."
                )

            docid += 1

    def iter_prepared_docdb_documents(
        self,
        start: int,
        limit: int,
        existing_versions: Dict[str, int],
        chunk_size: int,
        max_missing: int = 1000,
    ) -> Iterator[Tuple[int, int, List[Dict[str, Any]]]]:
        """Yield only fully extracted new or updated DocDB documents."""
        for docid, links, metadata in self.iter_changed_docdb_pages(
            start=start,
            limit=limit,
            existing_versions=existing_versions,
            max_missing=max_missing,
        ):
            try:
                version = int(metadata[0]["docdb_version"])
                chunks = self.extract_complete_docdb_document(
                    docid=docid,
                    links=links,
                    metadata=metadata,
                    chunk_size=chunk_size,
                )

                if not chunks:
                    self.run_stats["pages_failed"] += 1
                    logger.warning(
                        f"DocDB {docid} version {version}: "
                        f"extraction incomplete; retaining indexed version."
                    )
                    continue

            except Exception as exc:
                self.run_stats["pages_failed"] += 1
                logger.error(
                    f"DocDB {docid}: document preparation failed: {exc}. "
                    f"Skipping and continuing."
                )
                continue

            self.run_stats["pages_prepared"] += 1
            yield docid, version, chunks

    def _enumerate_pages_latest_first(
        self,
        start: int,
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
        docid = start
        consecutive_missing = 0
        consecutive_seen_indexed = 0
        assert isinstance(start,int), f"start is type {type(start)}"
        assert isinstance(limit_pages, int), f"limit is type {type(docid)}"
        assert isinstance(docid, int), f"doc is of type {type(docid)}"
        def should_continue() -> bool:
            if limit_pages == -1:
                return (consecutive_missing < max_missing)
            return (
                docid < limit_pages + start
                and consecutive_missing < max_missing
            )
        parsed = 0
        while should_continue():
            if docid % 4 == 0:
                time.sleep(3)
            status = self.check_document_page(docid)
            logger.debug(f"Status for {docid} = {status}")

            if status == "exists":
                # reset missing counter
                parsed += 1
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
                    page_url = f"{self.doc_url}{docid}"

                    resp = self.session.get(page_url)

                    # if there is no RetrieveFile link on that page, treat as "missing"
                    if "RetrieveFile?docid=" not in resp.text:
                        consecutive_missing += 1
                        logger.debug(f"DocDB {docid} has no attachments, skipping")
                        # do not reset consecutive_seen_indexed
                        docid += 1
                        continue

                    # otherwise it's truly a page with attachments → queue it
                    logger.debug(f"Logging {page_url} to store")
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
        return existing_pages, parsed


    def extract_complete_docdb_document(
        self,
        docid: int,
        links: List[str],
        metadata: List[Dict[str, Any]],
        chunk_size: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """Prepare every attachment before allowing document replacement."""
        if not links or len(links) != len(metadata):
            logger.error(f"DocDB {docid}: invalid attachment list")
            return None

        if not chunk_size or chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        dataset = []
        chunk_number = 0
        attachments_extracted = 0

        for link, md in zip(links, metadata):
            try:
                result = self._download_file(
                    link, self.session, self.max_file_bytes
                )
                if result is None or result[0] is None:
                    raise RuntimeError("Attachment download failed")

                content, headers = result
                content_type = self._detect_type(link, headers)
                raw_text, document_type = self.get_raw_text(
                    content_type=content_type,
                    content=content,
                )

                if not raw_text or not raw_text.strip():
                    logger.warning(
                        f"DocDB {docid}: attachment "
                        f"{md.get('filename', link)} has no extractable text; "
                        f"skipping this attachment."
                    )
                    continue

                chunks = self.get_chunks(
                    raw_text.split(), chunk_size=chunk_size
                )
                if not chunks:
                    raise RuntimeError("Attachment produced no chunks")

                for chunk in chunks:
                    chunk_number += 1
                    chunk_data = {
                        "document_id": f"{docid}_{chunk_number}",
                        "vector_id": (
                            f"docdb_{docid}_{chunk_number}_"
                            f"{md.get('filename', '0')}"
                        ),
                        "document_type": document_type,
                        "cleaned_text": chunk,
                        "event_url": md["url"],
                        "content_type": content_type,
                    }

                    for key in (
                        "title", "author", "submitted_by", "updated_by",
                        "content_last_modified_date",
                        "metadata_last_modified_date", "abstract",
                        "topic", "keywords", "created_date",
                        "source", "docdb_version", "filename",
                    ):
                        chunk_data[key] = md.get(key, "")

                    dataset.append(chunk_data)

                attachments_extracted += 1

            except Exception as exc:
                logger.error(
                    f"DocDB {docid}: failed to process attachment "
                    f"{md.get('filename', link)}: {exc}. "
                    f"Preserving previously indexed version."
                )
                return None

        logger.info(
            f"DocDB {docid}: prepared {len(dataset)} chunks "
            f"from {attachments_extracted} successfully extracted "
            f"attachments ({len(links)} total)"
        )
        return dataset

    def extract_documents(
        self,
        start: int = 0,
        limit: int = -1,
        indexed_document_ids: Optional[Set[int]] = None,
        mode: str = "incremental",
        stop_after_seen: int = 100,
        max_missing: int = 1000,
        latest_hint: Optional[int] = None, # NEW: optional manual starting hint (e.g., 34626)
        chunk_size: Optional[int] = None,
        existing_versions: Optional[Dict] = None,
    ) -> Iterator[Tuple[int, List[Dict[str, Any]], int]]:

        if existing_versions is None:
            existing_versions = {}

        logger.info(
            f"Extracting documents from DocDB (mode: {mode}, limit: {limit}, latest_hint={latest_hint})"
        )
        pages, parsed = self._enumerate_pages_latest_first(
            start=start,
            limit_pages=limit,
            indexed_doc_ids=indexed_document_ids,
            stop_after_seen=stop_after_seen,
            max_missing=max_missing,
            mode=mode,
        )

        # Extract links and metadata from pages (parallel)
        all_links: List[str] = []
        all_metadata: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers_pages) as pool:
            future_to_page = {
                pool.submit(self.extract_document_links_and_metadata, page_url): page_url
                for page_url in pages
            }

            for fut in as_completed(future_to_page):
                page_url = future_to_page.pop(fut)
                try:
                    links, metadata = fut.result()
                    for l, m in zip(links, metadata):
                        if int(m.get("docdb_version", -1)) <= int(
                            existing_versions.get(m.get("doc_id"), -1)
                        ):
                            continue
                        all_links.append(l)
                        all_metadata.append(m)
                except Exception:
                    logger.exception(f"Error processing document page {page_url}")

        logger.info(f"Extracted {len(all_links)} attachments to download.")

        dataset: List[Dict[str, Any]] = []
        existing_ids: Dict[str, str] = {}
        last_yield_time = time.time()

        # Download attachments (parallel)
        with ThreadPoolExecutor(max_workers=self.max_workers_files) as pool:
            futures = {
                pool.submit(self._download_file, link, self.session, self.max_file_bytes): (
                    link,
                    metadata,
                )
                for link, metadata in zip(all_links, all_metadata)
            }

            for fut in as_completed(futures):
                link, metadata = futures.pop(fut)

                try:
                    result = fut.result()
                    if not result:
                        logger.warning(f"Couldn't fetch attachment content from {link}")
                        continue

                    content, headers = result
                    ct = self._detect_type(link, headers)
                    qs = parse_qs(urlparse(link).query)
                    doc_id = qs.get("docid", ["unknown"])[0]

                    # Process content to text
                    raw_text, document_type = self.get_raw_text(
                        content_type=ct, content=content
                    )
                    cleaned_text_list = raw_text.split()
                    chunks = self.get_chunks(cleaned_text_list, chunk_size=chunk_size)

                    for chunk in chunks:
                        root_id = doc_id
                        child_id = int(str(existing_ids.get(doc_id, "0_0")).split("_")[-1])
                        new_chunk_id = f"{root_id}_{child_id+1}"
                        existing_ids[doc_id] = new_chunk_id

                        # give each vector a unique "document_id" – e.g. include filename
                        unique_id = f"docdb_{new_chunk_id}_{metadata.get('filename','0')}"

                        # Downstream payload: preserving original keys to avoid breaking external schema
                        dataset.append(
                            {
                                "document_id": new_chunk_id,
                                "vector_id": unique_id,
                                "document_type": document_type,
                                "cleaned_text": chunk,
                                "event_url": metadata["url"],
                                "title": metadata["title"],
                                "author": metadata["author"],
                                "submitted_by": metadata["submitted_by"],
                                "updated_by": metadata["updated_by"],
                                "content_last_modified_date": metadata[
                                    "content_last_modified_date"
                                ],
                                "metadata_last_modified_date": metadata[
                                    "metadata_last_modified_date"
                                ],
                                "abstract": metadata["abstract"],
                                "topic": metadata["topic"],
                                "keywords": metadata["keywords"],
                                "created_date": metadata["created_date"],
                                "source": metadata["source"],
                                "docdb_version": metadata.get("docdb_version", ""),
                                "filename": metadata.get("filename", ""),
                                "content_type": ct,
                            }
                        )

                        logger.debug(
                            f"Processed DocDB document chunk: {new_chunk_id} - Title: {metadata['title']} Chunk size: {len(chunk.split())}"
                        )

                    # Log attachment details: doc_id, filename, version, title, URL
                    fname = metadata.get("filename", "<no filename>")
                    version = metadata.get("docdb_version", "<no version>")
                    url = metadata.get("url", "")

                    if chunks:
                        logger.debug(
                            f"Processed attachment: doc={doc_id} "
                            f"file={fname} version={version} "
                            f"title=\"{metadata['title']}\"\n"
                        )
                    else:
                        logger.debug(
                            f"Skipped text-extraction (metadata-only): doc={doc_id} "
                            f"file={fname} version={version} "
                            f"title=\"{metadata['title']}\"\n"
                            f"    URL={url}"
                        )

                    if time.time() - last_yield_time >= 1800:
                        yield len(existing_ids), dataset, parsed
                        dataset = []
                        last_yield_time = time.time()

                except Exception:
                    logger.exception(f"Error downloading attachment {link}")

        logger.info(f"Extracted {len(existing_ids)} unique DocDB documents.")
        if dataset:
            yield len(existing_ids), dataset, parsed
