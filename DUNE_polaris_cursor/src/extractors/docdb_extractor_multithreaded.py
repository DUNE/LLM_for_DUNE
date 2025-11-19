import collections
import functools
import json
import mimetypes
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Set
from urllib.parse import parse_qs, urlparse, urljoin
import time
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from src.extractors.indico_extractor import DEFAULT_COOKIES_PATH, DEFAULT_UA, HAS_CLOUDSCRAPER
from urllib3.util.retry import Retry

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


class DocDBExtractor(BaseExtractor, Session):
    """Extractor for DUNE DocDB documents (latest-first, backfill + weekly incrementals).
    - Downloads files (in memory), size-capped, and extracts text.
    - Uses ShowDocument (latest revision) pages only.
    """

    def __init__(
        self,
        faiss= None,
        max_retries: int = 5,
        timeout_sec: int = 20,
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

        # Example base: "https://docs.dunescience.org/cgi-bin/ShowDocument?docid="
        self.base_url = DOCDB_BASE_URL.rstrip("?&")
        self.session = self._build_session()
        self.max_file_bytes = max_file_bytes
        self.max_workers_pages = max_workers_pages
        self.max_workers_files = max_workers_files

        parsed = urlparse(self.base_url)
        self.root_base = f"{parsed.scheme}://{parsed.netloc}"
        self.cgi_base = urljoin(self.root_base, "/cgi-bin/")
        self.start_time =time.time() 

        # Early auth sanity check (GET to avoid friendly 200 pages)
        try:
            r = self.session.get(self.base_url + "1", allow_redirects=True)
            if r.status_code in (401, 403):
                raise PermissionError("Unauthorized to access DocDB (check credentials or ACLs)")
        except requests.RequestException as e:
            logger.warning(f"Could not perform initial DocDB auth check: {e}")

        super().__init__()

    def _build_session(self) -> requests.Session:
        s = requests.Session()
        s.auth = HTTPBasicAuth(self.username, self.password)
        s.headers.update({"User-Agent": "DUNE-DocDB-Extractor/1.2 (+python-requests)"})
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.request = functools.partial(s.request, timeout=10)  # default timeout
        return s




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
                docid <= limit_pages + start
                and consecutive_missing < max_missing
            )

        while should_continue():
            if docid % 4 == 0:
                tiem.sleep(3)
            status = self.check_document_page(docid)
            logger.warning(f"Status for {docid} = {status}")
            

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
                    logger.info(f"Logging {page_url} to store")
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
        start: int=0,
        limit: int = -1,
        indexed_doc_ids: Optional[Set[int]] = None,
        mode: str = "incremental",
        stop_after_seen: int = 100,
        max_missing: int = 1000,
        latest_hint: Optional[int] = None,  # NEW: optional manual starting hint (e.g., 34626)
        chunk_size:int=None,
    ) -> List[Dict[str, Any]]:
        logger.info(f"Extracting events from DocDB (mode: {mode}, limit: {limit}, latest_hint={latest_hint})")
        pages = self._enumerate_pages_latest_first(
            start=start,
            limit_pages=limit,
            indexed_doc_ids=indexed_doc_ids,
            stop_after_seen=stop_after_seen,
            max_missing=max_missing,
            mode=mode,
        )

        documents_processed = collections.defaultdict(list)
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
                    documents_processed[page_url].extend(links)
                except Exception as e:
                    logger.error(f"Error processing page {page_url}: {e}")
                
                
        logger.info(f"{len(all_links)} were extracted: from {len(documents_processed)} websites")
        # Download files (parallel)
        documents: List[Dict[str, Any]] = []
        existing_ids=collections.defaultdict(int)
        with ThreadPoolExecutor(max_workers=self.max_workers_files) as pool:
            futures = {pool.submit(self._download_file, link, self.session, self.max_file_bytes): (link, metadata) for link, metadata in zip(all_links, all_metadata)}
            logger.info(f"Downloaded {len(futures)} links")
            for fut in as_completed(futures):
                link, metadata = futures[fut]
            
                try:
                    result = fut.result()
                    if not result:
                        logger.error(f"Couldn't fetch content from {link}")
                        continue
                    content, headers = result
                   
                    ct = self._detect_type(link, headers)
                    qs = parse_qs(urlparse(link).query)
                    doc_id = qs.get("docid", ["unknown"])[0]

                    raw_text, document_type = self.get_raw_text(content_type=ct, content=content)
                    cleaned_text_list = raw_text.split()
                    chunks = self.get_chunks(cleaned_text_list, chunk_size=chunk_size)
                    logger.info(f"Found {len(chunks)} chunks for {link}")
                    for chunk in chunks:
                        
                        root_id = doc_id
                        
                        child_id = int(existing_ids.get(doc_id,'0_0').split("_")[-1])
                        metadata['document_type'] = document_type
                        documents.append({
                            'document_id': f"{root_id}_{child_id+1}",
                            'cleaned_text': chunk,
                            'content_type': ct,
                            'metadata': metadata,
                            
                        })
                        existing_ids[doc_id] = f"{root_id}_{child_id+1}"
                        logger.info(f"Added chunk of size {len(chunk.split())} from {link} to documents")
                
                    logger.info(f"Finished extracting from {link}")
                except Exception as e:
                    logger.error(f"Error downloading document {link}: {e}")

        # Process content to text
        dataset: List[Dict[str, Any]] = []
        
        for ct, doc in enumerate(documents):
            
            try:
                content_type = doc['content_type']
                metadata = doc['metadata']
                cleaned_text = doc['cleaned_text']

                
                logger.info(f"doc is = {doc['document_id']}")
                

                
                # give each vector a unique "document_id" – e.g. include filename
                unique_id = f"docdb_{doc['document_id']}_{metadata.get('filename','0')}"
               
                #PUT IN DOCUMENTATION THAT THESE ARE THE FIELDS
                if cleaned_text:
                    dataset.append({
                        'document_id': doc['document_id'],
                        'vector_id': unique_id,
                        'document_type':metadata['document_type'],
                        'cleaned_text': cleaned_text,
                        'event_url': metadata['url'],
                        'title': metadata['title'],
                        'author': metadata['author'],
                        'submitted_by': metadata['submitted_by'],
                        'updated_by': metadata['updated_by'],
                        'content_last_modified_date': metadata['content_last_modified_date'],
                        'metadata_last_modified_date': metadata['metadata_last_modified_date'],
                        'abstract': metadata['abstract'],
                        'topic': metadata['topic'],
                        'keywords': metadata['keywords'],
                        'created_date': metadata['created_date'],
                        'source': metadata['source'],
                        'docdb_version': metadata.get('docdb_version', ''),
                        'filename': metadata.get('filename', ''),
                        'content_type': content_type,
                    })

                if cleaned_text:
                    logger.info(f"Processed DocDB event: {doc['document_id']} -  Title: {metadata['title']} Chunk size: {len(doc['cleaned_text'].split())}")
                
                else:
                    logger.info(f"Not logging anything because metadata-only (no text) for DocDB {doc['document_id']} - {metadata['title']}")

                # Log attachment details: doc_id, filename, version, title, URL
                fname = metadata.get('filename', '<no filename>')
                version = metadata.get('docdb_version', '<no version>')
                url = metadata.get('url', '')

                if cleaned_text:
                    logger.info(
                        f"Processed attachment: doc={doc['document_id']} "
                        f"file={fname} version={version} "
                        f"title=\"{metadata['title']}\"\n"
                    )
                else:
                    logger.info(
                        f"Skipped text-extraction (metadata-only): doc={doc['document_id']} "
                        f"file={fname} version={version} "
                        f"title=\"{metadata['title']}\"\n"
                        f"    URL={url}"
                    )
                if time.time() - self.start_time % 1800 == 0 and ct > 0:
                    yield dataset
                    dataset.clear()



            except Exception as e:
                logger.error(f"Error processing document {doc.get('document_id','unknown')}: {e}")

        logger.info(f"Extracted {len(dataset)} DocDB attachments")

        
        yield len(existing_ids), dataset
