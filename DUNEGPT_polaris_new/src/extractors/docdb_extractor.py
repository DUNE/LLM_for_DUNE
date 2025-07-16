import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import re
from .base import BaseExtractor
from config import DUNE_DOCDB_USERNAME, DUNE_DOCDB_PASSWORD, DOCDB_BASE_URL
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DocDBExtractor(BaseExtractor):
    """Extractor for DUNE DocDB documents"""
    
    def __init__(self):
        super().__init__()
        self.username = DUNE_DOCDB_USERNAME
        self.password = DUNE_DOCDB_PASSWORD
        self.base_url = DOCDB_BASE_URL
        
        if not self.username or not self.password:
            raise ValueError("DocDB credentials not provided")
    
    def document_page_exists(self, docid: int) -> bool:
        """Check if a document page exists"""
        url = f"{self.base_url}{docid}"
        try:
            response = requests.get(url, auth=HTTPBasicAuth(self.username, self.password))
            return response.ok
        except Exception as e:
            logger.error(f"Error checking document page {docid}: {e}")
            return False
    
    def extract_document_links_and_metadata(self, webpage_url: str) -> tuple:
        """Extract document links and metadata from a webpage"""
        try:
            response = requests.get(webpage_url, auth=HTTPBasicAuth(self.username, self.password))
            if not response.ok:
                return [], []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            document_links = []
            metadata_list = []
            
            # Extract title
            title_div = soup.find('div', id='DocTitle')
            title = title_div.h1.get_text(strip=True) if title_div and title_div.h1 else 'Unknown Title'
            
            # Extract author
            author_div = soup.find('div', id='Authors')
            if author_div:
                author_link = author_div.find('a', href=re.compile(r'ListBy\?authorid=\d+'))
                author = author_link.get_text(strip=True) if author_link else 'Unknown Author'
            else:
                author = 'Unknown Author'
            
            # Extract creation date
            created_date_dd = soup.find('dt', string='Document Created:')
            if created_date_dd and created_date_dd.find_next_sibling('dd'):
                full_created_date = created_date_dd.find_next_sibling('dd').get_text(strip=True)
                updated_match = re.search(r'\d{2} \w+ \d{4}', full_created_date)
                updated = updated_match.group(0) if updated_match else 'Unknown Date'
            else:
                updated = 'Unknown Date'
            
            # Extract document links
            for row in soup.find_all('li'):
                link_tag = row.find('a', href=True)
                if not link_tag or 'RetrieveFile?docid=' not in link_tag['href']:
                    continue
                
                href = link_tag['href']
                full_url = href if href.startswith('http') else f"https://docs.dunescience.org{href}"
                
                document_links.append(full_url)
                metadata_list.append({
                    'url': full_url,
                    'title': title,
                    'author': author,
                    'updated': updated,
                    'source': 'docdb'
                })
            
            return document_links, metadata_list
            
        except Exception as e:
            logger.error(f"Error extracting from {webpage_url}: {e}")
            return [], []
    
    def extract_documents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Extract documents from DocDB"""
        logger.info(f"Extracting documents from DocDB (limit: {limit})")
        
        existing_pages = []
        consecutive_missing = 0
        max_missing = 50
        docid = 1
        
        # Find existing document pages
        while consecutive_missing < max_missing and docid <= limit:
            if self.document_page_exists(docid):
                existing_pages.append(f"{self.base_url}{docid}")
                consecutive_missing = 0
            else:
                consecutive_missing += 1
            docid += 1
        
        # Extract documents and metadata
        all_links = []
        all_metadata = []
        
        for page_url in existing_pages:
            links, metadata = self.extract_document_links_and_metadata(page_url)
            all_links.extend(links)
            all_metadata.extend(metadata)
        
        # Download and process documents
        documents = []
        for link in all_links:
            try:
                response = requests.get(link, auth=HTTPBasicAuth(self.username, self.password))
                if response.ok:
                    doc_id = link.split('docid=')[-1]
                    content_type = response.headers.get('content-type', '')
                    
                    documents.append({
                        'doc_id': doc_id,
                        'content': response.content,
                        'content_type': content_type
                    })
            except Exception as e:
                logger.error(f"Error downloading document {link}: {e}")
        
        # Process documents
        dataset = []
        for doc, metadata in zip(documents, all_metadata):
            try:
                raw_text = ""
                if 'application/pdf' in doc['content_type']:
                    raw_text = self.extract_text_from_pdf(doc['content'])
                elif 'application/vnd.openxmlformats-officedocument.presentationml.presentation' in doc['content_type']:
                    raw_text = self.extract_text_from_pptx(doc['content'])
                
                if raw_text:
                    cleaned_text = self.clean_text(raw_text)
                    
                    dataset.append({
                        'document_id': doc['doc_id'],
                        'raw_text': raw_text,
                        'cleaned_text': cleaned_text,
                        'link': metadata['url'],
                        'title': metadata['title'],
                        'author': metadata['author'],
                        'date': metadata['updated'],
                        'source': metadata['source']
                    })
                    
                    logger.info(f"Processed DocDB document: {doc['doc_id']} - {metadata['title']}")
            
            except Exception as e:
                logger.error(f"Error processing document {doc['doc_id']}: {e}")
        
        logger.info(f"Extracted {len(dataset)} DocDB documents")
        return dataset 