import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import re
from urllib.parse import urlparse
from .base import BaseExtractor
from config import DUNE_INDICO_ACCESS_KEY, INDICO_BASE_URL, INDICO_CATEGORY_ID
from src.utils.logger import get_logger

logger = get_logger(__name__)

class IndicoExtractor(BaseExtractor):
    """Extractor for DUNE Indico documents"""
    
    def __init__(self):
        super().__init__()
        self.access_key = DUNE_INDICO_ACCESS_KEY
        self.base_url = INDICO_BASE_URL
        self.category_id = INDICO_CATEGORY_ID
    
    def extract_documents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Extract documents from Indico"""
        logger.info(f"Extracting documents from Indico category {self.category_id} (limit: {limit})")
        
        # Fetch category data
        category_url = f"{self.base_url}/export/categ/{self.category_id}.json"
        if self.access_key:
            category_url += f"?access_token={self.access_key}"
        
        try:
            response = requests.get(category_url)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Could not fetch category data: {e}")
            return []
        
        events = data.get("results", [])[:limit]
        dataset = []
        
        for event in events:
            try:
                event_id = event.get("id")
                title = event.get("title", "Unknown Title")
                author = event.get("author", "Unknown Author")
                start_date = event.get("startDate", "Unknown Date")
                event_url = f"{self.base_url}/event/{event_id}/"
                
                if self.access_key:
                    event_url += f"?access_token={self.access_key}"
                
                logger.info(f"Processing event: {title} ({event_id})")
                
                # Get event HTML
                html_response = requests.get(event_url)
                html_response.raise_for_status()
                soup = BeautifulSoup(html_response.text, 'html.parser')
                
                # Process contributions
                for contrib in soup.select('li.timetable-item.timetable-contrib'):
                    contrib_title = self._extract_contribution_title(contrib)
                    contrib_author = self._extract_contribution_speaker(contrib)
                    
                    # Process attachments
                    for attachment_url in self._extract_attachment_urls(contrib):
                        doc_data = self._process_attachment(
                            attachment_url, contrib_title, contrib_author, start_date
                        )
                        if doc_data:
                            dataset.append(doc_data)
            
            except Exception as e:
                logger.error(f"Error processing event {event_id}: {e}")
                continue
        
        logger.info(f"Extracted {len(dataset)} Indico documents")
        return dataset
    
    def _extract_contribution_title(self, contrib) -> str:
        """Extract contribution title from HTML element"""
        title_elem = contrib.select_one('.timetable-title')
        return title_elem.get_text(strip=True) if title_elem else "Unknown Title"
    
    def _extract_contribution_speaker(self, contrib) -> str:
        """Extract speaker name from HTML element"""
        speaker_elem = contrib.select_one('.speaker-list')
        if not speaker_elem:
            return "Unknown Author"
        
        speaker_text = speaker_elem.get_text(separator=' ', strip=True)
        # Remove "Speaker:" or "Speakers:" prefix
        speaker_text = re.sub(r'(?i)^speakers?\s*:\s*', '', speaker_text)
        # Remove affiliations in parentheses
        return re.sub(r'\s*\([^)]*\)', '', speaker_text).strip()
    
    def _extract_attachment_urls(self, contrib) -> List[str]:
        """Extract attachment URLs from contribution"""
        urls = []
        for link in contrib.select('a.attachment[href$=".pdf"], a.attachment[href$=".pptx"]'):
            href = link.get('href')
            if href:
                urls.append(self.base_url + href)
        return urls
    
    def _process_attachment(self, url: str, title: str, author: str, date: str) -> Dict[str, Any]:
        """Process a single attachment"""
        try:
            # Download file
            response = requests.get(url)
            response.raise_for_status()
            content = response.content
            
            # Determine file type
            filename = url.split('/')[-1]
            file_type = filename.split('.')[-1].lower()
            
            # Extract text
            if file_type == 'pdf':
                raw_text = self.extract_text_from_pdf(content)
            elif file_type == 'pptx':
                raw_text = self.extract_text_from_pptx(content)
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                return None
            
            if not raw_text.strip():
                logger.info(f"Skipping empty file: {filename}")
                return None
            
            # Generate document ID
            parsed_path = urlparse(url).path
            doc_id = "/".join(parsed_path.split("/")[-3:])
            
            # Clean text
            cleaned_text = self.clean_text(raw_text)
            
            logger.info(f"Processed Indico document: {doc_id} - {title}")
            
            return {
                'document_id': doc_id,
                'raw_text': raw_text,
                'cleaned_text': cleaned_text,
                'link': url,
                'title': title,
                'author': author,
                'date': date.get('date', 'unknown') if isinstance(date, dict) else date,
                'source': 'indico'
            }
        
        except Exception as e:
            logger.error(f"Error processing attachment {url}: {e}")
            return None 