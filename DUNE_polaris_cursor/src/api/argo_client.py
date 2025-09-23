import requests
from typing import Optional
from config import ARGO_API_URL, LLM_TEMPERATURE, LLM_TOP_P, LLM_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ArgoAPIClient:
    """Client for Argo API interactions"""
    
    def __init__(self, username: str, api_key: str):
        self.username = username
        self.api_key = api_key
        self.base_url = ARGO_API_URL
        
        if not self.username or not self.api_key:
            raise ValueError("Argo API credentials not provided")
    
    def chat_completion(
        self, 
        question: str, 
        context: str,
        temperature: float = LLM_TEMPERATURE,
        top_p: float = LLM_TOP_P,
        model: str = LLM_MODEL,
        timeout: int = 30
    ) -> str:
        """Send a chat completion request to Argo API"""
        #print(context)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "user": self.username,
            "model": model,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant specialized in scientific documentation for the DUNE experiment."
                },
                {
                    "role": "user", 
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}"
                }
            ],
            "stop": [],
            "temperature": temperature,
            "top_p": top_p,
        }
        
        
        try:
            logger.info("Sending request to Argo API")
            response = requests.post(
                self.base_url, 
                headers=headers, 
                json=payload, 
                timeout=timeout
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info("Received response from Argo API")
            return result.get("response", "No answer returned.")
            
        except requests.Timeout:
            logger.error("Argo API request timed out")
            return f"[ERROR] Request to Argo API timed out after {timeout} seconds."
        
        except requests.RequestException as e:
            logger.error(f"Argo API request failed: {e}")
            return f"[ERROR] Failed to get response from Argo API: {e}"
        
        except Exception as e:
            logger.error(f"Unexpected error with Argo API: {e}")
            return f"[ERROR] Unexpected error: {e}"
    
    def health_check(self) -> bool:
        """Check if the Argo API is accessible"""
        try:
            # Simple test request
            test_response = self.chat_completion(
                question="Hello",
                context="This is a test",
                timeout=10
            )
            return not test_response.startswith("[ERROR]")
        except Exception as e:
            logger.error(f"Argo API health check failed: {e}")
            return False 
