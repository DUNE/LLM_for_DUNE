import markdown
import requests

from typing import Optional
from config import FERMILAB_API_URL,  LLM_TEMPERATURE, LLM_TOP_P, LLM_MODEL
from src.utils.logger import get_logger
import json
logger = get_logger(__name__)

class FermilabAPIClient:
    """Client for Argo API interactions"""
    
    def __init__(self, username: str, api_key: str):
        self.username = username
        self.api_key = api_key
        self.base_url = FERMILAB_API_URL
        
        if not self.username or not self.api_key:
            raise ValueError("Argo API credentials not provided")
    
    def chat_completion(
        self, 
        question: str, 
        context: str,
        temperature: float = LLM_TEMPERATURE,
        top_p: float = LLM_TOP_P,
        model: str = LLM_MODEL,
        timeout: int = 60,
        base_url: str = FERMILAB_API_URL
    ) -> str:
        """Send a chat completion request to Argo API"""
        print(context.split()[:5])
        print(f"Question is {question} and context length is {len(context.split())}")
        headers = {
            #"Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"Using model {model}")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant specialized in scientific documentation for the DUNE experiment. Provide succinct answers. If you don't see an answer in the context, preceed your response with 'This answer does not reference Indico nor Dune DocDB."
                },
                {
                    "role": "user", 
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}"
                }
            ]
        }
        if 'ollama' in base_url:
            payload['stream'] = True
        response = []
        try:
            logger.info(f"Sending request to {base_url}")
            with requests.post(base_url, json=payload, stream=True, timeout=300) as resp:
                for line in resp.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode("utf-8"))
                            token = ""
                            if "message" in data and "content" in data["message"]:
                                token = data["message"]["content"]
                            elif "response" in data:
                                token = data["response"]
                            if token:
                                yield token #markdown.markdown(token)
                                response.append(token)
                        except Exception as e:
                            logger.error(e)
                            continue
            return ' '.join(response)
            
        except requests.Timeout:
            logger.error(f"Fermilab API request timed out {question}, {len(context.split())}")
            return f"[ERROR] Request to Fermilab API timed out after {timeout} seconds. {context}"
        
        except requests.RequestException as e:
            logger.error(f"Fermilab API request failed: {e}")
            return f"[ERROR] Failed to get response from Fermilab API: {e}"
        
        except Exception as e:
            logger.error(f"Unexpected error with Fermilab API: {e}")
            return f"[ERROR] Unexpected error: {e}"
    
    def health_check(self) -> bool:
        """Check if the Fermilab API is accessible"""
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
