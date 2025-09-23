from collections import defaultdict
import os
import pickle
import faiss
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import requests
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path


USERNAME = os.getenv('ARGO_API_USERNAME', 'aleena')
API_KEY = os.getenv('ARGO_API_KEY', 'XXX')
QUESTION = f'''You are given a dictionary with link:text pairs. Your job is to use the each text to generate a question from that respective text. Then give the answer to that question from the text. The question should be answerable by reading the text.  The link should be the link associated with the text you used to generate that question Return your answer very strictly in this format: **Question:** <your question> **Answer:*** <the answer> **Link:** <link>.  DO NOT put anything else within the ** **. All 3 of these components MUST be present and must be valid. Generate 10 questions '''


def _load_metadata(data_path) -> Dict[str, Any]:
    data_path = os.path.join(data_path, 'metadata_store.pkl')
    if Path(data_path).exists():
        with open(data_path, "rb") as f:
            return pickle.load(f)
    return {}

def _load_doc_ids(data_path) -> List[str]:
    data_path = os.path.join(data_path, 'doc_ids.pkl')
    if Path(data_path).exists():
        with open(data_path, "rb") as f:
            return pickle.load(f)
    return []




from config import ARGO_API_URL, LLM_TEMPERATURE, LLM_TOP_P, LLM_MODEL
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

            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()

            result = response.json()

            return result.get("response", "No answer returned.")

        except requests.Timeout:
            return f"[ERROR] Request to Argo API timed out after {timeout} seconds."

        except requests.RequestException as e:
            return e
        except Exception as e:
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
            return False
    
from collections import defaultdict
def get_qa(resp):
    '''
        Resp is the output of the LLM
        This function parses the LLM's output which must be of the form
            **Question:** <The question>
            **Answer:** <The answer>
            **Link:** <The link>
        into a dictionary that can later be saved as a CSV
    '''
    start = 0
    end = 0
    ans=defaultdict(list)
    question_String='**Question:**'
    answer_String = '**Answer:**'
    link_String='**Link:**'
    while end < len(resp):
        start = start + resp[start:].index(question_String) + len(question_String)
        end = start + resp[start:].index(answer_String)
        question = resp[start:end]
        ans['question'].append(question)
        

        answer_start =  end + len(answer_String)
        try:
          answer_end = answer_start + resp[answer_start:].index(link_String)
        except:
          answer_end=len(resp)
        answer = resp[answer_start:answer_end]
        ans['answer'].append(answer)
        
        link_start = answer_end
        try:
            link_end = link_start + resp[link_start:].index(question_String)
        except:
            link_end=len(resp)
I
        ans['link'].append(resp[link_start+len(link_String): link_end])
        start = link_end
        end=link_end
    return ans

if __name__=='__main__':
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(description="Example argparse script")

    # Add arguments
    parser.add_argument('--data_path',type=str, required=True, help='Path to Vector DB')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to store question/answer pairs')
    parser.add_argument('--datastore', type=str, required=True, help='Type of vector DB i.e faiss or chroma')
    # Parse arguments
    args = parser.parse_args()


    from collections import defaultdict
    if args.datastore == 'chroma':
        chroma = ChromaManager(path)
        texts=collections.defaultdict()
        metadata_store = chroma.chroma_collection.get(include=['metadatas', 'documents'])
    elif args.datastore == 'faiss':
        doc_ids = _load_doc_ids(args.data_path)
        metadata_store=_load_metadata(args.data_path)
    
    texts=defaultdict()


    for did in doc_ids:
        did = metadata_store[did]
        try:
            link = did['download_url']
        except:
            link  = did.get('url')
            if not link:
                link = did.get('event_url')
        if link: 
            try:
                texts[link] = did['cleaned_text']
            except:
                continue


    qa_pairs = defaultdict(list)
    start = 0

    #number of times to iterate
    num_loops=10

    #number of sources and associated text to pass into the LLM at once
    source_text_pairs_per_loop = int(len(texts)/num_loops)
    argo_client = ArgoAPIClient(USERNAME, API_KEY)
    assert argo_client.health_check() 

    links = list(texts.keys())
    for i in range(num_loops):
        end = (i+1)* source_text_pairs_per_loop
        if end >= len(links):break

        context = '{'
        for link in links[start:end]:
            context += link + ":" + texts[link]
        context += '}'

        resp = argo_client.chat_completion(timeout=60, question=QUESTION, context=context)
        
        if isinstance(resp, str):
            qa =get_qa(resp)
            qa_pairs['question'].extend(qa['question'])
            qa_pairs['answer'].extend(qa['answer'])
            qa_pairs['link'].extend(qa['link'])
        
        start = end
    
    import pandas as pd
    pd.DataFrame(qa_pairs).to_csv(os.path.join(save_dir, "qa.csv"))
