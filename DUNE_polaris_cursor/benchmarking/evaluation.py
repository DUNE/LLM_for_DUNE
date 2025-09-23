import mlflow
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines

from config import ( ARGO_API_USERNAME, ARGO_API_KEY, 
    DEFAULT_TOP_K, STORE,
)
import logging
import pandas as pd
from config import FERMILAB_SESSION_SECRET
from src.indexing.faiss_manager_reindexed import FAISSManager
from src.indexing.chroma_manager import ChromaManager
from src.api.argo_client import ArgoAPIClient
from src.auth.fermilab_auth import fermilab_auth
from src.utils.logger import get_logger

logger = get_logger(__name__)
import json
import re
import json
import os
from find_port import find_port
argo_client= ArgoAPIClient(ARGO_API_USERNAME, ARGO_API_KEY)


def normalize(s):
    # Remove all whitespace characters like \n, \t, spaces, etc.
    return re.sub(r'\s+', '', s)


@scorer
def relevant_refs(outputs:str, expectations:str):
    all_refs=outputs.split(',')
    print(f" expectation {expectations}")

    expect = normalize(expectations['expected_response'])
    if expect in all_refs:
        return 1
    print(f"extracted: {all_refs}\ndesired: {expect}")
    return 0


@scorer
def correctness(outputs: str, expectations: str = None) -> int:
    try:
        
        question = "How correct is the generated output from the expected output? Return a single value, 0 or 1 in the format of a json with the only permitted key being 'score'. Get the main points from the expected output and evaluate correctness on how closely aligned the generated out is to those points. If the generated output does not address those points, give a value of 0. If it does address these points, give it a value of 1." 
        context = f"Generated output: {outputs}\nExpected out: {expectations}"
        resp = argo_client.chat_completion(question=question, context=context)
        clean_str = re.sub(r'^```json\s*|```$', '', resp, flags=re.MULTILINE).strip()
        # Parse JSON
        data = json.loads(clean_str)
        # Return the "score" value as float
        
        results = float(data.get("score", 0))
        print("generated" , outputs)
        print("true", expectations)
        print(results)
    except Exception as e:
        print(e)
        return e
    return results


import re

def extract_https_url(text):
    match = re.search(r'(?:\(|\[|\{)?(https?://[^\s\)\]\}]+)(?:\)|\]|\})?', text)
    if match:
        return match.group(1)  # Group 1 contains just the URL
    return None

class Evalutation():
    def __init__(self, port, experiment_name, data_path):
        if STORE == 'faiss':
            self.faiss_manager = FAISSManager(data_path) 
            #self.faiss_manager = ChromaManager(data_path)#FAISSManager(data_path)
        print(f"connecting to client")
        self.argo_client = ArgoAPIClient(ARGO_API_USERNAME, ARGO_API_KEY)
        print("Conntected to client")
        #mlflow.tracing.disable()
        mlflow.set_tracking_uri("./my_mlruns")
        #mlflow.set_tracking_uri(f"http://127.0.0.1:{port}")
        mlflow.set_experiment("experiment_name")
        
    def create_validation_dataset(self):
        
        qa=pd.read_csv("/home/newg2/Projects/LLM/DUNE/LLM_for_DUNE/qa.csv")
        qas=[]
        for row in qa.iterrows():
            dictionary={}
            try: link = re.sub(r'\s+', '', row[1]['link'])
            except: continue
            dictionary['inputs'] = {'question': row[1]['question']}

            dictionary['expectations'] = {'expected_response': row[1]['answer']}
            qas.append(dictionary)
        self.eval_dataset = qas
    
    def create_refs_dataset(self):
    

        refs_dataset=pd.read_csv("/home/newg2/Projects/LLM/DUNE/LLM_for_DUNE/qa.csv")
        refs=[]
        for row in refs_dataset.iterrows():
            dictionary={}
            dictionary['inputs'] = {'question': row[1]['question']}
            print(row[1]['question'], row[1]['link'])
            try:
                link = re.sub(r'\s+', '', row[1]['link'])
                link=extract_https_url(link)
                assert link, row[1]['link']
            
                dictionary['expectations'] = {'expected_response': link}
                refs.append(dictionary)
            except:
                continue
        self.ref_dataset = refs

    def llm_qa_response(self, question):
        context_snippets, references = self.faiss_manager.search(question, top_k=DEFAULT_TOP_K)
        context = "\n\n".join(context_snippets)
        # Get answer from Argo API
        answer = self.argo_client.chat_completion(question, context)
        return answer
    
    def llm_references(self,question):
        context_snippets, references = self.faiss_manager.search(question, top_k=DEFAULT_TOP_K)
        return ','.join(references)

    #@mlflow.trace
    def evaluate(self,method):
        self.create_validation_dataset()
        print("in evaluate fn")
        print(f"Starting eval")
        with mlflow.start_run():
            if method == 'correctness':
                self.create_validation_dataset()
                results = mlflow.genai.evaluate(
                    data=self.eval_dataset,
                    predict_fn=self.llm_qa_response,
                    scorers=[correctness],
                )
            else:
                self.create_refs_dataset()
                results = mlflow.genai.evaluate(
                    data=self.ref_dataset,
                    predict_fn=self.llm_references,
                    scorers=[relevant_refs],
                )
        print(results)
        return results  

import argparse

parser = argparse.ArgumentParser(description="Example script to read command-line arguments")
parser.add_argument('--port', type=int, help='Port to use', default=5000)
parser.add_argument('--experiment_name', type=str, default=None)
parser.add_argument('--method', type=str, default='correctness')
parser.add_argument('--data_path', type=str)
parser.add_argument('--savedir', type=str)
args = parser.parse_args()


def init_logger(name=__name__, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

logger = init_logger('benchmarking_logger')


val = Evalutation(args.port, args.experiment_name, args.data_path)
results = val.evaluate(args.method)


for key in results.metrics:
    if '/mean' in key:
        metric=key.split('/')[0]
        DATA_FILE = f"{key.split('/')[0]}.json"
logger.info(f"Logging metrics to {DATA_FILE}")


if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
else:
    data = {}

run_name = args.data_path.split("/")[-1]
for key in results.metrics:
    if  '/mean' in key:
        #check if previous run's metric already stored and if so, add onto it so we can average later, else save it as a new entry 
        if run_name in data:
            data[run_name] += results.metrics[key]
        else:
            data[run_name] = results.metrics[key]

with open(DATA_FILE, "w") as f:
    json.dump(data, f)


save_results_file = f"{args.savedir}/{args.data_path.split('/')[-1]}_{metric}_evaluation_results.csv"
logger.info(f"Saved results to {save_results_file}")
results.result_df.to_csv(save_results_file, index=False)

