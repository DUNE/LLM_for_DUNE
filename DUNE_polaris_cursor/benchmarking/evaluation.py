import mlflow
import asyncio
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines
import json
from config import ( ARGO_API_USERNAME, ARGO_API_KEY, 
    DEFAULT_TOP_K, STORE, QA_PATH
)
import time
import logging
import pandas as pd
from config import FERMILAB_SESSION_SECRET
from src.indexing.chroma_manager import ChromaManager
from src.api.fermilab_client import FermilabAPIClient
from src.api.argo_client import ArgoAPIClient
from src.auth.fermilab_auth import fermilab_auth
from src.utils.logger import get_logger

logger = get_logger(__name__)
import json
import re
import json
import os
from find_port import find_port
from src.extractors.indico_extractor_multithreaded import IndicoExtractor
from src.extractors.docdb_extractor_multithreaded import DocDBExtractor


fermi_client=FermilabAPIClient(ARGO_API_USERNAME, ARGO_API_KEY)
MODEL='gpt-oss:20b'
rel=0

def normalize(s):
    # Remove all whitespace characters like \n, \t, spaces, etc.

    expected = re.sub(r'\s+', '', s)
    if expected[-1] == '.': 
        return expected[:-1]
    return expected

indico_session=IndicoExtractor()
docdb_session=DocDBExtractor()
docdb_session._build_session()

def relevant_refs(question, expected, contexts, references):

    expect = normalize(expected)
    if expect in references:
        print("found")
        return 1
    else:
        prompt = f'Read this context and determine if it provides an answer to the question: {question}, if so, return a float value closer to 1 in the format of a json with the only permitted key being "score". If the content in the file is not related to the question at all or loosely relates to the question, give a score close to 0'
        results = 0
        #look at the questoin, open the references and check if they relate to the quesiton
        for link, snippet in zip(references,contexts):
            if not link: continue
            document_text=snippet
            if not document_text:
                continue
            
            resp=fermi_client.chat_completion(question=prompt, context=document_text,  model=MODEL) #base_url='https://vllm.fnal.gov/v1/chat/completions')
            match = re.search(r'(\d[\d\s]*\.?[\d\s]*)', resp)

            if match:
                # Remove spaces to clean up
                number_str = match.group(1).replace(" ", "")

            results += float(number_str)
    length = 0
    for i in references:
        if i:
            length += 1
    return results / length

def correctness(expectation, output) -> float:
    results = 0

    try:
        if '[ERROR]' in output:
            return 0.0
        question="Using the expected output as the ground truth answer, determine if the generated output is correct .Return a float value between 0 and 1 in the format of a json with the only permitted key being 'score'. Your float value must not contain any letters, they must strictly be comprised of numerical values and decimals. You will evaluate correctness like this: Get the main points from the expected output and the generated output. Then evaluate how closely aligned these points are. The words do not need to match exactly. Even if the generated output is phrased differently, as long as the general idea behind the generated output is the same as that behind the expected out, give the generated output a score close to 1. However, if the generated output does not address the same points or convey the same ideas as that of the expected output, then give a value close to 0."
                            
        context = f"Generated output: {outputs}\nExpected out: {expectations}"
        resp = fermi_client.chat_completion(question=question, context=context, model =MODEL) #'nomic-embed-text:latest')# base_url='https://vllm.fnal.gov/v1/chat/completions')
        match = re.search(r'(\d[\d\s]*\.?[\d\s]*)', resp)
        if match:
            # Remove spaces to clean up
            number_str = match.group(1).replace(" ", "")
        
        results += float(number_str)
    except Exception as e:
        print("Excpetion ", e)
        return e
    print("results ", results)
    return results


import re

def extract_https_url(text):
    match = re.search(r'(?:\(|\[|\{)?(https?://[^\s\)\]\}]+)(?:\)|\]|\})?', text)
    if match:
        return match.group(1)  # Group 1 contains just the URL
    return None

class Evalutation():
    def __init__(self, port, experiment_name, data_path,model, top_k, keyword):

        #self.faiss_manager = FAISSManager(data_path) 
        self.faiss_manager = ChromaManager(data_path)#FAISSManager(data_path)
        print(f"connecting to client")
        self.model=model
        #self.fermi_client = ArgoAPIClient(ARGO_API_USERNAME, ARGO_API_KEY)
        self.fermi_client=FermilabAPIClient(ARGO_API_USERNAME, ARGO_API_KEY)
        print("Conntected to client")
        self.top_K=top_k
        self.keyword=keyword
        
    def create_validation_dataset(self):
        
        qa=pd.read_csv(QA_PATH)
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
    
        refs_dataset=pd.read_csv(QA_PATH)
        refs=[]
        for row in refs_dataset.iterrows():
            dictionary={}
            dictionary['inputs'] = {'question': row[1]['question']}
            try:
                link = re.sub(r'\s+', '', row[1]['link'])
                link=extract_https_url(link)
                assert link, row[1]['link']
            
                dictionary['expectations'] = {'expected_response': link}
                refs.append(dictionary)
            except:
                continue
        self.ref_dataset = refs
    def create_latency_dataset(self):


        latency_dataset=pd.read_csv(QA_PATH)[:2]
        latency=[]
        for row in latency_dataset.iterrows():
            dictionary={}
            dictionary['inputs'] = {'question': row[1]['question']}
            dictionary['expectations'] = {'expected_response': 0}
            latency.append(dictionary)
        print("made latenvy ds")
        self.latency_dataset = latency
    
    def evaluate(self,method):
        self.create_validation_dataset()
        print("in evaluate fn")
        print(f"Starting eval: {method}")
        with mlflow.start_run():
            if method == 'correctness':
                self.create_validation_dataset()
                '''results = mlflow.genai.evaluate(
                    data=self.eval_dataset,
                    predict_fn=self.llm_qa_response,
                    scorers=[correctness],
                )'''
                score = 0
                df=[]
                for dictionary in self.eval_dataset:
                    question = dictionary['inputs']['question']
                    expected = dictionary['expectations']['expected_response']
                    contexts, references = self.faiss_manager.search(question, top_k = self.top_K)
                    answer = self.fermi_client.chat_completion(question, " ".join(contexts))
                    score_temp = correctness(expected, answer)
                    df.append({'question': question, 'expected_response': expected, 'score': score_temp, 'true response': answer, 'contexts': contexts})
                    score += score_temp
                return df, [{'score' : score/len(self.eval_dataset)}]


            elif method == 'relevant_refs':
                self.create_refs_dataset()
                '''results = mlflow.genai.evaluate(
                    data=self.ref_dataset,
                    predict_fn=self.llm_references,
                    scorers=[relevant_refs],
                )'''
                score = 0
                df=[]
                for dictionary in self.ref_dataset:
                    question = dictionary['inputs']['question']
                    reference = dictionary['expectations']['expected_response']
                    
                    contexts, references = self.faiss_manager.search(question)
                    score_temp = relevant_refs(question, reference, contexts, references)
                    df.append({'question': question, 'expected_reference': reference, 'score': score_temp, 'true references': references, 'contexts': contexts})
                    score += score_temp
                print(score/len(self.ref_dataset), score)
                return df, [{'score' : score/len(self.ref_dataset)}]

            elif method=='latency':
                duration=0.0
                qa=pd.read_csv(QA_PATH)
                df = []
                for q in qa.iterrows():
                    question = q[1]['question']
                    start=time.time()
                    context,links=self.faiss_manager.search(question, top_k=self.top_K, keyword=self.keyword)
                    resp=self.argo_client.chat_completion(question, ' '.join(context))
                    end=time.time()
                    df.append({'question': q, 'latency' : end-start})
                    duration += (end-start)
                print(f"duration total: {duration}, total qs: {len(qa)}")
                return df, [{'latency/mean': duration/100}]

import argparse

parser = argparse.ArgumentParser(description="Example script to read command-line arguments")
parser.add_argument('--port', type=int, help='Port to use', default=5000)
parser.add_argument('--experiment_name', type=str, default=None)
parser.add_argument('--method', type=str, default='correctness')
parser.add_argument('--data_path', type=str)
parser.add_argument('--savedir', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--top_k',type=int)
parser.add_argument('--keyword', action='store_true')
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


val = Evalutation(args.port, args.experiment_name, args.data_path, args.model, args.top_k, args.keyword)
results, score = val.evaluate(args.method)

if isinstance(results, list):
    pd.DataFrame(results).to_csv(os.path.join(args.savedir, f'{args.method}_tracking.csv'))
    pd.DataFrame(score).to_csv(os.path.join(args.savedir, f'{args.method}_score.csv'))
    exit(0)

