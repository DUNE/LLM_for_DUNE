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
from src.auth.fermilab_auth import fermilab_auth
from src.utils.logger import get_logger

logger = get_logger(__name__)
import json
import re
import json
import os
from find_port import find_port
argo_client= FermilabAPIClient(ARGO_API_USERNAME, ARGO_API_KEY)
from src.extractors.indico_extractor_multithreaded import IndicoExtractor
from src.extractors.docdb_extractor_multithreaded import DocDBExtractor
MODEL='gpt-oss:20b'
def normalize(s):
    # Remove all whitespace characters like \n, \t, spaces, etc.
    expected = re.sub(r'\s+', '', s)
    if expected[-1] == '.': 
        return expected[:-1]
    return expected

indico_session=IndicoExtractor()
docdb_session=DocDBExtractor()
docdb_session._build_session()
@scorer
def relevant_refs(outputs:dict, expectations:dict):
    #logger.error(outputs)
    data= json.loads(outputs)
    question = data['question']
    references = data['references']
    contexts = data['context_snippets']
    
    logger.info(f"num snipbfore  {len(contexts)}")
    expect = normalize(expectations['expected_response'])
    assert len(references)>0
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
            
            resp=argo_client.chat_completion(question=prompt, context=document_text, model=MODEL) # base_url='https://vllm.fnal.gov/v1/chat/completions')
            match = re.search(r'(\d[\d\s]*\.?[\d\s]*)', resp)

            if match:
                # Remove spaces to clean up
                number_str = match.group(1).replace(" ", "")

            results += float(number_str)
    length = 0
    for i in references:
        if i:
            length += 1
    print(results)
    return results / length

@scorer
def latency(latency):
    print("latency")#, kwargs)
    try:
        outputs = outputs.get("latency", float("inf"))
        logger.info("in run")
        expectations = expectations.get("time", float("inf"))
    except Exception as e:
        print(e)
    print(outputs)
    return outputs
@scorer
def correctness(outputs: dict, expectations: dict= None) -> int:
    outputs = outputs.get("generated_response", "")
    expectations = expectations.get("expected_response", "")
    try:
        if '[ERROR]' in outputs:
            return 0.0
        question="Using the expected output as the ground truth answer, determine if the generated output is correct .Return a float value between 0 and 1 in the format of a json with the only permitted key being 'score'. Your float value must not contain any letters, they must strictly be comprised of numerical values and decimals. You will evaluate correctness like this: Get the main points from the expected output and the generated output. Then evaluate how closely aligned these points are. The words do not need to match exactly. Even if the generated output is phrased differently, as long as the general idea behind the generated output is the same as that behind the expected out, give the generated output a score close to 1. However, if the generated output does not address the same points or convey the same ideas as that of the expected output, then give a value close to 0."
        context = f"Generated output: {outputs}\nExpected out: {expectations}"
        resp = argo_client.chat_completion(question=question, context=context, model = MODEL) #'nomic-embed-text:latest')# base_url='https://vllm.fnal.gov/v1/chat/completions')
        match = re.search(r'(\d[\d\s]*\.?[\d\s]*)', resp)

        if match:
            # Remove spaces to clean up
            number_str = match.group(1).replace(" ", "")
        
        results = float(number_str)
    except Exception as e:
        return e
    print(results)
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
        self.argo_client = FermilabAPIClient(ARGO_API_USERNAME, ARGO_API_KEY)
        print("Conntected to client")
        self.top_K=top_k
        self.keyword=keyword
        #mlflow.tracing.disable()
        mlflow.set_tracking_uri("./my_mlruns")
        #mlflow.set_tracking_uri(f"http://127.0.0.1:{port}")
        mlflow.set_experiment("experiment_name")
        
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


        latency_dataset=pd.read_csv(QA_PATH)
        latency=[]
        for row in latency_dataset.iterrows():
            dictionary={}
            dictionary['inputs'] = {'question': row[1]['question']}
            dictionary['expectations'] = {'expected_response': 0}
            latency.append(dictionary)
        print("made latenvy ds")
        self.latency_dataset = latency
    def llm_qa_response(self, question):
        context_snippets, references = self.faiss_manager.search(question, top_k=self.top_K, keyword=self.keyword)
        context = "\n\n".join(context_snippets)
        # Get answer from Argo API
        answer = self.argo_client.chat_completion(question, context,model=self.model)
        print(answer)
        assert answer
        return {"generated_response": answer}
    def llm_references(self,question):
        context_snippets, references = self.faiss_manager.search(question, top_k=self.top_K, keyword=self.keyword)
        #logger.info(f"REFS {references}")
        data = {
            "question": question,
            "references": references,
            "context_snippets": context_snippets
        }

        return json.dumps(data) #f"question \ {question} \ {','.join(references)} \{json.dumps(context_snippets)}"
    def latency_collector(self, question):
        start = time.time()
        context_snippets, references = self.faiss_manager.search(question, top_k=top_K, keyword=self.keyword)
        context = "\n\n".join(context_snippets)
        # Get answer from Argo API
        answer = self.argo_client.chat_completion(question, context)
        end = time.time()
        print("time = ", end-start)
        return json.dumps({'latency' :end-start})


    #@mlflow.trace
    def evaluate(self,method):
        self.create_validation_dataset()
        print("in evaluate fn")
        print(f"Starting eval: {method}")
        with mlflow.start_run():
            if method == 'correctness':
                self.create_validation_dataset()
                results = mlflow.genai.evaluate(
                    data=self.eval_dataset,
                    predict_fn=self.llm_qa_response,
                    scorers=[correctness],
                )
            elif method == 'relevent_refs':
                self.create_refs_dataset()
                results = mlflow.genai.evaluate(
                    data=self.ref_dataset,
                    predict_fn=self.llm_references,
                    scorers=[relevant_refs],
                )
            elif method=='latency':
               self.create_latency_dataset()
               self.create_refs_dataset()
               results = mlflow.genai.evaluate(
                    data=self.latency_dataset,
                    predict_fn=self.latency_collector,
                    scorers=[latency],
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
results = val.evaluate(args.method)

print(results.metrics.keys())
for key in results.metrics:
    if '/' in key:
        metric=key.split('/')[0]
        DATA_FILE = f"{args.savedir}/{key.split('/')[0]}.json"
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


save_results_file = f"{args.savedir}/{metric}_evaluation_results.csv"
logger.info(f"Saved results to {save_results_file}")
results.result_df.to_csv(save_results_file, index=False)

