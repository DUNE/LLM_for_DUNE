from src.api.argo_client import ArgoAPIClient
from config import (
    HOST, PORT, DEBUG, ARGO_API_USERNAME, ARGO_API_KEY, 
    DEFAULT_TOP_K, ENABLE_AUTHENTICATION, FERMILAB_REDIRECT_URI, 
    validate_config, create_directories
)
from src.indexing.chroma_manager import ChromaManager
from collections import defaultdict
argo = ArgoAPIClient(ARGO_API_USERNAME, ARGO_API_KEY)

import os

chroma_manager = ChromaManager(os.getenv('DB_PATH'))
results=defaultdict(list)
questions = [
    'How many CPA planes will be installed in the DUNE Horizontal Drift Far Detector in South Dakota',
    'Can you find a subset of talks from Laura Fields Neutrino Beam experiments?',
    'Which selection cuts were used for event selection in the ProtoDUNE-ND 2×2 charged-particle multiplicity analysis?',
    'Could you describe the detailed steps for performing a cross-section analysis in DUNE?',
    'Where can I find the documentation for how to add a member to DUNE?',
    'What is the purpose of PRISM in the Near Detector?',
    'Can you show the current timeline for the DUNE Far Detector installation?',
    'Can you summarize the updates to DUNE’s science goals presented at the recent Fermilab Colloquium by Chris Marshall?',
    'When did the 2×2 demonstrator start collecting data, and how long did data taking last?',
    'Can you provide a list of current DUNE Speakers Committee members?',
        ]
for question in questions:
    context_snippet, ref = chroma_manager.search(question,top_k=3)
    context = "\n\n".join(context_snippet)

    answer = argo.chat_completion(question, context)
    results['Question'].append(question)
    results['Answer'].append(answer)
    results['Links'].append(' '.join(ref))
import pandas as pd
pd.DataFrame(results).to_csv("TestQuestions_nodistinciton_search.csv")
