from src.api.fermi_client import FermilabAPIClient
from src.indexing.chroma_manager import ChromaManager
from collections import defaultdict
import pandas as pd
import os

fermi = FermilabAPIClient()
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
    context_snippet, ref = chroma_manager.search(question)
    context = "\n\n".join(context_snippet)

    answer = fermi.chat_completion(question, context)
    results['Question'].append(question)
    results['Answer'].append(answer)
    results['Links'].append(' '.join(ref))

pd.DataFrame(results).to_csv("Results/TestQuestions_results.csv")
