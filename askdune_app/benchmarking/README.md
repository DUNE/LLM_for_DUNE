**evaluation.py**
- Evaluates the LLM Response Accuracy (Correctness), Retrieval Accuracy, and Latency

**test_models.sh**
- Calls evaluation.py to run each metric to evaluate each possible LLM

**test_k_docs.sh**
- Calls evaluation.py to run each metric to evaluate each possible k_docs configuration
  - This tests different k_docs value but can be modified/run multiple times to test different search configurations

**benchmark_plot.py**
- Generates plots based on experiments. Each function must be modified slightly to account for different means of storing scores in terms of paths, xeperiment names, file formats. The reccomended formats are specified 
