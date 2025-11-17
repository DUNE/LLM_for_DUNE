from collections import defaultdict
import json
import matplotlib
import matplotlib.pyplot as plt
import os

# Load JSON


#SETUP
plot_dir = "Plots"
os.makedirs(plot_dir, exist_ok=True)

import pandas as pd


def create_vectordb_vs_embedder_plot(metric, title, save_path, factor):
    CORRECTNESS_METRIC = './metrics/db_chunk_comparison/FAISS/correctness_FAISS.json'
    RELEVENT_REFS_METRIC = './metrics/db_chunk_comparison/FAISS/relevant_refs_FAISS.json'
    res={}
    metrics = pd.read_csv("metrics/results.csv")
    for row in metrics.iterrows():
        print(row)
        
        db, embedder, method, cs= row[1]['index'].split("_")
        # Open the JSON file
        
        

        res[f"{db}_{embedder}_{method}_{cs}"] = row[1][metric]

    res =  dict(sorted(res.items(), key=lambda x: int(x[0].rsplit("_", 1)[-1])))

    print(res.keys())
    compare_db_with_chatlas, compare_db_with_multi, compare_emb_with_chroma, compare_emb_faiss = [], [], [], []
    compare_emb_with_chroma_CHATvalues, compare_db_with_chatlas_FAISSvalues, compare_db_with_multi_CHROMAvalues, compare_db_with_multi_FAISSvalues = [], [], [], []
    compare_db_with_chatlas_CHROMAvalues, compare_emb_with_chroma_MULTIvalues, compare_emb_faiss_CHATvalues, compare_emb_faiss_MULTIvalues = [], [], [], []
    text_v_semantic_chunking, text_v_semantic_chunking_values=[],[]
    data=[]
    for x in res:
        l=x.split('_')[-1]
        if '8000' in x:
            if 'semantic' in x:
                text_v_semantic_chunking.append('Semantic Chunking')
            else:
                text_v_semantic_chunking.append('Text Chunking')

            text_v_semantic_chunking_values.append(res[x]) 
        if '7000' in l:
            l = 'NoChunking'
        if '800' in l: continue
        if ('Chroma_chatlas' in x or 'FAISS_chatlas' in x) and 'embedder' in x:
            if l not in  compare_db_with_chatlas:compare_db_with_chatlas.append(l)
            if 'Chroma' in x:
                compare_db_with_chatlas_CHROMAvalues.append(res[x])
            else:
                compare_db_with_chatlas_FAISSvalues.append(res[x])
            
            
        if ('Chroma_multi' in x or 'FAISS_multi-qa' in x) and 'embedder' in x:
            if l not in compare_db_with_multi: compare_db_with_multi.append(l)
            if 'Chroma' in x:
                compare_db_with_multi_CHROMAvalues.append(res[x])
                if '8000' in x:
                    compare_db_with_multi_FAISSvalues.append(0)
            else:
                compare_db_with_multi_FAISSvalues.append(res[x])
                

        if ('Chroma_chatlas' in x or 'Chroma_multi' in x) and 'embedder' in x:
            if l not in compare_emb_with_chroma: compare_emb_with_chroma.append(l)
            if 'chatlas' in x:
                compare_emb_with_chroma_CHATvalues.append(res[x])
            else:
                compare_emb_with_chroma_MULTIvalues.append(res[x])
                if '8000' in x:
                    compare_emb_with_chroma_CHATvalues.append(0)
        if ('FAISS_chatlas' in x or 'FAISS_multi-qa' in x) and 'embedder' in x:
            if l not in compare_emb_faiss: compare_emb_faiss.append(l)
            if 'chatlas' in x:
                compare_emb_faiss_CHATvalues.append(res[x])
            else:
                compare_emb_faiss_MULTIvalues.append(res[x])
        
    print(compare_emb_with_chroma_MULTIvalues)
     
    data.append({'title': 'Comparing Chatlas against VectorStores', 'labels':compare_db_with_chatlas, 'values1': compare_db_with_chatlas_CHROMAvalues, 'values2': compare_db_with_chatlas_FAISSvalues })
    data.append({'title': 'Comparing Multi-QA against VectorStores', 'labels':compare_db_with_multi, 'values1': compare_db_with_multi_CHROMAvalues, 'values2': compare_db_with_multi_FAISSvalues})
    data.append({'title': 'Comparing Embeddings with Chroma', 'labels':compare_emb_with_chroma, 'values1': compare_emb_with_chroma_CHATvalues , 'values2':compare_emb_with_chroma_MULTIvalues})
    data.append({'title': 'Comparing Embeddings with FAISS', 'labels':compare_emb_faiss, 'values1': compare_emb_faiss_CHATvalues, 'values2': compare_emb_faiss_MULTIvalues})

    # Set up a 2x2 grid
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    import numpy as np
    # Plot each chart
    bar_width = 0.35  # width of each bar
    for i in range(4):
        labels = data[i]["labels"]
        
        x = np.arange(len(labels))  # x locations for the groups
        values1 = data[i]["values1"]
        values2 = data[i]["values2"]
   

        ax = axes[i]
        if i < 2:
            print(labels,len(values1), len(values2))
            bars1 = ax.bar(x - bar_width/2, values1, width=bar_width, label='CHROMA')#, color='skyblue')
            bars2 = ax.bar(x + bar_width/2, values2, width=bar_width, label='FAISS')#, color='salmon')
        else:
            bars1 = ax.bar(x - bar_width/2, values1, width=bar_width, label='CHATLAS')#, color='skyblue')
            bars2 = ax.bar(x + bar_width/2, values2, width=bar_width, label='MULTI-QA')#, color='salmon')
        ax.bar_label(bars1,fmt='%.3f', padding=3, fontsize=9)
        ax.bar_label(bars2, fmt='%.3f',padding=3, fontsize=9)
        ax.set_title(f"{title}:\n{data[i]['title']}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Chunk Size")
        ax.legend()
    



    #fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    x=np.arange(2)
    bars1=axes[-1].bar(x,text_v_semantic_chunking_values)#color='skyblue' )
    axes[-1].bar_label(bars1,fmt='%.3f', padding=3, fontsize=9)
    axes[-1].set_title(f"{title}:\nComparing Chunking")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(text_v_semantic_chunking)
    axes[-1].set_ylabel("Accuracy")
    axes[-1].set_xlabel("Chunk Size")
    axes[-1].legend()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.savefig(save_path)


#create_vectordb_vs_embedder_plot('correctness', 'LLM Response Accuracy', 'correctness_plot.png',factor=2)
#create_vectordb_vs_embedder_plot('relevant_refs', 'LLM Source Retrieval Accuracy', 'relevant_refs_plot.png',factor=1)
#create_plot(RELEVENT_REFS_METRIC, 'Extraction Method: PDF Plumber', 'Chunk Size', 'Relevant References', 'relevant_refs_plot.png')
import json
import collections
import numpy as np
def create_model_plot(basedir):
    correctness_results = collections.defaultdict(lambda: collections.defaultdict(float))
    for model in os.listdir(basedir):
        if 'qwen' not in model: continue
        if '.DS' in model: continue
        for param in os.listdir(os.path.join(basedir, model)):
            if '.DS' in param: continue
            filename = os.path.join(basedir, model,param, 'relevant_refs', 'relevant_refs.json')
            
            try:
                with open(filename,"r") as file:
                    data = json.load(file)
                    
                for k in data:
                    correctness_results[model][param]=data[k]
            except Exception as e:
                print(f"Skipping {model} {param} bc of {e}")
    # Extract keys (x-axis)
   
    # Collect all unique keys across models
    all_keys = sorted({k for subdict in correctness_results.values() for k in subdict.keys()})
    models = list(correctness_results.keys())

    x = np.arange(len(all_keys))  # positions for each key
    width = 0.8 / len(models)     # width for each model’s bar

    # Assign colors per model
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    # Plot each model’s bars
    for i, model in enumerate(models):
        vals = [correctness_results[model].get(k, 0) for k in all_keys]
        print(vals)
        plt.bar(x + i * width, vals, width, label=model)

    # Format
    plt.xticks(x + width * (len(models) - 1) / 2, all_keys, rotation=90)
    plt.xlabel("Key")
    plt.ylabel("Value")
    plt.title("Model Comparison by Key (missing = 0)")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()


    # Now `data` is a Python dict (or list, depending on the JSON structure)

    plt.savefig("metrics/models.png")

#create_model_plot('/Users/rishikasrinivas/SULI/DUNEGPT_CHROMAFASS/LLM_for_DUNE/LLM_for_DUNE/metrics/metrics')

def create_reranker_plot(metric):
    title = 'Source Retrieval' if metric == 'relevant_refs' else 'LLM Response'
    scores = {
        "Chroma_multi-qa_embedder_2000_noreranking": 0.6971,
        "Chroma_multi-qa_embedder_2000_withreranking": 'gpt4o', # 0.63
        'e5_embedding768_with_reranking_chroma_2000': 'gpt4o',
        'e5_embedding768_no_reranking_chroma_2000': 'gpt4o',
    }

    labels = {
        "Chroma_multi-qa_embedder_2000_noreranking": 'Chroma_MultiQA_NoRerank_2000',
        "Chroma_multi-qa_embedder_2000_withreranking": 'Chroma_MultiQA_Rerank_2000', # 0.63
        'e5_embedding768_with_reranking_chroma_2000': 'Chroma_e5_Rerank_2000',
        'e5_embedding768_no_reranking_chroma_2000': 'Chroma_e5_NoRerank_2000',
    }
    results={}

    for method in scores:
        if isinstance(scores[method],float):
            results[method] = scores[method]
        else:
            with open(os.path.join("metrics", scores[method], method, f'{metric}/{metric}.json'), "r") as f:
                data = json.load(f)
            for _, value in data.items():
                results[method] = value

    # Split keys and values
    labels = list(labels.values())
    values = list(results.values())

    # Set bar colors (emphasize reranking in red)
    colors = ['skyblue' for i in range(len(labels))]  # Assume second is reranked

    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors)

    # Add score labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}",
                ha='center', va='bottom', fontsize=12)

    # Highlight the drop due to reranking
    plt.title(f"Effect of Reranking on {title} Performance", fontsize=14)
    plt.ylabel("Score", fontsize=12)
    plt.xticks(rotation=25, ha='center')
    plt.ylim(0, 1.0)   
    plt.tight_layout()
    plt.savefig(f"metrics/{metric}_comparison.png") 
import matplotlib.pyplot as plt
import numpy as np

def generate_vectordb_vs_embedder_plot(metric):
    """
    Generate a side-by-side bar plot for two bar groups.

    Parameters:
        bar1name (str): Label for the first bar group.
        bar1values (list): Values for the first bar group.
        bar2name (str): Label for the second bar group.
        bar2values (list): Values for the second bar group.
    """
    
    if metric == 'correctness':
        ylabel = 'LLM Accuracy'
        title = 'Accuracy per Chunk Size'
        filename = os.path.join(plot_dir, 'correctness_Chroma_v_Faiss.png')
        metric_file = CORRECTNESS_METRIC
    elif metric == 'relevant_refs':
        ylabel = 'Source Retrieval Accuracy'
        title = 'Retrieval Per Chunk Size'
        filename = os.path.join(plot_dir, 'retrieval_Chroma_v_Faiss.png')
        metric_file = RELEVENT_REFS_METRIC

    with open(metric_file, "r") as f:
        data = json.load(f)
    
    dataset_faiss = {}
    dataset_chroma = {}
    for key in data:
        if '800' in key: print(key)
        if 'FAISS' in key:
            o_key=key
            if '7000' in key:
                o_key=key
                key = '_No Chunking'
            
            dataset_faiss[key] = data[o_key]/5
        elif 'Chroma' in key:
            o_key=key
            if '7000' in key:
                key = '_No Chunking'
            dataset_chroma[key] = data[o_key]/2

    categories = [f'{key.split("_")[1]}' for key in dataset_faiss]
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots()
    bar1=ax.bar(x - width/2, dataset_faiss.values(), width, label='faiss')
    bar2=ax.bar(x + width/2, dataset_chroma.values(), width, label='chroma')

    for bar in bar1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01 * max(dataset_faiss.values()), f'{yval:.2f}', ha='center', va='bottom')

    for bar in bar2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01 * max(dataset_chroma.values()), f'{yval:.2f}', ha='center', va='bottom')


    ax.set_xlabel('Chunk Sizes')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    plt.tight_layout()
    plt.show()
    #plt.savefig(filename)
#generate_vectordb_vs_embedder_plot(metric='relevant_refs')
#create_reranker_plot(metric='correctness')

def create_search_distinction_plot(methods, metric,save_path):
    accuracy = defaultdict()
    for method in methods:
        with open(f"./metrics/gpt4o/sigmoid_{method}_e5_embedder_with_priority_2000/relevant_refs/{metric}.json", 'r') as f:
            res = json.load(f)
            accuracy[method] = res['e5_embedder_with_priority_2000']
    fig, ax = plt.subplots()
    x=np.arange(2)
    x=np.arange(2)
    bars1=ax.bar(x,accuracy.values(),color='skyblue' )
    ax.bar_label(bars1,fmt='%.3f', padding=3, fontsize=9)
    ax.set_title(f"{metric.upper()}:\nComparing Search Methods")
    ax.set_xticks(x)
    ax.set_xticklabels(accuracy.keys())
    ax.set_ylabel("Retrival Accuracy")
    ax.set_xlabel("With/Without Document vs Slide Separation")
    ax.legend()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.savefig(save_path)
#create_search_distinction_plot(['distinction', 'withoutdistinction'], 'relevant_refs',"./metrics/searchComparison.png")

def plot_search_keyword_configuration_parameters_keyword_nokey(method, flder, title):
    accuracy = defaultdict(dict)
    root=f"./metrics/{flder}"
    
    for folder in os.listdir(root):
        path = os.path.join(root, folder, method, f"{method}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) 
                accuracy[folder] = data['data']
        except:
            print('dne')
    accuracy = dict(sorted(accuracy.items()))
    
    # Separate key_ and no_key configurations
    key_configs = {}
    no_key_configs = {}
    
    for k, v in accuracy.items():
        value = v/2 if v > 1 else v
        if k.startswith('keyword'):
            # Get everything after 'key_'
            base_name = k[len('keyword'):]  # Remove 'key_' (4 characters)
            key_configs[base_name] = value
        elif k.startswith('no_keyword'):
            # Get everything after 'no_key_'
            base_name = k[len('no_keyword'):]  # Remove 'no_key_' (7 characters)
            no_key_configs[base_name] = value
    
    # Get all unique base configurations
    all_configs = sorted(set(list(key_configs.keys()) + list(no_key_configs.keys())))
    
    # Prepare data for plotting
    x_labels = [' '.join(config.split("_")) for config in all_configs]
    key_values = [key_configs.get(config, 0) for config in all_configs]
    no_key_values = [no_key_configs.get(config, 0) for config in all_configs]
    
    # Create bar positions
    x_pos = np.arange(len(all_configs))
    width = 0.35
    
    # Plot adjacent bars

    bars1 =plt.bar(x_pos - width/2, key_values, width, label='keyword', color='steelblue')
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8)
    
    if any('no' in k for k in accuracy):
        bars2 =plt.bar(x_pos + width/2, no_key_values, width, label='no keyword', color='coral')
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=8)

    else:
        print("skippng none")
    
    
    plt.xlabel('Configuration')
    plt.ylabel('Accuracy')
    plt.xticks(x_pos, x_labels, rotation=45, ha='right')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"Plots/{'_'.join(title.split())}.png")
    
    print(f"Configurations: {all_configs}")
    print(f"key_ =values: {key_values}")
    print(f"no_key values: {no_key_values}")
#plot_search_keyword_configuration_parameters_keyword_nokey('relevant_refs', 'gpt4o_return3_gen_newsearch', 'Retrieval Acc: Search Config K Documents, Slides, Keyword and Returning top 3')
#plot_search_keyword_configuration_parameters_keyword_nokey('correctness', 'gpt4o_return3_gen_newsearch', 'Response Acc: Search Config K Documents, Slides, Keyword and Returning top 3')

import os
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def plot_search_methods(old_search_path, new_search_path):
    accuracy = defaultdict(lambda: defaultdict(float))
   
    # --- Load Old Search ---
    for metric in os.listdir(old_search_path):
        path = os.path.join(old_search_path, metric, f'{metric}.json')
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) 
            accuracy['old_search'][metric] = data['data']

    # --- Load New Search ---
    for metric in os.listdir(new_search_path):
        path = os.path.join(new_search_path, metric, f'{metric}.json')
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) 
            accuracy['new_search'][metric] = data['data']

    categories = ['old_search', 'new_search']

    # Subplot metric groups
    subplot1_metrics = ['correctness', 'relevant_refs']
    subplot2_metrics = ['latency']

    # Set up figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # -------------------------------
    #   SUBPLOT 1 — correctness + relevant_res
    # -------------------------------
    ax = axes[0]
    width = 0.3
    x = np.arange(len(categories))

    colors = ['#1f77b4', '#2ca02c']  # blue, green

    for i, metric in enumerate(subplot1_metrics):
        values = [accuracy[cat][metric] for cat in categories]
        bars = ax.bar(x + i * width, values, width, label=metric.replace("_", " ").title(), color=colors[i])

        # Add labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_title("Correctness & Relevant Results")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Without Keyword and Document/Slide\n Distinction\n(18 attachments + reranking\n select top 3)", "With Keyword and Document/Slide\n Distinction (5 docs + 5 slides + 5 keyword\n select top 3)"])
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # -------------------------------
    #   SUBPLOT 2 — latency
    # -------------------------------
    ax = axes[1]

    metric = 'latency'
    values = [accuracy[cat][metric] for cat in categories]
    bars = ax.bar(x, values, width=0.4, color='#ff7f0e', label='Latency')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_title("Latency Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(["Without Keyword and Document/Slide\n Distinction\n(18 attachments + reranking\n select top 3)", "With Keyword and Document/Slide\n Distinction (5 docs + 5 slides + 5 keyword\n select top 3)"])
    ax.set_ylabel("Time (seconds)")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("Plots/SearchMethod+Configuration.png")
    plt.show()


plot_search_methods()

