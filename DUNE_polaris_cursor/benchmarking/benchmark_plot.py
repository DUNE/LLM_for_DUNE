import json
import matplotlib.pyplot as plt
import os

# Load JSON
CORRECTNESS_METRIC = '/home/newg2/Projects/LLM/DUNE/LLM_for_DUNE/metrics/FAISS/correctness_FAISS.json'
RELEVENT_REFS_METRIC = '/home/newg2/Projects/LLM/DUNE/LLM_for_DUNE/metrics/FAISS/relevant_refs_FAISS.json'

#SETUP
plot_dir = "Plots"
os.makedirs(plot_dir, exist_ok=True)


def create_plot(json_path, constant, x_label, y_label, save_path):
    with open(CORRECTNESS_METRIC, "r") as f:
        data = json.load(f)

    # Keys and values
    labels = [x.split('_')[-1] for x in data.keys()]
    values = list(data.values())

        # Create bar plot
    plt.bar(labels, values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{constant}|{x_label} vs {y_label}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)

#create_plot(CORRECTNESS_METRIC, 'Extraction Method: PDF Plumber', 'Chunk Size', 'Correctness', 'correctness_plot.png')
#create_plot(RELEVENT_REFS_METRIC, 'Extraction Method: PDF Plumber', 'Chunk Size', 'Relevant References', 'relevant_refs_plot.png')


import matplotlib.pyplot as plt
import numpy as np

def generate_plot(metric):
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
    elif metric == 'relevent_refs':
        ylabel = 'Source Retrieval Accuracy'
        title = 'Retrieval Per Chunk Size'
        filename = os.path.join(plot_dir, 'retrieval_Chroma_v_Faiss.png')
        metric_file = RELEVENT_REFS_METRIC

    with open(metric_file, "r") as f:
        data = json.load(f)
    
    dataset_faiss = {}
    dataset_chroma = {}
    for key in data:
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
    plt.savefig(filename)

for metrics in  ['correctness', 'relevent_refs']:
    generate_plot(metrics)
