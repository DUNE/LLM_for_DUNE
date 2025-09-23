import json
import matplotlib.pyplot as plt

# Load JSON
CORRECTNESS_METRIC = 'correctness.json'
RELEVENT_REFS_METRIC = 'relevant_refs.json'

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

def generate_plot():
    """
    Generate a side-by-side bar plot for two bar groups.

    Parameters:
        bar1name (str): Label for the first bar group.
        bar1values (list): Values for the first bar group.
        bar2name (str): Label for the second bar group.
        bar2values (list): Values for the second bar group.
    """

    with open(RELEVENT_REFS_METRIC, "r") as f:
        data = json.load(f)
    dataset_faiss = {}
    for key in data:
        if 'FAISS' in key:
            dataset_faiss[key] = data[key]
        elif 'Chroma' in key:
            dataset_chroma[key] = data[key]

    categories = [f'{key.split("_")[1]}' for key in dataset_faiss]
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, dataset_faiss.values(), width, label='faiss')
 #   ax.bar(x + width/2, dataset_chroma.values(), width, label='chroma')

    ax.set_xlabel('Chunk Sizes')
    ax.set_ylabel('Source Retrieval Accuracy')
    ax.set_title('Retrieval Per Chunk Size')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig("retrival_Chroma_v_Faiss.png")
generate_plot()
