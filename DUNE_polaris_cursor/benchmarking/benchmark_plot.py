import json
import matplotlib.pyplot as plt
import os

# Load JSON


def create_plot(metric, title, save_path, factor):
    res={}
    for dir in os.listdir("metrics/"):
        db, embedder,  cs= dir.split("_")[0], dir.split("_")[1], dir.split("_")[-1]
        # Open the JSON file
        with open(os.path.join("metrics", dir, metric, f'{metric}.json'), 'r') as file:
            data = json.load(file)
        for k in data:
            acc = data[k]/factor
        res[f"{db}_{embedder}_{cs}"] = acc
    print(res)
    
    compare_db_with_chatlas, compare_db_with_multi, compare_emb_with_chroma, compare_emb_faiss = [], [], [], []
    compare_emb_with_chroma_CHATvalues, compare_db_with_chatlas_FAISSvalues, compare_db_with_multi_CHROMAvalues, compare_db_with_multi_FAISSvalues = [], [], [], []
    compare_db_with_chatlas_CHROMAvalues, compare_emb_with_chroma_MULTIvalues, compare_emb_faiss_CHATvalues, compare_emb_faiss_MULTIvalues = [], [], [], []
    data=[]
    for x in res:
        l=x.split('_')[-1]
        if '7000' in l:
            l = 'NoChunking'
        if 'Chroma_chatlas' in x or 'FAISS_chatlas' in x:
            compare_db_with_chatlas.append(l)
            if 'Chroma' in x:
                compare_db_with_chatlas_CHROMAvalues.append(res[x])
            else:
                compare_db_with_chatlas_FAISSvalues.append(res[x])
            
            
        if 'Chroma_multi' in x or 'FAISS_multi-qa' in x:
            compare_db_with_multi.append(l)
            if 'Chroma' in x:
                compare_db_with_multi_CHROMAvalues.append(res[x])
            else:
                compare_db_with_multi_FAISSvalues.append(res[x])

        if 'Chroma_chatlas' in x or 'Chroma_multi' in x :
            compare_emb_with_chroma.append(l)
            if 'chatlas' in x:
                compare_emb_with_chroma_CHATvalues.append(res[x])
            else:
                compare_emb_with_chroma_MULTIvalues.append(res[x])
        if 'FAISS_chatlas' in x or 'FAISS_multi-qa' in x:
            compare_emb_faiss.append(l)
            if 'chatlas' in x:
                compare_emb_faiss_CHATvalues.append(res[x])
            else:
                compare_emb_faiss_MULTIvalues.append(res[x])
   
    data.append({'title': 'Comparing Chatlas against VectorStores', 'labels':list(set(compare_db_with_chatlas)), 'values1': compare_db_with_chatlas_CHROMAvalues, 'values2': compare_db_with_chatlas_FAISSvalues })
    data.append({'title': 'Comparing Multi-QA against VectorStores', 'labels':list(set(compare_db_with_multi)), 'values1': compare_db_with_multi_CHROMAvalues, 'values2': compare_db_with_multi_FAISSvalues})
    data.append({'title': 'Comparing Embeddings with Chroma', 'labels':list(set(compare_emb_with_chroma)), 'values1': compare_emb_with_chroma_CHATvalues , 'values2':compare_emb_with_chroma_MULTIvalues})
    data.append({'title': 'Comparing Embeddings with FAISS', 'labels':list(set(compare_emb_faiss)), 'values1': compare_emb_faiss_CHATvalues, 'values2': compare_emb_faiss_MULTIvalues})

    # Set up a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    import numpy as np
    # Plot each chart
    bar_width = 0.35  # width of each bar
    for i in range(4): #4 plots: 2 comparing DB's 2 comparing Embedders 
        labels = data[i]["labels"]
        
        x = np.arange(len(labels))  # x locations for the groups
        values1 = data[i]["values1"]
        values2 = data[i]["values2"]
   

        ax = axes[i]
        if i < 2:
            bars1 = ax.bar(x - bar_width/2, values1, width=bar_width, label='CHROMA', color='skyblue')
            bars2 = ax.bar(x + bar_width/2, values2, width=bar_width, label='FAISS', color='salmon')
        else:
            bars1 = ax.bar(x - bar_width/2, values1, width=bar_width, label='CHATLAS', color='skyblue')
            bars2 = ax.bar(x + bar_width/2, values2, width=bar_width, label='MULTI-QA', color='salmon')
        ax.bar_label(bars1,fmt='%.3f', padding=3, fontsize=9)
        ax.bar_label(bars2, fmt='%.3f',padding=3, fontsize=9)
        ax.set_title(f"{title}:\n{data[i]['title']}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Chunk Size")
        ax.legend()
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.savefig(save_path)

plot_dir = "Plots"
os.makedirs(plot_dir, exist_ok=True)

create_plot('correctness', 'LLM Response Accuracy', os.path.join(plot_dir,'correctness_plot.png'),factor=2)
create_plot('relevant_refs', 'LLM Source Retrieval Accuracy', os.path.join(plot_dir, 'relevant_refs_plot.png'),factor=1)
