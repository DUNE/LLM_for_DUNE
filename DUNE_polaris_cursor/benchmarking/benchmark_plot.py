from collections import defaultdict
import json
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
# Load JSON


#SETUP
plot_dir = "Plots"
os.makedirs(plot_dir, exist_ok=True)


def create_vectordb_vs_embedder_plot(metric, title, save_path, factor):
    """
    Creates comparison plots for vector databases and embedders.
    
    This function generates 5 subplots:
    1-2: Compare vector databases (CHROMA vs FAISS) for fixed embedders
    3-4: Compare embedders (CHATLAS vs MULTI-QA) for fixed databases
    5: Compare chunking strategies (Text vs Semantic)

    args:
        metric: 'correctness' or 'relevant_refs'
        title: Title of final plot
        save_path: Where to save plot (as .png)
        factor: If results are accumulated over multiple runs and need to be averaged out
    """
    
    # Load metrics data
    metrics = pd.read_csv("metrics/results.csv")
    results = {}
    
    # Parse metrics into a dictionary
    for _, row in metrics.iterrows():
        db, embedder, method, chunk_size = row['index'].split("_")
        key = f"{db}_{embedder}_{method}_{chunk_size}"
        results[key] = row[metric]
    
    # Sort by chunk size
    results = dict(sorted(results.items(), key=lambda x: int(x[0].rsplit("_", 1)[-1])))
    
    # Initialize data structures for different comparisons
    comparisons = {
        'chatlas_dbs': {'labels': [], 'chroma': [], 'faiss': []},
        'multiqa_dbs': {'labels': [], 'chroma': [], 'faiss': []},
        'chroma_embedders': {'labels': [], 'chatlas': [], 'multiqa': []},
        'faiss_embedders': {'labels': [], 'chatlas': [], 'multiqa': []},
        'chunking': {'labels': [], 'values': []}
    }
    
    # Parse results and organize data for plotting
    for key, value in results.items():
        parts = key.split('_')
        db, embedder, method, chunk_size = parts[0], parts[1], parts[2], parts[3]
        
        # Format chunk size label
        label = 'NoChunking' if '7000' in chunk_size else chunk_size
        
        # Skip 800 chunk size
        if '800' in chunk_size:
            continue
        
        # Chunking comparison (only for 8000 chunk size)
        if '8000' in chunk_size:
            chunking_type = 'Semantic Chunking' if 'semantic' in key else 'Text Chunking'
            comparisons['chunking']['labels'].append(chunking_type)
            comparisons['chunking']['values'].append(value)
        
        # Only process embedder comparisons
        if 'embedder' not in key:
            continue
        
        # Compare databases with fixed CHATLAS embedder
        if 'chatlas' in embedder:
            if label not in comparisons['chatlas_dbs']['labels']:
                comparisons['chatlas_dbs']['labels'].append(label)
            
            if db == 'Chroma':
                comparisons['chatlas_dbs']['chroma'].append(value)
            else:  # FAISS
                comparisons['chatlas_dbs']['faiss'].append(value)
        
        # Compare databases with fixed MULTI-QA embedder
        if 'multi' in embedder:
            if label not in comparisons['multiqa_dbs']['labels']:
                comparisons['multiqa_dbs']['labels'].append(label)
            
            if db == 'Chroma':
                comparisons['multiqa_dbs']['chroma'].append(value)
                # Add placeholder for missing FAISS data at chunk size 8000
                if '8000' in chunk_size:
                    comparisons['multiqa_dbs']['faiss'].append(0)
            else:  # FAISS
                comparisons['multiqa_dbs']['faiss'].append(value)
        
        # Compare embedders with fixed CHROMA database
        if db == 'Chroma':
            if label not in comparisons['chroma_embedders']['labels']:
                comparisons['chroma_embedders']['labels'].append(label)
            
            if 'chatlas' in embedder:
                comparisons['chroma_embedders']['chatlas'].append(value)
            else:  # multi-qa
                comparisons['chroma_embedders']['multiqa'].append(value)
                # Add placeholder for missing chatlas data at chunk size 8000
                if '8000' in chunk_size:
                    comparisons['chroma_embedders']['chatlas'].append(0)
        
        # Compare embedders with fixed FAISS database
        if db == 'FAISS':
            if label not in comparisons['faiss_embedders']['labels']:
                comparisons['faiss_embedders']['labels'].append(label)
            
            if 'chatlas' in embedder:
                comparisons['faiss_embedders']['chatlas'].append(value)
            else:  # multi-qa
                comparisons['faiss_embedders']['multiqa'].append(value)
    
    # Prepare plot configurations
    plot_configs = [
        {
            'title': 'Comparing Chatlas against VectorStores',
            'data': comparisons['chatlas_dbs'],
            'legend': ['CHROMA', 'FAISS']
        },
        {
            'title': 'Comparing Multi-QA against VectorStores',
            'data': comparisons['multiqa_dbs'],
            'legend': ['CHROMA', 'FAISS']
        },
        {
            'title': 'Comparing Embeddings with Chroma',
            'data': comparisons['chroma_embedders'],
            'legend': ['CHATLAS', 'MULTI-QA']
        },
        {
            'title': 'Comparing Embeddings with FAISS',
            'data': comparisons['faiss_embedders'],
            'legend': ['CHATLAS', 'MULTI-QA']
        }
    ]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    bar_width = 0.35
    
    # Plot database and embedder comparisons
    for i, config in enumerate(plot_configs):
        labels = config['data']['labels']
        x = np.arange(len(labels))
        
        # Get values based on plot type
        if i < 2:  # Database comparisons
            values1 = config['data']['chroma']
            values2 = config['data']['faiss']
        else:  # Embedder comparisons
            values1 = config['data']['chatlas']
            values2 = config['data']['multiqa']
        
        # Create bars
        bars1 = axes[i].bar(x - bar_width/2, values1, width=bar_width, label=config['legend'][0])
        bars2 = axes[i].bar(x + bar_width/2, values2, width=bar_width, label=config['legend'][1])
        
        # Add labels and formatting
        axes[i].bar_label(bars1, fmt='%.3f', padding=3, fontsize=9)
        axes[i].bar_label(bars2, fmt='%.3f', padding=3, fontsize=9)
        axes[i].set_title(f"{title}:\n{config['title']}")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(labels)
        axes[i].set_ylabel("Accuracy")
        axes[i].set_xlabel("Chunk Size")
        axes[i].legend()
    
    # Plot chunking comparison
    x = np.arange(len(comparisons['chunking']['labels']))
    bars = axes[4].bar(x, comparisons['chunking']['values'])
    axes[4].bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    axes[4].set_title(f"{title}:\nComparing Chunking")
    axes[4].set_xticks(x)
    axes[4].set_xticklabels(comparisons['chunking']['labels'])
    axes[4].set_ylabel("Accuracy")
    axes[4].set_xlabel("Chunking Strategy")
    
    # Final adjustments and save
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.savefig(save_path)
    plt.close()
    


def create_model_plot(basedir, metric):
    """
    Creates a bar chart comparing different models on a given metric.
    
    This function:
    1. Loads metric scores for each model and parameter combination
    2. Creates grouped bar charts showing how each model performs
    3. Groups bars by parameter, with each model shown side-by-side
    
    Args:
        basedir: Base directory containing model subdirectories
        metric: The metric to evaluate (e.g., 'correctness', 'accuracy')

    Directory
        basedir
            |__ modelname
                |__ experiment (param)
                    |__ metric
                        |__ {metric}_score.csv
                        |__ {metric}_tracking.csv
            |__ modelname2
                |__ etc...
    """
    
    # Store scores: model -> parameter -> score
    model_scores = defaultdict(lambda: defaultdict(float))
    
    # Load scores for each model
    for model_name in os.listdir(basedir):
        # Skip non-model directories
        if '.DS' in model_name:
            continue
        
        model_path = os.path.join(basedir, model_name)
        
        # Load scores for each parameter configuration
        for param in os.listdir(model_path):
            if '.DS' in param:
                continue
            
            try:
                # Load the metric score CSV file
                score_file = os.path.join(basedir, model_name, param, metric, f'{metric}_score.csv')
                scores_df = pd.read_csv(score_file)
                
                # Extract the score value
                model_scores[model_name][param] = scores_df['score'].values[0]
                
            except Exception as e:
                print(f"Skipping {model_name}/{param}: {e}")
    
    # Prepare data for plotting
    models = list(model_scores.keys())
    parameters = sorted({param for model in model_scores.values() for param in model.keys()})
    
    # Set up bar positions
    x_positions = np.arange(len(parameters))
    bar_width = 0.8 / len(models)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bars for each model
    for i, model in enumerate(models):
        # Get scores for this model across all parameters
        scores = [model_scores[model].get(param, 0) for param in parameters]
        
        # Calculate bar positions for this model
        offset = i * bar_width
        ax.bar(x_positions + offset, scores, bar_width, label=model)
    
    # Format the plot
    ax.set_xlabel("Parameter Configuration")
    ax.set_ylabel(f"{metric.capitalize()} Score")
    ax.set_title(f"{metric.capitalize()}: Model Performance Comparison")
    ax.set_xticks(x_positions + bar_width * (len(models) - 1) / 2)
    ax.set_xticklabels(parameters, rotation=45, ha='right')
    ax.legend(title="Model", loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    output_path = f"Plots/{metric}_models.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")
    
    return model_scores
create_model_plot('./metrics')

import os
import json
import matplotlib.pyplot as plt

def create_distinction_plot(metric, title, save_filename, output_path):
    """
    Creates a bar chart comparing metric performance with and without some concept (ex: with text/slide distinction vs wo text/slide distinction).
    
    This function:
    1. Loads scores for different configurations
    2. Compares performance with <some concept> enabled vs disabled
    3. Shows the effect of <some concept> on the specified metric
    
    Args:
        metric: The metric to evaluate ('relevant_refs' for source retrieval, 
                or other metrics for LLM response quality)
    """
    
    # Set plot title based on metric
    plot_title = 'Source Retrieval' if metric == 'relevant_refs' else 'LLM Response'
    
    # Configuration: maps experiment keys to their data sources: can replace with any with/without based experiment
    # - float values: hardcoded scores
    # - string values: model names to load scores from files
    experiment_configs = {
        #EXPERIMENT: RESULT
        #OR
        #EXPERIMENT: FOLDER
    }
    # ex: {
        #"Chroma_multi-qa_embedder_2000_noreranking": 0.6971,
        #"Chroma_multi-qa_embedder_2000_withreranking": 'gpt4o',
        #'e5_embedding768_with_reranking_chroma_2000': 'gpt4o',
        #'e5_embedding768_no_reranking_chroma_2000': 'gpt4o',
    #}
    
    
    # Human-readable labels for display
    display_labels = {
        EXPERIMENT: x-axis label
    }
    #ex:{
        #"Chroma_multi-qa_embedder_2000_noreranking": 'Chroma_MultiQA_NoRerank_2000',
        #"Chroma_multi-qa_embedder_2000_withreranking": 'Chroma_MultiQA_Rerank_2000',
        #'e5_embedding768_with_reranking_chroma_2000': 'Chroma_E5_Rerank_2000',
        #'e5_embedding768_no_reranking_chroma_2000': 'Chroma_E5_NoRerank_2000',
    #}
    
    # Load scores for each experiment
    experiment_scores = {}
    
    for experiment_key, data_source in experiment_configs.items():
        if isinstance(data_source, float):
            # Use hardcoded score
            experiment_scores[experiment_key] = data_source
        else:
            # Load score from JSON file
            score_file = os.path.join("metrics", data_source, experiment_key, 
                                     metric, f'{metric}.json')
            try:
                with open(score_file, "r") as f:
                    data = json.load(f)
                # Extract the score (assumes single value in dict)
                experiment_scores[experiment_key] = list(data.values())[0]
            except Exception as e:
                print(f"Warning: Could not load {experiment_key}: {e}")
                experiment_scores[experiment_key] = 0
    
    # Prepare data for plotting
    labels = [display_labels[key] for key in experiment_scores.keys()]
    scores = list(experiment_scores.values())
    
    # Assign colors: highlight With/Without experiments
    colors = ['#ff7f7f' if 'With' in label else '#87ceeb' 
              for label in labels]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, scores, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add score labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
               f"{height:.3f}",
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Format the plot
    ax.set_title(f"{title}" 
                fontsize=14, fontweight='bold', pad=20)
                
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#87ceeb', edgecolor='black', label='Without'),
        Patch(facecolor='#ff7f7f', edgecolor='black', label='With')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save the plot
    plt.tight_layout()
    output_path = f"metrics/{metric}_{save_filename}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
   
    
    # Print summary statistics
    print(f"\n{plot_title} Scores:")
    for label, score in zip(labels, scores):
        print(f"  {label}: {score:.4f}")
    
    return experiment_scores
    

