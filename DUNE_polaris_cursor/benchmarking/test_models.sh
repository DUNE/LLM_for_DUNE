#!/bin/bash


data_dir=("data")  #("Faiss-multiqa", "Chroma-multiqa")
for dir in "${data_dir[@]}"; do
	models=("qwen2.5vl:latest" "mixtral:latest" "gpt-oss:20b" "qwen2.5-coder:1.5b" "llama3.1:8b" "llama3.2:latest" "gemma3:latest")
	for model in "${models[@]}"; do	
		save="metrics/$model/keyword_top_k_3/correctness"
		if [ ! -d "$save" ]; then
                        mkdir -p "$save"
                fi
		python3.11 -m benchmarking.evaluation --port 9 --model $model --experiment_name test --method correctness --data_path "$dir" --savedir $save --keyword --top_k 5
        
		save="metrics/$model/keyword_top_k_3/latency"
        	if [ ! -d "$save" ]; then
                	mkdir -p "$save"
        	fi
        	python3.11 -m benchmarking.evaluation --port 9 --experiment_name test --method latency --data_path "$item" --savedir $save --top_k 5
	done
done

