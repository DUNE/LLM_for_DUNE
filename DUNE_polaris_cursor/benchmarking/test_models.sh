#!/bin/bash


db=Chroma
cs=2000
chatlas=("Faiss-chatlas" "Chroma-chatlas")
multiqa=("data")  #("Faiss-multiqa", "Chroma-multiqa")
for dir in "${multiqa[@]}"; do
	models=("qwen2.5vl:latest" "mixtral:latest" "gpt-oss:20b" "qwen2.5-coder:1.5b" "llama3.1:8b" "llama3.2:latest" "gemma3:latest")
	for model in "${models[@]}"; do	
		save="metrics/$model/keyword_top_k_3/correctness"
		if [ ! -d "$save" ]; then
                        mkdir -p "$save"
                fi
		python3.11 -m benchmarking.evaluation --port 9 --model $model --experiment_name test --method correctness --data_path "$dir" --savedir $save --keyword --top_k 3
	 	
	done
        
	#save=metrics/gpt-oss:120b/no_keyword_general_top_k_9_${name}/latency
        #if [ ! -d "$save" ]; then
         #       mkdir -p "$save"
        #fi
        #python3.11 -m benchmarking.evaluation --port 9 --experiment_name test --method latency --data_path "$item" --savedir $save
done

