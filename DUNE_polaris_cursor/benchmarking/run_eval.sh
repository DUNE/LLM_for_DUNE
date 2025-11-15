#!/bin/bash


data_dirs=("data")  #("Faiss-multiqa", "Chroma-multiqa")
for dir in "${data_dirs[@]}"; do
    for item in "${data_dirs[@]}"; do
        name=$(basename "$item")
	model='gpt4o_return3_gen' #"gpt-oss:20b"
	for k in 3 5; do
		save="metrics/$model/keyword_top_k_${k}_${name}/relevant_refs"
		if [ ! -d "$save" ]; then
                        mkdir -p "$save"
                fi
		#python3.11 -m benchmarking.evaluation --port 9 --model $model --experiment_name test --method relevent_refs --data_path "$item" --savedir $save --keyword --top_k $k
	 	
		save="metrics/$model/keyword_top_k_${k}_${name}/correctness"
		if [ ! -d "$save" ]; then
                        mkdir -p "$save"
                fi
		python3.11 -m benchmarking.evaluation --port 9 --model $model --experiment_name test --method correctness --data_path "$item" --savedir $save --top_k $k

