#!/bin/bash


db=Chroma
cs=2000
chatlas=("Faiss-chatlas" "Chroma-chatlas")
multiqa=("data")  #("Faiss-multiqa", "Chroma-multiqa")
for dir in "${multiqa[@]}"; do
    for item in "${multiqa[@]}"; do
        name=$(basename "$item")
        #save=metrics/gpt4o/${name}/correctness
        echo $item
        echo $name
        echo $save
	model="gpt-oss:120b"
	for k in 2 3 5 6 7 9; do
		save="metrics/$model/keyword_top_k_${k}_${name}/latency"
		if [ ! -d "$save" ]; then
                        mkdir -p "$save"
                fi
                python3.11 -m benchmarking.evaluation --port 44059 --model $model --experiment_name test --method latency --data_path "$item" --savedir $save --keyword --top_k $k

		#save="metrics/$model/keyword_top_k_${k}_${name}/relevant_refs"
		#if [ ! -d "$save" ]; then
                        #mkdir -p "$save"
                #fi
		#python3.11 -m benchmarking.evaluation --port 9 --model $model --experiment_name test --method relevent_refs --data_path "$item" --savedir $save --keyword --top_k $k
	 	
		#save="metrics/$model/no_keyword_top_k_${k}_${name}/relevant_refs"
		#if [ ! -d "$save" ]; then
                #        mkdir -p "$save"
                #fi
		#python3.11 -m benchmarking.evaluation --port 9 --model $model --experiment_name test --method relevent_refs --data_path "$item" --savedir $save --top_k $k
	done
	#models=("qwen2.5vl:latest" "nomic-embed-text:latest" "qwen2.5-coder:1.5b" "llama3.1:8b" "llama3.2:latest" "gemma3:latest")
	#models=("qwen2.5vl:latest")
	#for model in "${models[@]}"; do
        #        echo "Processing model: $model"
        #        save="metrics/$model/keyword_general_top_k_3_${name}/correctness"
        #        if [ ! -d "$save" ]; then
        #                mkdir -p "$save"
        #        fi
        #       #python3.11 -m benchmarking.evaluation --port 9 --model $model --experiment_name test --method correctness --data_path "$item" --savedir $save
        #done

	#models=("gpt-oss:20b" "qwen2.5vl:latest" "nomic-embed-text:latest" "mixtral:latest" \
        #"qwen2.5-coder:1.5b" "llama3.1:8b" "llama3.2:latest" "gemma3:latest" "Qwen3-coder:latest")
	#	if [[ "$model" == "qwen2.5vl:latest" ]]; then
    	#		echo "Processing model: $model"
	#		save="metrics/$model/keyword_general_top_k_3_${name}/relevant_refs"
        #		if [ ! -d "$save" ]; then
        #        		mkdir -p "$save"
        #		fi
        #		python3.11 -m benchmarking.evaluation --port 9 --model $model --experiment_name test --method relevent_refs --data_path "$item" --savedir $save
	
	#	fi
	#done

        
	#save=metrics/gpt-oss:120b/no_keyword_general_top_k_9_${name}/latency
        #if [ ! -d "$save" ]; then
         #       mkdir -p "$save"
        #fi
        #python3.11 -m benchmarking.evaluation --port 9 --experiment_name test --method latency --data_path "$item" --savedir $save
    done
done

