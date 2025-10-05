#!/bin/bash


db=Chroma
cs=2000
chatlas=("Faiss-chatlas" "Chroma-chatlas")
multiqa=("NewFAISS")  #("Faiss-multiqa", "Chroma-multiqa")
for dir in "${multiqa[@]}"; do
    for item in "$dir"/*; do
        name=$(basename "$item")
        save=metrics/${name}/correctness
        echo $item
        echo $name
        echo $save
	
	#Run correctness evaluation	
        if [ ! -d "$save" ]; then
            mkdir -p "$save"
        fi
        #python3.11 -m benchmarking.evaluation --port 9 --experiment_name test --method correctness --data_path "$item" --savedir $save
	
	#Run relevant source retrieval evaluation
	save=metrics/gpt5/${name}/relevant_refs
	if [ ! -d "$save" ]; then
            mkdir -p "$save"
        fi
	python3.11 -m benchmarking.evaluation --port 9 --experiment_name test --method relevant_refs --data_path "$item" --savedir $save
        
    done
done

