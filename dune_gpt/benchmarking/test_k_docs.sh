#This script tests different k_docs values
data_dirs=("data") 

for item in "${data_dirs[@]}"; do
    name=$(basename "$item")
    model="gpt-oss:120b" 
    for k in {2,3,5,6,7,9}; do
            save="metrics/tok_$model/keyword_top_k_${k}_${name}/relevant_refs"
            if [ ! -d "$save" ]; then
                    mkdir -p "$save"
            fi
            python3.11 -m benchmarking.evaluation --port 9 --model $model --experiment_name test --method relevant_refs --data_path "$item" --savedir $save --keyword --top_k $k

            save="metrics/tok_$model/keyword_top_k_${k}_${name}/correctness"
            if [ ! -d "$save" ]; then
                    mkdir -p "$save"
            fi
            python3.11 -m benchmarking.evaluation --port 9 --model $model --experiment_name test --method correctness --data_path "$item" --savedir $save --top_k $k
            save="metrics/tok_$model/keyword_top_k_${k}_${name}/latency"
            if [ ! -d "$save" ]; then
                    mkdir -p "$save"
            fi
            python3.11 -m benchmarking.evaluation --port 9 --model $model --experiment_name test --method latency --data_path "$item" --savedir $save --keyword --top_k $k
    done
done

~           
