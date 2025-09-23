#!/bin/bash

chunk_size="2000 4000 6000 8000 10000 7000000"  #0.36 0.48 0.59 0.67 0.73 0.79, 0.83 0.86 0.89 0.91 0.93 0.94 0.95"
data_path=test_store_FAISS/FAISS
docdb_limit=50
indico_limit=50
log_file=benchmarking/benchmark_log.txt
for cs in ${chunk_size}
do
	echo "Running extraction with chunk size ${cs} and storing in ${data_path}" >> ${log_file}
	#python3.11 cli.py index --docdb-limit ${docdb_limit} --indico-limit ${indico_limit} --data-path "${data_path}_${cs}" --chunk-size ${cs}
	echo "Extracted ${docdb_limit} from docdb and ${indico_limit} from indico. Stored extractions in ${data_path}" >> ${log_file}
	echo ${cs}
	for i in {1..5}; do
		save="metrics"/"correctness"/"Run_${i}"
		
		if [ ! -d "$save" ]; then
			mkdir -p "$save"
		fi
		python3.11 -m benchmarking.evaluation --port 9 --experiment_name test --method correctness --data_path "${data_path}_${cs}" --savedir $save
		echo "Ran correctness metric to ${save}" >> ${log_file}
		save="metrics"/"relevant_refs"/"Run_${i}"
		if [ ! -d "$save" ]; then
                        mkdir -p "$save"
		fi
		python3.11 -m ./benchmarking/evaluation --port 9 --experiment_name test --method references --data_path "${data_path}_${cs}" --savedir $save
		echo "Ran reference match metric" >> ${log_file}
	done
done

