#!/bin/bash

chunk_size="8000"  #0.36 0.48 0.59 0.67 0.73 0.79, 0.83 0.86 0.89 0.91 0.93 0.94 0.95"
data_path=CHROMA/Chroma
docdb_limit=1
indico_limit=1
log_file=benchmarking/benchmark_log.txt


for cs in ${chunk_size}
do
	echo "Running extraction with chunk size ${cs} and storing in ${data_path}" >> ${log_file}
	
	python3.11 cli.py index --docdb-limit ${docdb_limit} --indico-limit ${indico_limit} --data-path "${data_path}_${cs}" --chunk-size ${cs}
done
