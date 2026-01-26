#!/bin/bash

chunk_size="2000"  #0.36 0.48 0.59 0.67 0.73 0.79, 0.83 0.86 0.89 0.91 0.93 0.94 0.95"
data_path=ChromaRerun/mutliqa
docdb_limit=100
indico_limit=100
log_file=benchmarking/benchmark_log.txt
ddb_start=0

ind_start=0

for cs in ${chunk_size}
do
	echo "Running extraction with chunk size ${cs} and storing in ${data_path}" >> ${log_file}
	python3.11 cli.py index --docdb-limit ${docdb_limit} --indico-limit ${indico_limit} --data-path "${data_path}_${cs}" --chunk-size ${cs} --start_idx_ddb $ddb_start --start_idx_ind $ind_start
	

done
