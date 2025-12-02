#!/bin/bash
#

# Active states in Balsam
active_states=("CREATED" "READY" "STAGED_IN" "PREPROCESSED" "RUNNING")

# Function to check if any jobs are in active state
has_active_jobs() {
    for s in "${active_states[@]}"; do
        if balsam job ls --state "$s" | grep -q "$s"; then
            return 0  # yes, active jobs exist
        fi
    done
    return 1  # no active jobs
}

for i in {0..24}; do
   	start=$((5 * i))
	end=$((5 * i + 5)) 
	python3.11 embed_docdb_indico.py --c $start
	echo "Launched 5 jobs starting from directory data/$start to data/$end"
	
	while has_active_jobs; do
		sleep 3600
	done
done
