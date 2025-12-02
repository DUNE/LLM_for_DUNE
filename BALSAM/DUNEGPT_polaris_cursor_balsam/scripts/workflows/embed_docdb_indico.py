import os

#os.environ["PATH"] = "/lus/flare/projects/LLM_for_DUNE/users/Rishika/LLM_for_DUNE/BALSAM/DUNEGPT_polaris_cursor_balsam/balsam_venv/bin:" + os.environ["PATH"]
#os.environ["PYTHONHOME"] = "/lus/flare/projects/LLM_for_DUNE/users/Rishika/LLM_for_DUNE/BALSAM/DUNEGPT_polaris_cursor_balsam/balsam_venv/"
from balsam.api import ApplicationDefinition, BatchJob, Job, Site
import yaml

#Machine variables
#########################
queue = "prod"         # debug or prod 
project="LLM_for_DUNE" #
job_mode="mpi"          #
#########################

#build larndsim again in the venv after changing the constant

#release of jobs
#first 16 jobs,
#then 10 jobs of 128 each

#need to filter/tag submit index for jobs while submitting otherwise balsam submits
#all of the jobs 

#default prod
cpu_packing_count = 1
#need to experiment with this as we have 1024 jobs
gpu_packing_count = 1

if queue == "debug":
    cpu_packing_count = 1

name = 'embed_docdb_indico'

site_name = "DUNEGPT_polaris_cursor_balsam"

runs = ["all","submit_all"]

version = "v0"

"""
Functions
"""
num_jobs=5
num_nodes=5
wall_time_min = 300 #180

doc_limit = 1000 if queue == 'prod' else 50 #-------16K for indico 37K for ddb
indico_limit = 1 if queue == 'prod' else 1
def create_single_dependent_job(app_id, i, parent_ids, params, node_packing_count, start):
    params.update({"i": i})
    return Job.objects.create( app_id=app_id,
                        site_name=site_name,
                        workdir=f"workdir/{version}_{i}_{app_id}",
                        node_packing_count=node_packing_count,
                        tags={"workflow": f"{name}_{app_id}_{version}", "queue": f"{queue}", "index":f"{i}", "job_start":f"{start}", 'num_nodes': f"{num_nodes}", 'ddb_document_limit' : f"{doc_limit}", 'ind_document_limit': f"{indico_limit}", 'ddb_start_idx': f"{params['ddb_start']}", 'ind_start_idx':f"{params['ind_start']}"},  
                        data=params,
                        parent_ids=parent_ids)



def submit_all_jobs(start, num_nodes=1, wall_time_min=60):
    site = Site.objects.get(site_name)
    BatchJob.objects.create(
        num_nodes=num_nodes,
        wall_time_min=wall_time_min,
        queue=queue,
        project=project,
        site_id=site.id,
        filter_tags={"job_start": f"{start}"},
        job_mode=job_mode
    )

if __name__=="__main__":

    # parent_job_ids = [job.id for job in Job.objects.filter(
    #     site_id=site.id, 
    #     tags={"workflow": f"{name}_convert2h5_{version}"},
    #     )]
    import argparse
    #no parents for systematics
    parser = argparse.ArgumentParser(description="Example argparse program")

    parser.add_argument("--c", type=int, default=0, help="Doc Num to start at divided by 5")

    args = parser.parse_args()
    #Creating Parent Job
    if "embed" in runs or "all" in runs:

        site = Site.objects.get(site_name)
        c=args.c
        tmp_parent_job_ids =[]
        for start in range(num_jobs // num_nodes):
            parent_job_ids = tmp_parent_job_ids
            tmp_parent_job_ids = []
            for i in range(args.c, args.c + num_nodes):
                params={'data_dir':i , 'ddb_start': doc_limit*c, 'ind_start': indico_limit*c}
                parent_job = []
                if parent_job_ids:
                    parent_job = [parent_job_ids[i]]
                parent = create_single_dependent_job("embed", c, parent_job, params, gpu_packing_count, start=start)
                print(f"Created job starting at dddb doc {doc_limit*c} and indico doc: {indico_limit*c}")
                print(f"Created job w/ id {parent.id} and parent is {parent_job}")
                tmp_parent_job_ids.append(parent.id)
                c += 1
        print(f"Created {num_jobs-1} jobs with start= {start}")
        print("embed jobs created")


#submit cpu jobs
    if "submit_all" in runs:
        # submit_all_jobs(num_nodes = spill_size//cpu_packing_count)
        # submit_all_jobs(num_nodes = min(spill_size//cpu_packing_count,max_nodes), wall_time_min = 60)
        #for start in range(begin_index, end_index, num_nodes):
        for start in range(num_jobs//num_nodes):
            submit_all_jobs(start=start, num_nodes=num_nodes, wall_time_min=wall_time_min)
