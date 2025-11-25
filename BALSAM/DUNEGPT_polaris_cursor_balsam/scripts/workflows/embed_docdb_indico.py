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
num_jobs=10
num_nodes=9
begin_index = 0 #0
spill_size = num_nodes #4
end_index = num_jobs #4
wall_time_min = 360 #180
doc_limit = 3000 if queue == 'prod' else 50 #-------16K for indico 37K for ddb
def create_single_dependent_job(app_id, i, parent_ids, params, node_packing_count, start):
    params.update({"i": i})
    return Job.objects.create( app_id=app_id,
                        site_name=site_name,
                        workdir=f"workdir/{version}_{i}_{app_id}",
                        node_packing_count=node_packing_count,
                       tags={"workflow": f"{name}_{app_id}_{version}", "queue": f"{queue}", "index":f"{i}", "job_start":f"{start}", 'num_nodes': f"{num_nodes}", f"num_jobs_per_node": f"{spill_size}", 'document_limit' : f"{doc_limit}"},
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



if "embed" in runs or "all" in runs:

    site = Site.objects.get(site_name)

    # parent_job_ids = [job.id for job in Job.objects.filter(
    #     site_id=site.id, 
    #     tags={"workflow": f"{name}_convert2h5_{version}"},
    #     )]

    #no parents for systematics
    
    #Creating Parent Job
    c=0
    params={'ddb_start': doc_limit*c, 'ind_start': doc_limit*c}
    start = 0
    initial_job = create_single_dependent_job("embed", c, [], params, gpu_packing_count, start=start)
    parent_job_ids = [initial_job.id]

    #Creating Respective Child Jobs
    for _ in range(num_jobs - 1):
        c += 1
        params={'ddb_start': doc_limit*c, 'ind_start': doc_limit*c}
        create_single_dependent_job("embed", c, parent_job_ids, params, gpu_packing_count, start=start)
    print(f"Created {num_jobs-1} jobs with start= {start}")
    print("embed jobs created")


#submit cpu jobs
if "submit_all" in runs:
    # submit_all_jobs(num_nodes = spill_size//cpu_packing_count)
    # submit_all_jobs(num_nodes = min(spill_size//cpu_packing_count,max_nodes), wall_time_min = 60)
    #for start in range(begin_index, end_index, num_nodes):
    submit_all_jobs(start=start, num_nodes=num_nodes, wall_time_min=wall_time_min)
