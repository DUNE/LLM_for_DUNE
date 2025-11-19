from balsam.api import models, BatchJob, Job, EventLog
from balsam.analytics import utilization_report, available_nodes
from matplotlib import pyplot as plt
from collections import defaultdict
from datetime import datetime
# Setup
site_name = "DUNEGPT_polaris_cursor_balsam"
app_name = "embed"
workflow_name = "embed_docdb_indico"
version = "v0"

file="job_analytics.json"

workflow_tag = f"{workflow_name}_{app_name}_{version}"



# Fetch App
app = models.App.objects.get(site_name=site_name, name=app_name)

# Fetch all jobs in this workflow (not just JOB_FINISHED)
jobs = Job.objects.filter(app_id=app.id, tags={"workflow": workflow_tag})

if not jobs:
    print(f"No jobs found for workflow tag: {workflow_tag}")
    exit()

job_ids = [job.id for job in jobs]
print(F"Number of jobs: {len(job_ids)}")
# Fetch EventLogs for those jobs
events = EventLog.objects.filter(job_id=job_ids)

if not events:
    print("No event logs found.")
    exit()


config={job.id: job.tags for job in jobs}

# Print state transitions per job and extract start/end times
print("\nState transitions:")
job_times = defaultdict(dict)
for ev in sorted(events, key=lambda e: (e.job_id, e.timestamp)):
    #print(f"Job {ev.job_id}: {ev.from_state} → {ev.to_state} @ {ev.timestamp}")
    if ev.to_state == 'STAGED_IN':
        print(f"{ev.job_id} staged at {ev.timestamp}")
    if ev.to_state == "RUNNING":
        job_times[ev.job_id]['start'] = ev.timestamp
        print(f"{ev.job_id} started running at {ev.timestamp}")
    elif ev.to_state == "RUN_DONE":
        job_times[ev.job_id]['end'] = ev.timestamp
        print(f"{ev.job_id} finished running at {ev.timestamp}")

# Compute durations and extract doc counts
job_labels = []
docs_processed = []
durations = []
start_times=[]
end_times=[]
print("\nJob run durations:")
for job in jobs:
    print(f"Tags: {job.tags}")
    job_id = job.id
    job_labels.append(str(job_id))

    doc_count = int(job.tags.get("document_limit", 0))  # fallback to 0
    docs_processed.append(doc_count)

    times = job_times.get(job_id, {})
    
    if 'start' in times and 'end' in times:
        duration = (times['end'] - times['start']).total_seconds() / 60
        durations.append(duration)
        print(f"Job {job_id} ran for {duration:.2f} minutes")

        start_times.append(times['start'])
        end_times.append(times['end'])
    else:
        durations.append(0)
        print(f"Job {job_id} missing start or end time")

#Record duration of run
start_times.sort()
end_times.sort()
print(f"Took {end_times[-1] - start_times[0]} to extract {sum(docs_processed)} events")
# Plot throughput (docs/minute)
throughput = [d / t if t > 0 else 0 for d, t in zip(docs_processed, durations)]
print(f"Throughput for {job_ids[0]} = {throughput}")
plt.figure(figsize=(8, 5))
bars = plt.bar(job_labels, throughput, color='skyblue')
plt.ylabel("Total Docs per Minute")
plt.xlabel("Job ID")
plt.title("Embedding Throughput per Job")

def save_to_json(job_data, dest):
    import json
    try:
        # Read the existing data from the JSON file
        with open(dest, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        # If the file doesn't exist, start with an empty list
        data = []
    except json.JSONDecodeError:
        # Handle cases where the file might be empty or malformed
        print(f"Warning: JSON file '{dest}' is empty or malformed. Initializing with an empty list.")
        data = []

    # Ensure the top-level element is a list for appending
    if not isinstance(data, list):
        print(f"Error: JSON file '{dest}' does not contain a list as its top-level element. Cannot append.")
        return

    # Append the new data to the list
    data.append(job_data)

    # Write the updated data back to the JSON file
    with open(dest, 'w') as f:
        json.dump(job_data, f, indent=4) # indent for pretty-printing


#update config for this run and save to json
for i, job in enumerate(jobs):
    config[job.id]['docdb_throughput'] = throughput[i]
    config[job.id]['job_duration'] = durations[i]
    config[job.id]['docs_processed_docdb'] = docs_processed[i]
save_to_json(config, file)


for bar, val in zip(bars, throughput):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.05, f"{val:.2f}", ha='center')

plt.tight_layout()
output_file = f"throughput_{workflow_tag}.png"
plt.savefig(output_file)
print(f"\n✅ Saved custom throughput plot to: {output_file}")

# Plot durations
plt.figure(figsize=(8, 5))
bars = plt.bar(job_labels, durations, color='orange')
plt.ylabel("Run Time (minutes)")
plt.xlabel("Job ID")
plt.title("Job Duration per Job")

for bar, val in zip(bars, durations):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f} min", ha='center')

plt.tight_layout()
output_file2 = f"durations_{workflow_tag}.png"
plt.savefig(output_file2)
print(f"✅ Saved duration plot to: {output_file2}")

# Try utilization report
try:
    times, util = utilization_report(events, node_weighting=True)
    if times:
        t0 = min(times)
        elapsed_minutes1 = [(t - t0).total_seconds() / 60 for t in times]
    else:
        elapsed_minutes1 = []
except Exception as e:
    print(f"Utilization report failed: {e}")
    elapsed_minutes1 = []
    util = []

# Try node count report
try:
    batch_jobs = BatchJob.objects.filter(filter_tags={"workflow": workflow_tag})
    times_nodes, node_counts = available_nodes(batch_jobs)
    if times_nodes:
        t0 = min(times_nodes)
        elapsed_minutes = [(t - t0).total_seconds() / 60 for t in times_nodes]
    else:
        elapsed_minutes = []
except Exception as e:
    print(f"Node count report failed: {e}")
    elapsed_minutes = []
    node_counts = []

print(f"len(times): {len(times)}, len(util): {len(util)}")
print(f"len(times_nodes): {len(times_nodes)}, len(node_counts): {len(node_counts)}")

# Plot utilization
if elapsed_minutes1:
    plt.figure(figsize=(10, 5))
    plt.step(elapsed_minutes1, util, where="post", label='Utilization')
    plt.xlabel('Elapsed Time (minutes)')
    plt.ylabel('Utilization')
    plt.title(f"Balsam Job Utilization: {workflow_tag}")
    plt.legend()
    out_path = f"summary_{workflow_tag}_util_only.png"
    plt.savefig(out_path)
    print(f"\n✅ Utilization-only plot saved to: {out_path}")
else:
    print("\n  Skipping utilization plot due to insufficient data.")

