from balsam.api import models, BatchJob, Job, EventLog
from balsam.analytics import utilization_report, available_nodes
from matplotlib import pyplot as plt
from collections import defaultdict
from datetime import datetime

# Setup
site_name = "LLM_DUNEGPT"
app_name = "embed"
workflow_name = "embed_docdb_indico"
version = "v0"

workflow_tag = f"{workflow_name}_{app_name}_{version}"

# Fetch App
app = models.App.objects.get(site_name=site_name, name=app_name)

# Fetch all jobs in this workflow (not just JOB_FINISHED)
jobs = Job.objects.filter(app_id=app.id, tags={"workflow": workflow_tag})

if not jobs:
    print(f"No jobs found for workflow tag: {workflow_tag}")
    exit()

job_ids = [job.id for job in jobs]

# Fetch EventLogs for those jobs
events = EventLog.objects.filter(job_id=job_ids)

if not events:
    print("No event logs found.")
    exit()

# Print state transitions per job
print("\nState transitions:")
job_times = defaultdict(dict)
for ev in sorted(events, key=lambda e: (e.job_id, e.timestamp)):
    print(f"Job {ev.job_id}: {ev.from_state} → {ev.to_state} @ {ev.timestamp}")
    if ev.to_state == "RUNNING":
        job_times[ev.job_id]['start'] = ev.timestamp
    elif ev.to_state == "RUN_DONE":
        job_times[ev.job_id]['end'] = ev.timestamp

# Compute and print job durations
print("\nJob run durations:")
for job_id, times in job_times.items():
    if 'start' in times and 'end' in times:
        duration = (times['end'] - times['start']).total_seconds() / 60
        print(f"Job {job_id} ran for {duration:.2f} minutes")
    else:
        print(f"Job {job_id} missing start or end time")

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

# Plot if data available
if elapsed_minutes and elapsed_minutes1:
    plt.figure(figsize=(10, 5))
    plt.plot(elapsed_minutes, node_counts, color='grey', linewidth=2, alpha=0.5, label='Node Count')
    util_scaled = [u * max(node_counts) for u in util]
    plt.step(elapsed_minutes1, util_scaled, where="post", label='Utilization')

    plt.xlabel('Elapsed Time (minutes)')
    plt.ylabel('Utilization / Node Count')
    plt.title(f"Balsam Job Utilization: {workflow_tag}")
    plt.legend()
    out_path = f"summary_{workflow_tag}.png"
    plt.savefig(out_path)
    print(f"\n✅ Plot saved to: {out_path}")
else:
    print("\n⚠️ Skipping plot due to insufficient data.")

