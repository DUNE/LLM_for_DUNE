from balsam.api import Job
job = Job.objects.get(42568374)
print(job.tags)

