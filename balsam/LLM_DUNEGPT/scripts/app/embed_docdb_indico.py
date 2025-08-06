from balsam.api import ApplicationDefinition, BatchJob, Job, Site
import yaml

'''
Create and store Embeddings
'''

yaml_file_path = '/lus/flare/projects/LLM_for_DUNE/users/aditya/BALSAM/LLM_DUNEGPT/specs/embed_docdb_indico/embed_docdb_indico.yaml'
site_name = "LLM_DUNEGPT"
path_module = f"/lus/flare/projects/LLM_for_DUNE/users/aditya/DUNEGPT_aurora/"

"""
Functions
"""

def get_env_from_yaml():
    with open(yaml_file_path, 'r') as file:
        yaml_content = file.read()
    yaml_dict = yaml.safe_load(yaml_content)
    env = yaml_dict['base_envs'][0]['env']
    return env


"""
embed
------------------------------------------
"""

env = get_env_from_yaml()
app_id = "embed_docdb_indico"

class embed(ApplicationDefinition):
    site = site_name
    print(env)
    environment_variables = env
    
    command_template = f"{path_module}/new_venv/bin/python cli.py index --docdb-limit 2000 --indico-limit 2000"
    def shell_preamble(self):
        return f'''
        export INDEX={self.job.data["i"]}
        module load autoconf cmake
        module load frameworks
        module load cmake
        module use /soft/modulefiles
        cd {path_module}
        source {path_module}/new_venv/bin/activate
        source {path_module}/init.sh 
        '''

embed.sync()
