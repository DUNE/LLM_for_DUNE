from balsam.api import ApplicationDefinition, BatchJob, Job, Site
import yaml

'''
Create and store Embeddings
'''

yaml_file_path = '/lus/flare/projects/LLM_for_DUNE/users/Rishika/BALSAM/DUNEGPT_polaris_cursor_balsam/specs/embed_docdb_indico/embed_docdb_indico.yaml'
site_name = "DUNEGPT_polaris_cursor_balsam"
 
path_to_python = f"/lus/flare/projects/LLM_for_DUNE/users/Rishika/BALSAM/"
path_module = f"/lus/flare/projects/LLM_for_DUNE/users/Rishika/DUNEGPT_polaris_cursor"

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
    
    command_template = (
        f"{path_to_python}/new_venv/bin/python3 cli.py index --indico-limit 0 "
    )


    def shell_preamble(self):
        return f'''
        export INDEX={self.job.data["i"]}
        export DOCUMENT_LIMIT={self.job.tags["document_limit"]}
        export DDB_START_IDX={self.job.data["ddb_start"]}
        export IND_START_IDX={self.job.data["ind_start"]}
        module load autoconf cmake
        module load frameworks
        module load cmake
        module use /soft/modulefiles
        cd {path_to_python}
        source {path_to_python}/new_venv/bin/activate
        cd {path_module}
        python3 -m spacy download en_core_web_sm
        python3 {path_module}/config.py

        '''

embed.sync()
