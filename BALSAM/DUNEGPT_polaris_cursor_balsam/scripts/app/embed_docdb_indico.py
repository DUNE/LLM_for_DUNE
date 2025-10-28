from balsam.api import ApplicationDefinition, BatchJob, Job, Site
import yaml

'''
Create and store Embeddings
'''
yaml_file_path = '/lus/flare/projects/LLM_for_DUNE/users/Rishika/LLM_for_DUNE/BALSAM/DUNEGPT_polaris_cursor_balsam/specs/embed_docdb_indico/embed_docdb_indico.yaml'
site_name = "DUNEGPT_polaris_cursor_balsam"
 
path_to_python = "/lus/flare/projects/LLM_for_DUNE/users/Rishika/LLM_for_DUNE/BALSAM/DUNEGPT_polaris_cursor_balsam"
path_module = f"/lus/flare/projects/LLM_for_DUNE/users/Rishika/LLM_for_DUNE/DUNE_polaris_cursor"
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
        f"{path_to_python}/new_venv/bin/python3.11 cli.py index"
    )

    def shell_preamble(self):
        return f'''
        cd ~
        wget https://www.sqlite.org/2024/sqlite-autoconf-3450100.tar.gz
        tar xzf sqlite-autoconf-3450100.tar.gz
        cd sqlite-autoconf-3450100
        ./configure --prefix=$HOME/sqlite3
        make
        make install

        export CFLAGS="-I$HOME/sqlite3/include"
        export LDFLAGS="-L$HOME/sqlite3/lib"
        export LD_LIBRARY_PATH=$HOME/sqlite3/lib:$LD_LIBRARY_PATH


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
        ./new_venv/bin/pip install chromadb
        echo Installed
        ./new_venv/bin/pip install -r {path_module}/requirements.txt
        cd {path_module}
        {path_to_python}/new_venv/bin/python3.11 -m spacy download en_core_web_sm
        {path_to_python}/new_venv/bin/python3.11 update_sqlite.py --venv_path {path_to_python}/new_venv
        python {path_module}/config.py
        '''

embed.sync()
