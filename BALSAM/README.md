This folder is used to initiate parallel BALSAM jobs.


**To Connect the Jobs to the Document Extraction Code:**

- cd into the scripts directory

        cd app
        python3 embed_docdb_indico.py
- This command sets the command the job needs to run and exports all the necessary variables to the environment. Specifically, it reads the variables that are defined within the jobs with respect to which docdb/indico document to start from and end at and exports these to the environment. When the jobs are initiated and the respective command (cli.py) is run, the code in cli.py will read these recently exported variables and process the documents with respect to these variabled.
    
For example, if app/embed_dodb_indico.py specifies:

        DDB_DOCUMENT_LIMIT = 50
        DDB_START_IDX = 0
        IND_DOCUMENT_LIMIT = 50
        IND_START_IDX = 0

- cli.py will start the document extraction process to process docdb document id's 0 to 49, and will read indico categories 0 to 49.

**To Start the Jobs:**
```
cd ../workflows 
python3 embed_docdb_indico.py --c 0
```
- This command initiates the jobs by creating 5 jobs running on 5 nodes. Whether they are 5 different nodes or some shared nodes depends on the job scheduler. The exact number of jobs and nodes can be adjusted by modifying:
    
    - variables:
        - num_jobs
        - num_nodes

- It also logs the document limit (number of documents/categories to read from docdb/indico respectively) and the starting document/category numbers so that the former script can read it. To adjust these modify:
    
    - variables: 
        - doc_limit 
        - indico_limit

**How to Decide Which Commands To Run**

The argument c is used to tell the script which document number to start with. If the intent is to start at document 20,000 in docdb 
    
    --c 20  # as one job is extracting 1000 docdb documents 
To decide the value of c:

    D = docdb document to start from
    L_D = number of docdb documents 1 job is extracting, or defined as "doc_limit" in "workflows/embed_docdb_indico.py"
    c = D / L_D
    
    OR
    
    I = indico category to start from
    L_I = number of indico documents 1 job is extracting, or defined as "indico_limit" in "workflows/embed_docdb_indico.py"
    c = I / L_I
    
The 'c value' is the same for both docdb and indico, so if:
    
    c = 20
    L_D = 1,000
    L_I = 1

c = 20 will start from docdb document 20,000 and indico category 20.

**Indico Category Clear Definition**:

- We are only reading the high level category 443 but within that category there are 42 subcategories. Indico numbers each category uniquely but this script (to initiate the jobs and read the documents) parses categories by indexing as 0,1,2,...41. 
- So the above defined "category 20" means the 20th category in the list of categories which our document extraction maps to the respective ID defined in indico.



