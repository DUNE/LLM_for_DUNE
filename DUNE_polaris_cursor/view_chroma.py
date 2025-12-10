import chromadb
from chromadb.config import Settings
import pandas as pd 
collection_name = "DUNE_VECTOR_DB"  # replace with your actual collection name
merged_dir='merged_db'

# Initialize merged Chroma client and collection
merged_client = chromadb.PersistentClient(path=merged_dir, settings=Settings())
merged_collection = merged_client.get_or_create_collection(name=collection_name)

# Loop through each instance and merge
print("Initiated collection")
mcd = merged_collection.get(include=['documents', 'metadatas', 'uris'])
max_id=0
min_id=float("inf")
avg_words = 0
max_len=-1
ids={'indico':set(), 'docdb':set()}
for doc, id_, md in zip(mcd['documents'], mcd['ids'], mcd['metadatas']):
    i = id_.split('_')[0]
    if '/' in i:
        i = i.split('/')[0]

    if md.get('source', 'indico') == 'docdb':
        min_id = min(min_id, int(i))
        max_id = max(max_id, int(i))
        avg_words += len(doc.split())
        max_len = max(len(doc.split()), max_len)
    ids[md.get('source', 'indico')].add(i)
    avg_words += len(doc.split())


#Store id's in a DB
max_len = max(len(ids['indico']), len(ids['docdb']))
if len(ids['indico']) < max_len:
    for i in range(max_len - len(ids['indico'])):
        ids['indico'].append('')
else:
    for i in range(max_len - len(ids['docdb'])):
        ids['docdb'].append('')

pd.DataFrame(ids).to_csv("allids.csv")

print(f"Avg length of chunk ", avg_words/len(mcd['documents']))
print(f"Number of events accessed ", len(ids['indico']) + len(ids['docdb']))
print(f"Max number of words per chunk : {max_len}, Latest doc stored form DDB {max_id}, First doc stored from DDB: {min_id}")

