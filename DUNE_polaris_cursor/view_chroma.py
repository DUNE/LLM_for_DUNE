import chromadb
from chromadb.config import Settings
collection_name = "DUNE_VECTOR_DB"  # replace with your actual collection name
merged_dir='data'
# Initialize merged Chroma client and collection
merged_client = chromadb.PersistentClient(path=merged_dir, settings=Settings())
merged_collection = merged_client.get_or_create_collection(name=collection_name)
# Loop through each instance and merge

mcd = merged_collection.get(include=['documents', 'metadatas', 'uris'])
all_ids=set()
max_id=0
min_id=float("inf")
avg_words = 0
max_len=0
for doc, id_, md in zip(mcd['documents'], mcd['ids'], mcd['metadatas']):
    i = id_.split('_')[0]
    if '/' in i:
        i = i.split('/')[0]

    if md.get('source', 'indico') == 'docdb':
        min_id = min(min_id, int(i))
        max_id = max(max_id, int(i))
        avg_words += len(doc.split())
        max_len = max(len(doc.split()), max_len)
    else:
        print(i)
    all_ids.add(i)
    if md.get('source', 'indico') == 'docdb':
        min_id = min(min_id, int(i))
        max_id = max(max_id, int(i))
    avg_words += len(doc.split())
    max_len = max(len(doc.split()), max_len)

print(f"Avg length of chunk ", avg_words/len(mcd['documents']))
print(f"Number of events accessed ", len(all_ids))
print(f"Max number of words per chunk : {max_len}, Latest doc stored form DDB {max_id}, First doc stored from DDB: {min_id}")

