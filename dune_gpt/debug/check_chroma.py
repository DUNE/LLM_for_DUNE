import chromadb

from src.indexing.chroma_manager import ChromaManager


path='discard_2000'
chroma = ChromaManager(path)

metadatas = chroma.chroma_collection.get(include=['uris', 'documents'])
for i, md in zip(metadatas['ids'],metadatas['documents']):
    if '12522' in i: #if '10438' in i:
       print(md)
