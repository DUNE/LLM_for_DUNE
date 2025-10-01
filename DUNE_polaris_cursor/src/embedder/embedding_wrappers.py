from chATLAS_Embed.EmbeddingModels import SentenceTransformerEmbedding
from src.utils.logger import get_logger
from sentence_transformers import SentenceTransformer
logger = get_logger(__name__)
from config import EMBEDDING_MODEL
from chromadb import Documents, EmbeddingFunction, Embeddings
 

class ChatlasEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformerEmbedding(model_name)
        self.model_name = model_name  # Required by Chroma

    def __call__(self, input):
        print(type(input))
        return self.model.embed(input)
    
    def embed_query(self, input):
        # You can just delegate to your existing embed method
        import numpy as np
        return self.model.embed(input)
    def embed_documents(self, input):
        # You can just delegate to your existing embed method
        import numpy as np
        return self.model.embed(input)
    
    def name(self):
        return self.model_name
    
class OriginalEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device='cpu'):
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name  # Required by Chroma

    def __call__(self, input) -> Embeddings:
        return self.model.encode(input).tolist()
    
    def name(self):
        return self.model_name
    def embed_query(self, input):
        # You can just delegate to your existing embed method
    
        return self.model.encode(input)
    def embed_documents(self, input):
        # You can just delegate to your existing embed method
        if not isinstance(input,list):
            raise ValueError(f"Input must be of type list, recieved {type(input)}")
    
        return self.model.encode(input)
