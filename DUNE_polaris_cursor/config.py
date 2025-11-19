import os
from pathlib import Path
from dotenv import load_dotenv
# Base configuration
BASE_DIR = Path(__file__).parent.absolute()
load_dotenv() 
# API Configuration
MAX_VARIABLE_NUMBER=5461


ARGO_API_USERNAME = os.getenv("ARGO_API_USERNAME")
ARGO_API_KEY = os.getenv("ARGO_API_KEY")
ARGO_API_URL = os.getenv("ARGO_API_URL", "https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/")
FERMILAB_API_URL=os.getenv("FERMILAB_API_URL","")

QA_PATH = './benchmarking/QuestionAnswer/QA.csv'
#'./benchmarking/QuestionAnswer/Cleaned_questions2.csv'

# Database Configuration
DUNE_DOCDB_USERNAME = os.getenv("DUNE_DOCDB_USERNAME")
DUNE_DOCDB_PASSWORD = os.getenv("DUNE_DOCDB_PASSWORD")
DUNE_INDICO_ACCESS_KEY = os.getenv("DUNE_INDICO_ACCESS_KEY")
INDICO_API_TOKEN = os.getenv("INDICO_API_TOKEN", "secret-indico_token")

# Fermilab Authentication Configuration
FERMILAB_CLIENT_ID = os.getenv("FERMILAB_CLIENT_ID")
print(f"Got FERMILAB_CLIENT_ID={FERMILAB_CLIENT_ID}")
FERMILAB_CLIENT_SECRET = os.getenv("FERMILAB_CLIENT_SECRET")
FERMILAB_SESSION_SECRET = os.getenv("FERMILAB_SESSION_SECRET", "super-secret-session-key-change-in-production")
FERMILAB_SCOPE = os.getenv("FERMILAB_SCOPE", "openid email profile")
FERMILAB_REDIRECT_URI = os.getenv("FERMILAB_REDIRECT_URI", "http://127.0.0.1:8000/auth")
ENABLE_AUTHENTICATION = os.getenv("ENABLE_AUTHENTICATION", "true").lower() == "true"
FERMI_SSO_USERNAME = os.getenv("FERMI_SSO_USERNAME")
print(f"Got FERMI_SSO_USERNAME={FERMI_SSO_USERNAME}")

FERMI_SSO_PASSWORD = os.getenv("FERMI_SSO_PASSWORD")

# Document limits for processing
DOC_LIMIT_DOCDB = int(os.getenv("DOC_LIMIT_DOCDB", "50"))
DOC_LIMIT_INDICO = int(os.getenv("DOC_LIMIT_INDICO", "50"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE",4000))

STORE = os.getenv("VECTOR_STORE", 'chroma')
# FAISS Configuration
CHROMA_PATH=os.getenv('DB_PATH',None)
FAISS_DIR = BASE_DIR / "data" / "faiss"
FAISS_INDEX_PATH = FAISS_DIR / "faiss_index.index"
METADATA_PATH = FAISS_DIR / "metadata_store.pkl"
DOC_IDS_PATH = FAISS_DIR / "doc_ids.pkl"

# Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "multi-qa-mpnet-base-dot-v1")
EMBEDDING_DIM = 768

# DocDB Configuration
DOCDB_BASE_URL = "https://docs.dunescience.org/cgi-bin/private/ShowDocument?docid="




# Indico Configuration
INDICO_BASE_URL = "https://indico.fnal.gov"
INDICO_CATEGORY_ID = 443
INDICO_COOKIES_FILE = "/Users/L00298625/.indico_cookies.json"
# Whether to bootstrap cookies with a real browser (Playwright) on first run
INDICO_USE_BROWSER_LOGIN = os.getenv("INDICO_USE_BROWSER_LOGIN", False)

# Which Playwright browser to use if INDICO_USE_BROWSER_LOGIN is true
INDICO_BROWSER = os.getenv("INDICO_BROWSER", "chromium")

# Application Configuration
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Query Configuration
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt4o")

# Validate required environment variables
def validate_config():
    """Validate that all required environment variables are set"""
    required_vars = [
        ("ARGO_API_USERNAME", ARGO_API_USERNAME),
        ("ARGO_API_KEY", ARGO_API_KEY),
        ("DUNE_DOCDB_USERNAME", DUNE_DOCDB_USERNAME),
        ("DUNE_DOCDB_PASSWORD", DUNE_DOCDB_PASSWORD),
    ]
    
    # Add Fermilab auth requirements if authentication is enabled
    if ENABLE_AUTHENTICATION:
        required_vars.extend([
            ("FERMILAB_CLIENT_ID", FERMILAB_CLIENT_ID),
            ("FERMILAB_CLIENT_SECRET", FERMILAB_CLIENT_SECRET),
        ])
    
    missing_vars = [var_name for var_name, var_value in required_vars if not var_value]
    
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

# Create necessary directories
def create_directories(path):
    """Create necessary directories if they don't exist"""
    FAISS_DIR=BASE_DIR / path
    FAISS_DIR.mkdir(parents=True, exist_ok=True) 
