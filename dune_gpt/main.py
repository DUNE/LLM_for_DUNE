#!/usr/bin/env python3.11
"""
DUNE-GPT: A RAG-based LLM application for DUNE scientific documentation
"""
import fastapi
from fastapi import FastAPI, Request, Form, HTTPException, Depends, status
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
from typing import Optional, Dict, Any

from config import (
    HOST, PORT, DEBUG, ARGO_API_USERNAME, ARGO_API_KEY,
    DEFAULT_TOP_K, ENABLE_AUTHENTICATION, FERMILAB_REDIRECT_URI,
    STORE, validate_config, create_directories, CHROMA_PATH, FERMILAB_SESSION_SECRET, K_DOCS,
    LLM_PROVIDER
)

if STORE == 'faiss':
    from src.indexing.faiss_manager_langchain import FAISSManager
    db_manager: Optional[FAISSManager] = None
elif STORE == 'chroma':
    from src.indexing.chroma_manager import ChromaManager 
    db_manager: Optional[ChromaManager] = None
else:
    raise Exception(f"DUNE-GPT requires Faiss or Chroma. Got {STORE}")
from src.api.fermilab_client import FermilabAPIClient
from src.api.argo_client import ArgoAPIClient
from src.auth.fermilab_auth import fermilab_auth
from src.utils.logger import get_logger
print("ALL IMPORTS IMPORTED")
logger = get_logger(__name__)

# Global variables
llm_client = None  # FermilabAPIClient or ArgoAPIClient

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global db_manager, llm_client
    
    # Startup
    logger.info("Starting DUNE-GPT application")
    
    # Validate configuration
    validate_config()
    create_directories(CHROMA_PATH)
    
    # Initialize components
    if STORE == 'faiss':
        db_manager = FAISSManager(CHROMA_PATH)
    elif STORE == 'chroma':
        db_manager = ChromaManager(CHROMA_PATH)
    else:
        raise Exception(f"Requires Faiss or Chroma. Got {STORE}")
    
    if LLM_PROVIDER == "argo":
        logger.info("Using Argo API client (LLM_PROVIDER=argo)")
        if not ARGO_API_USERNAME or not ARGO_API_KEY:
            raise RuntimeError("LLM_PROVIDER=argo but ARGO_API_USERNAME/ARGO_API_KEY not set")
        llm_client = ArgoAPIClient(ARGO_API_USERNAME, ARGO_API_KEY)
    else:
        logger.info("Using Fermilab API client (LLM_PROVIDER=fermilab)")
        llm_client = FermilabAPIClient()
    
    # Check if index is empty
    stats = db_manager.get_stats()
    if stats["total_documents"] == 0:
        logger.warning("Vector store index is empty. Run the indexing process first.")
    else:
        logger.info(f"Vector store index loaded with {stats['total_documents']} documents")
    
    yield
    
    # Shutdown
    logger.info("Shutting down DUNE-GPT application")
    if db_manager:
        db_manager.cleanup()

# Create FastAPI app
app = FastAPI(
    title="DUNE-GPT",
    description="A RAG-based LLM application for DUNE scientific documentation",
    version="1.0.0",
    lifespan=lifespan
)

# Add session middleware if authentication is enabled
if ENABLE_AUTHENTICATION:
    from starlette.middleware.sessions import SessionMiddleware
    #app.add_middleware(SessionMiddleware, secret_key=fermilab_auth.get_session_secret())
    app.add_middleware(SessionMiddleware, secret_key=FERMILAB_SESSION_SECRET)
    logger.info("Authentication enabled - added session middleware")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Authentication helper functions
async def get_user_if_auth_enabled(request: Request) -> Optional[Dict[str, Any]]:
    """Get user if authentication is enabled, None otherwise"""
    if not ENABLE_AUTHENTICATION:
        return None
    return await fermilab_auth.get_current_user_optional(request)

async def require_auth_if_enabled(request: Request) -> Optional[Dict[str, Any]]:
    """Require authentication if enabled"""
    if not ENABLE_AUTHENTICATION:
        return None
    return await fermilab_auth.get_current_user(request)

# Exception handler for authentication redirects
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == status.HTTP_401_UNAUTHORIZED and ENABLE_AUTHENTICATION:
        return RedirectResponse(url="/login")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

# Authentication routes (only if enabled)
if ENABLE_AUTHENTICATION:
    @app.get("/login")
    async def login(request: Request):
        """Initiate Fermilab OAuth2 login"""
        logger.info(f"OAuth2 redirect URI: {FERMILAB_REDIRECT_URI}")
        return await fermilab_auth.authorize_redirect(request, FERMILAB_REDIRECT_URI)
    
    @app.get("/auth", name="auth_callback")
    async def auth_callback(request: Request):
        """Handle OAuth2 callback"""
        logger.info("Processing OAuth2 callback")
        await fermilab_auth.handle_callback(request)
        return RedirectResponse(url="/ui")
    
    @app.get("/logout")
    async def logout(request: Request):
        """Logout user"""
        fermilab_auth.logout_user(request)
        return RedirectResponse(url="/ui")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request, user: Optional[Dict[str, Any]] = Depends(get_user_if_auth_enabled)):
    """Redirect to main UI"""
    return RedirectResponse(url="/ui")

@app.get("/ui", response_class=HTMLResponse)
async def form_get(request: Request, user: Optional[Dict[str, Any]] = Depends(require_auth_if_enabled)):
    """Display the main UI form"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "question": "",
        "answer": None,
        "reference": [],
        "user": user,
        "auth_enabled": ENABLE_AUTHENTICATION
    })

@app.post("/ui", response_class=HTMLResponse)
async def form_post(request: Request, question: str = Form(...), user: Optional[Dict[str, Any]] = Depends(require_auth_if_enabled)):
    """Handle form submission and return results"""
    try:
        logger.info(f"Question received from {user.get('email', 'anonymous') if user else 'anonymous'}: {question}")
        # Search Chroma index
        context_snippets, references = db_manager.search(question, k_docs= K_DOCS, top_k=DEFAULT_TOP_K)
        context = "\n\n".join(context_snippets)
        
        # Get answer from Fermilab API
        return StreamingResponse(llm_client.chat_completion(question, context, links=references), media_type="text/html")

    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "question": question,
            "answer": f"Error processing your question: {str(e)}",
            "reference": [],
            "user": user,
            "auth_enabled": ENABLE_AUTHENTICATION
        })

@app.get("/api/search")
async def api_search(q: str, top_k: int = DEFAULT_TOP_K):
    """API endpoint for search queries"""
    try:
        if not q.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Search FAISS index
        context_snippets, references = db_manager.search(q, top_k=top_k)
        context = "\n\n".join(context_snippets)
        
        # Get answer from Fermilab API
        answer = llm_client.chat_completion(q, context)
        
        return {
            "question": q,
            "answer": answer,
            "references": references,
            "context_snippets": len(context_snippets)
        }
    
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        stats = db_manager.get_stats()
        api_healthy = llm_client.health_check()

        return {
            "status": "healthy" if api_healthy else "degraded",
            "db_index": stats,
            "llm_provider": LLM_PROVIDER,
            "llm_api": "available" if api_healthy else "unavailable",
        }
    
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/api/stats")
async def get_stats():
    """Get application statistics"""
    try:
        return db_manager.get_stats()
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main entry point"""
    logger.info(f"Starting DUNE-GPT server on {HOST}:{PORT}")
    print(f"Starting DUNE-GPT server on {HOST}:{PORT}")
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info" if not DEBUG else "debug"
    )

if __name__ == "__main__":
    print("Calling main")
    main() 
