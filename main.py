#!/usr/bin/env python3
"""
DUNE-GPT: A RAG-based LLM application for DUNE scientific documentation
"""

from fastapi import FastAPI, Request, Form, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
from typing import Optional, Dict, Any

from config import (
    HOST, PORT, DEBUG, ARGO_API_USERNAME, ARGO_API_KEY, 
    DEFAULT_TOP_K, ENABLE_AUTHENTICATION, FERMILAB_REDIRECT_URI, 
    validate_config, create_directories
)
from config import FERMILAB_SESSION_SECRET
from src.indexing.faiss_manager_reindexed import FAISSManager
from src.api.argo_client import ArgoAPIClient
from src.auth.fermilab_auth import fermilab_auth
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global variables
faiss_manager: Optional[FAISSManager] = None
argo_client: Optional[ArgoAPIClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global faiss_manager, argo_client
    
    # Startup
    logger.info("Starting DUNE-GPT application")
    
    # Validate configuration
    validate_config()
    create_directories()
    
    # Initialize components
    faiss_manager = FAISSManager()
    argo_client = ArgoAPIClient(ARGO_API_USERNAME, ARGO_API_KEY)
    
    # Check if index is empty
    stats = faiss_manager.get_stats()
    if stats["total_documents"] == 0:
        logger.warning("FAISS index is empty. Run the indexing process first.")
    else:
        logger.info(f"FAISS index loaded with {stats['total_documents']} documents")
    
    yield
    
    # Shutdown
    logger.info("Shutting down DUNE-GPT application")
    if faiss_manager:
        faiss_manager.cleanup()

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
        
        # Search FAISS index
        context_snippets, references = faiss_manager.search(question, top_k=DEFAULT_TOP_K)
        context = "\n\n".join(context_snippets)
        
        # Get answer from Argo API
        answer = argo_client.chat_completion(question, context)
        print(answer)
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "question": question,
            "answer": answer,
            "reference": references,
            "user": user,
            "auth_enabled": ENABLE_AUTHENTICATION
        })
    
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
        context_snippets, references = faiss_manager.search(q, top_k=top_k)
        context = "\n\n".join(context_snippets)
        
        # Get answer from Argo API
        answer = argo_client.chat_completion(q, context)
        
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
        stats = faiss_manager.get_stats()
        argo_healthy = argo_client.health_check()
        
        return {
            "status": "healthy" if argo_healthy else "degraded",
            "faiss_index": stats,
            "argo_api": "available" if argo_healthy else "unavailable"
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
        return faiss_manager.get_stats()
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main entry point"""
    logger.info(f"Starting DUNE-GPT server on {HOST}:{PORT}")
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info" if not DEBUG else "debug"
    )

if __name__ == "__main__":
    main() 