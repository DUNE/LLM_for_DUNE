"""
Fermilab OAuth2 Authentication Module for DUNE-GPT
"""

import os
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
from config import (
    FERMILAB_CLIENT_ID, FERMILAB_CLIENT_SECRET, 
    FERMILAB_SESSION_SECRET, FERMILAB_SCOPE
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FermilabAuth:
    """Fermilab OAuth2 authentication handler"""
    
    def __init__(self):
        self.oauth = OAuth()
        self._setup_oauth()
    
    def _setup_oauth(self):
        """Setup OAuth2 configuration for Fermilab PingFederate"""
        if not FERMILAB_CLIENT_ID or not FERMILAB_CLIENT_SECRET:
            raise ValueError("Fermilab OAuth2 credentials not configured")
        
        self.oauth.register(
            name="fermilab",
            client_id=FERMILAB_CLIENT_ID,
            client_secret=FERMILAB_CLIENT_SECRET,
            access_token_url="https://pingprod.fnal.gov/as/token.oauth2",
            authorize_url="https://pingprod.fnal.gov/as/authorization.oauth2",
            api_base_url="https://pingprod.fnal.gov",
            client_kwargs={
                "scope": FERMILAB_SCOPE,
            },
            server_metadata_url="https://pingprod.fnal.gov/.well-known/openid-configuration",
        )
        
        logger.info("Fermilab OAuth2 configured successfully")
    
    def get_session_middleware(self) -> SessionMiddleware:
        """Get session middleware for FastAPI app"""
        return SessionMiddleware(secret_key=FERMILAB_SESSION_SECRET)
    
    async def get_current_user(self, request: Request) -> Dict[str, Any]:
        """Get current authenticated user from session"""
        user = request.session.get("user")
        if not user:
            logger.warning("Unauthenticated access attempt")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Authentication required"
            )
        return user
    
    async def get_current_user_optional(self, request: Request) -> Optional[Dict[str, Any]]:
        """Get current user if authenticated, None otherwise"""
        return request.session.get("user")
    
    async def authorize_redirect(self, request: Request, redirect_uri: str):
        """Initiate OAuth2 authorization flow"""
        logger.info(f"Starting OAuth2 flow with redirect URI: {redirect_uri}")
        return await self.oauth.fermilab.authorize_redirect(request, redirect_uri)
    
    async def handle_callback(self, request: Request) -> Dict[str, Any]:
        """Handle OAuth2 callback and get user info"""
        try:
            # Get access token
            token = await self.oauth.fermilab.authorize_access_token(request)
            
            # Get user info
            user_info = await self.oauth.fermilab.userinfo(token=token)
            user_dict = dict(user_info)
            
            # Store in session
            request.session["user"] = user_dict
            
            logger.info(f"User authenticated: {user_dict.get('email', 'unknown')}")
            return user_dict
            
        except Exception as e:
            logger.error(f"OAuth2 callback error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Authentication failed"
            )
    
    def logout_user(self, request: Request):
        """Clear user session"""
        user_email = request.session.get("user", {}).get("email", "unknown")
        request.session.clear()
        logger.info(f"User logged out: {user_email}")
    
    def is_user_authorized(self, user: Dict[str, Any], required_groups: list = None) -> bool:
        """Check if user is authorized (can be extended for group-based auth)"""
        if not user:
            return False
        
        # Basic check - user has email
        if not user.get("email"):
            return False
        
        # TODO: Add group-based authorization if needed
        # if required_groups:
        #     user_groups = user.get("groups", [])
        #     return any(group in user_groups for group in required_groups)
        
        return True

# Global auth instance
fermilab_auth = FermilabAuth() 