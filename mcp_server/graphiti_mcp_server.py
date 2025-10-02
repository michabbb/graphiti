import asyncio
import base64
import hashlib
import html
import json
import logging
import os
import secrets
import time
import uuid
from urllib.parse import urlencode
from datetime import datetime, timedelta
from threading import RLock
from typing import AsyncGenerator, Dict, List, Optional

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security.utils import get_authorization_scheme_param
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graph import Graph
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.models.graph import Edge, Node
from graphiti_core.models.message import Message, MessageRole
from jose import JWTError, jwt
from mcp import MCP, DATETIME_FORMAT, Server, StreamEvent
from mcp.models.server import PublicServerInfo
from passlib.context import CryptContext
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from sse_starlette.sse import EventSourceResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Decrease this if you're experiencing 429 rate limit errors
# Increase if you have high rate limits.
SEMAPHORE_LIMIT = int(os.getenv('SEMAPHORE_LIMIT', 10))


# --- Security Configuration ---
# To generate a new secret key, you can use:
# import secrets
# secrets.token_hex(32)
MCP_SERVER_SECRET_KEY = os.environ.get(
    "MCP_SERVER_SECRET_KEY", "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
)
if MCP_SERVER_SECRET_KEY == "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7":
    logger.warning(
        "Using default secret key. Please set MCP_SERVER_SECRET_KEY environment"
        " variable for production."
    )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

ALLOWED_NONCE_TOKENS = [
    token.strip()
    for token in os.environ.get("MCP_SERVER_NONCE_TOKENS", "").split(",")
    if token.strip()
]
if ALLOWED_NONCE_TOKENS:
    logger.info(
        "Loaded %d MCP_SERVER_NONCE_TOKENS for query parameter authentication.",
        len(ALLOWED_NONCE_TOKENS),
    )

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _is_nonce_valid(candidate: str) -> bool:
    for token in ALLOWED_NONCE_TOKENS:
        if secrets.compare_digest(candidate, token):
            return True
    return False


class RegisteredClient(BaseModel):
    client_id: str
    client_secret_hash: Optional[str]
    client_name: str
    redirect_uris: List[str]
    grant_types: List[str]
    token_endpoint_auth_method: str
    scope: Optional[str] = None
    client_id_issued_at: int
    client_secret_expires_at: int = 0


CLIENT_REGISTRY: Dict[str, RegisteredClient] = {}
CLIENT_REGISTRY_LOCK = RLock()


# --- Pydantic Models for API Responses ---
class ErrorResponse(BaseModel):
    error: str


class SuccessResponse(BaseModel):
    success: bool
    message: str = "Operation completed successfully"


class FactSearchResponse(BaseModel):
    facts: List[Dict]


class EpisodeSearchResponse(BaseModel):
    episodes: List[Dict]


class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None


class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    scope: str = ""
    refresh_token: Optional[str] = None


class ClientRegistrationRequest(BaseModel):
    client_name: str
    redirect_uris: List[str]
    grant_types: List[str] = Field(
        default_factory=lambda: ["authorization_code", "refresh_token"]
    )
    token_endpoint_auth_method: str = Field(default="client_secret_basic")
    scope: Optional[str] = None

    @field_validator("redirect_uris")
    def validate_redirect_uris(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("At least one redirect URI must be provided")
        return value

    @field_validator("grant_types")
    def validate_grant_types(cls, value: List[str]) -> List[str]:
        supported = {"client_credentials", "authorization_code", "refresh_token"}
        invalid = [grant for grant in value if grant not in supported]
        if invalid:
            raise ValueError(
                "Unsupported grant types requested: " + ", ".join(sorted(set(invalid)))
            )
        return sorted(set(value))

    @field_validator("token_endpoint_auth_method")
    def validate_auth_method(cls, value: str) -> str:
        supported = {"client_secret_basic", "client_secret_post", "none"}
        if value not in supported:
            raise ValueError(
                f"Unsupported token_endpoint_auth_method. Supported: {', '.join(sorted(supported))}"
            )
        return value

    @model_validator(mode="after")
    def validate_public_client(cls, values: "ClientRegistrationRequest") -> "ClientRegistrationRequest":
        if values.token_endpoint_auth_method == "none":
            disallowed = {
                grant
                for grant in values.grant_types
                if grant not in {"authorization_code", "refresh_token"}
            }
            if disallowed:
                raise ValueError(
                    "Public clients may only request authorization_code or refresh_token grants"
                )
        return values


class ClientRegistrationResponse(BaseModel):
    client_id: str
    client_secret: Optional[str] = None
    client_id_issued_at: int
    client_secret_expires_at: int
    client_name: str
    redirect_uris: List[str]
    grant_types: List[str]
    token_endpoint_auth_method: str
    scope: Optional[str] = None

    model_config = {
        "json_schema_extra": {"example": {"client_id": "...", "client_secret": "..."}},
        "use_enum_values": True,
    }


class AuthorizationRequestParams(BaseModel):
    client_id: str
    redirect_uri: str
    response_type: str = Field(default="code", pattern="^code$")
    scope: Optional[str] = ""
    state: Optional[str] = None
    code_challenge: Optional[str] = None
    code_challenge_method: Optional[str] = None

    @field_validator("response_type")
    def validate_response_type(cls, value: str) -> str:
        if value != "code":
            raise ValueError("Only authorization_code response_type is supported")
        return value

class AuthorizationCodeGrant(BaseModel):
    code: str
    client_id: str
    redirect_uri: str
    scope: str
    user_email: str
    code_challenge: Optional[str] = None
    code_challenge_method: Optional[str] = None
    expires_at: datetime


class RefreshTokenGrant(BaseModel):
    token: str
    client_id: str
    scope: str
    user_email: str
    expires_at: datetime


class AuthUser(BaseModel):
    email: str
    password_hash: str
    display_name: str


AUTHORIZATION_CODE_TTL_SECONDS = int(os.getenv("MCP_SERVER_AUTH_CODE_TTL", "300"))
REFRESH_TOKEN_TTL_SECONDS = int(os.getenv("MCP_SERVER_REFRESH_TOKEN_TTL", str(7 * 24 * 60 * 60)))

AUTHORIZATION_CODES: Dict[str, AuthorizationCodeGrant] = {}
AUTHORIZATION_CODES_LOCK = RLock()

REFRESH_TOKENS: Dict[str, RefreshTokenGrant] = {}
REFRESH_TOKENS_LOCK = RLock()

AUTH_USERS: Dict[str, AuthUser] = {}
AUTH_USERS_LOCK = RLock()

DEMO_USER_EMAIL = os.getenv("MCP_SERVER_DEMO_USER_EMAIL", "demo@example.com")
DEMO_USER_PASSWORD = os.getenv("MCP_SERVER_DEMO_USER_PASSWORD", "demo-password")
DEMO_USER_NAME = os.getenv("MCP_SERVER_DEMO_USER_NAME", "Demo User")
ALLOW_DEMO_USER = os.getenv("MCP_SERVER_ALLOW_DEMO_USER", "true").lower() not in {
    "0",
    "false",
    "no",
}


def _load_auth_users() -> None:
    raw_users = os.getenv("MCP_SERVER_AUTH_USERS", "")
    loaded = 0
    with AUTH_USERS_LOCK:
        AUTH_USERS.clear()
        if raw_users.strip():
            entries = [entry.strip() for entry in raw_users.split(",") if entry.strip()]
            for entry in entries:
                parts = entry.split(":", 2)
                if len(parts) < 2:
                    logger.warning(
                        "Skipping malformed MCP_SERVER_AUTH_USERS entry. Expected 'email:password[:display_name]'"
                    )
                    continue
                email = parts[0].strip().lower()
                password = parts[1]
                display_name = parts[2].strip() if len(parts) == 3 else parts[0].strip()
                if not email or not password:
                    logger.warning("Skipping MCP user with missing email or password")
                    continue
                AUTH_USERS[email] = AuthUser(
                    email=parts[0].strip(),
                    password_hash=pwd_context.hash(password),
                    display_name=display_name or parts[0].strip(),
                )
                loaded += 1
        if loaded == 0 and ALLOW_DEMO_USER:
            AUTH_USERS[DEMO_USER_EMAIL.lower()] = AuthUser(
                email=DEMO_USER_EMAIL,
                password_hash=pwd_context.hash(DEMO_USER_PASSWORD),
                display_name=DEMO_USER_NAME,
            )
            loaded = 1
            logger.info(
                "Demo MCP OAuth user enabled (email=%s). Override via MCP_SERVER_AUTH_USERS.",
                DEMO_USER_EMAIL,
            )
    logger.info("Loaded %d MCP OAuth user account(s)", loaded)


def _get_auth_user(email: str) -> Optional[AuthUser]:
    if not email:
        return None
    with AUTH_USERS_LOCK:
        return AUTH_USERS.get(email.lower())


def _authenticate_user(email: str, password: str) -> bool:
    user = _get_auth_user(email)
    if user is None:
        return False
    try:
        return pwd_context.verify(password, user.password_hash)
    except Exception:
        return False


def _get_demo_user() -> Optional[AuthUser]:
    if not ALLOW_DEMO_USER:
        return None
    return _get_auth_user(DEMO_USER_EMAIL)


_load_auth_users()


# --- Authentication Functions ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    now = datetime.utcnow()
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=15)
    to_encode.update({"exp": expire, "iat": now})
    encoded_jwt = jwt.encode(to_encode, MCP_SERVER_SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def _get_registered_client(client_id: str) -> Optional[RegisteredClient]:
    with CLIENT_REGISTRY_LOCK:
        return CLIENT_REGISTRY.get(client_id)


def _store_registered_client(client: RegisteredClient) -> None:
    with CLIENT_REGISTRY_LOCK:
        CLIENT_REGISTRY[client.client_id] = client


def _verify_client_secret(client: RegisteredClient, secret: str) -> bool:
    if not client.client_secret_hash:
        return False
    try:
        return pwd_context.verify(secret, client.client_secret_hash)
    except Exception:
        return False


def _client_requires_secret(client: RegisteredClient) -> bool:
    return client.token_endpoint_auth_method in {"client_secret_basic", "client_secret_post"}


def _store_authorization_code(grant: AuthorizationCodeGrant) -> None:
    with AUTHORIZATION_CODES_LOCK:
        AUTHORIZATION_CODES[grant.code] = grant


def _get_authorization_code(code: str) -> Optional[AuthorizationCodeGrant]:
    with AUTHORIZATION_CODES_LOCK:
        grant = AUTHORIZATION_CODES.get(code)
    if grant and grant.expires_at < datetime.utcnow():
        with AUTHORIZATION_CODES_LOCK:
            AUTHORIZATION_CODES.pop(code, None)
        return None
    return grant


def _consume_authorization_code(code: str) -> Optional[AuthorizationCodeGrant]:
    with AUTHORIZATION_CODES_LOCK:
        grant = AUTHORIZATION_CODES.pop(code, None)
    if grant and grant.expires_at < datetime.utcnow():
        return None
    return grant


def _store_refresh_token(grant: RefreshTokenGrant) -> None:
    with REFRESH_TOKENS_LOCK:
        REFRESH_TOKENS[grant.token] = grant


def _get_refresh_token(token: str) -> Optional[RefreshTokenGrant]:
    with REFRESH_TOKENS_LOCK:
        grant = REFRESH_TOKENS.get(token)
    if grant and grant.expires_at < datetime.utcnow():
        with REFRESH_TOKENS_LOCK:
            REFRESH_TOKENS.pop(token, None)
        return None
    return grant


def _consume_refresh_token(token: str) -> Optional[RefreshTokenGrant]:
    with REFRESH_TOKENS_LOCK:
        grant = REFRESH_TOKENS.pop(token, None)
    if grant and grant.expires_at < datetime.utcnow():
        return None
    return grant


def _verify_pkce(code_verifier: Optional[str], grant: AuthorizationCodeGrant) -> bool:
    if grant.code_challenge is None:
        return True
    if not code_verifier:
        return False
    if grant.code_challenge_method and grant.code_challenge_method.upper() != "S256":
        return False
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return secrets.compare_digest(computed, grant.code_challenge)


def _extract_bearer_token(request: Request) -> Optional[str]:
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None
    scheme, param = get_authorization_scheme_param(auth_header)
    if not scheme:
        return None
    if scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unsupported authorization scheme",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not param:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return param


async def get_authenticated_principal(request: Request):
    """Authenticate the incoming request using OAuth or nonce token."""

    nonce = request.query_params.get("nonce")
    if nonce is not None:
        if _is_nonce_valid(nonce):
            return {
                "client_id": f"nonce:{nonce}",
                "auth_method": "query_token",
                "scope": "",
            }
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid nonce token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    bearer = _extract_bearer_token(request)
    if bearer is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(bearer, MCP_SERVER_SECRET_KEY, algorithms=[ALGORITHM])
        client_id: Optional[str] = payload.get("sub")
        if not client_id:
            raise credentials_exception
        client = _get_registered_client(client_id)
        if client is None:
            raise credentials_exception
        scope = payload.get("scope", "")
        user_email = payload.get("user_email")
    except JWTError:
        raise credentials_exception

    principal = {"client_id": client_id, "auth_method": "oauth", "scope": scope}
    if user_email:
        principal["user_email"] = user_email
    return principal


# --- Graphiti Configuration ---
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDER_MODEL = "text-embedding-3-small"
SMALL_LLM_MODEL = "gpt-4o-mini"


class GraphitiLLMConfig(BaseModel):
    """Configuration for the LLM client."""

    api_key: Optional[str] = Field(
        default=None, description="The OpenAI API key to use."
    )
    model: str = Field(
        default=DEFAULT_LLM_MODEL, description="The LLM model to use."
    )
    small_model: str = Field(
        default=SMALL_LLM_MODEL, description="The small LLM model to use."
    )
    temperature: float = 0.0

    # Azure OpenAI
    azure_openai_endpoint: Optional[str] = Field(
        default=None, description="The Azure OpenAI endpoint to use."
    )
    azure_openai_deployment_name: Optional[str] = Field(
        default=None, description="The Azure OpenAI deployment name to use."
    )
    azure_openai_api_version: Optional[str] = Field(
        default=None, description="The Azure OpenAI API version to use."
    )

    @field_validator("api_key", "model")
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be empty")
        return v

    @classmethod
    def from_args(cls, args) -> "GraphitiLLMConfig":
        """Create a configuration from command-line arguments."""
        model_env = os.environ.get("MODEL_NAME", "")
        model = model_env if model_env.strip() else DEFAULT_LLM_MODEL
        small_model_env = os.environ.get("SMALL_MODEL_NAME", "")
        small_model = (
            small_model_env if small_model_env.strip() else SMALL_LLM_MODEL
        )

        azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", None)
        azure_openai_deployment_name = os.environ.get(
            "AZURE_OPENAI_DEPLOYMENT_NAME", None
        )
        azure_openai_api_version = os.environ.get(
            "AZURE_OPENAI_API_VERSION", None
        )

        config = cls(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=model,
            small_model=small_model,
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.0")),
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_deployment_name=azure_openai_deployment_name,
            azure_openai_api_version=azure_openai_api_version,
        )

        # Override with command-line arguments if provided
        if args.openai_api_key:
            config.api_key = args.openai_api_key
        if args.model:
            config.model = args.model
        if args.small_model.strip():
            config.small_model = args.small_model
        else:
            logger.warning(
                f"Empty small_model name provided, using default: {SMALL_LLM_MODEL}"
            )

        if hasattr(args, "temperature") and args.temperature is not None:
            config.temperature = args.temperature

        return config

    def create_llm_client(self) -> LLMClient:
        """Create an LLM client based on this configuration."""
        from graphiti_core.llm_client.azure_openai_client import (
            AzureOpenAIClient,
        )
        from graphiti_core.llm_client.openai_client import OpenAIClient
        from openai import AsyncAzureOpenAI

        if (
            self.azure_openai_endpoint
            and self.azure_openai_deployment_name
            and self.azure_openai_api_version
        ):
            logger.info("Using Azure OpenAI for LLM.")
            # If you need to use a token provider, you can add it here.
            # See https://github.com/openai/openai-python#microsoft-azure-endpoints
            # for more details.
            token_provider = None
            if token_provider:
                return AzureOpenAIClient(
                    azure_client=AsyncAzureOpenAI(
                        azure_endpoint=self.azure_openai_endpoint,
                        azure_deployment=self.azure_openai_deployment_name,
                        api_version=self.azure_openai_api_version,
                        azure_ad_token_provider=token_provider,
                    ),
                    config=LLMConfig(
                        api_key=self.api_key,
                        model=self.model,
                        small_model=self.small_model,
                        temperature=self.temperature,
                    ),
                )
            else:
                return AzureOpenAIClient(
                    azure_client=AsyncAzureOpenAI(
                        azure_endpoint=self.azure_openai_endpoint,
                        azure_deployment=self.azure_openai_deployment_name,
                        api_version=self.azure_openai_api_version,
                        api_key=self.api_key,
                    ),
                    config=LLMConfig(
                        api_key=self.api_key,
                        model=self.model,
                        small_model=self.small_model,
                        temperature=self.temperature,
                    ),
                )
        logger.info("Using OpenAI for LLM.")
        return OpenAIClient(
            api_key=self.api_key, model=self.model, small_model=self.small_model
        )


class GraphitiEmbedderConfig(BaseModel):
    """Configuration for the embedder client."""

    api_key: Optional[str] = Field(
        default=None, description="The OpenAI API key to use."
    )
    model: str = Field(
        default=DEFAULT_EMBEDDER_MODEL, description="The embedder model to use."
    )
    # Azure OpenAI
    azure_openai_endpoint: Optional[str] = Field(
        default=None, description="The Azure OpenAI endpoint to use."
    )
    azure_openai_deployment_name: Optional[str] = Field(
        default=None, description="The Azure OpenAI deployment name to use."
    )
    azure_openai_api_version: Optional[str] = Field(
        default=None, description="The Azure OpenAI API version to use."
    )

    @classmethod
    def from_args(cls, args) -> "GraphitiEmbedderConfig":
        """Create a configuration from command-line arguments."""
        model_env = os.environ.get("EMBEDDER_MODEL_NAME", "")
        model = model_env if model_env.strip() else DEFAULT_EMBEDDER_MODEL

        azure_openai_endpoint = os.environ.get(
            "AZURE_OPENAI_EMBEDDING_ENDPOINT", None
        )
        azure_openai_deployment_name = os.environ.get(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", None
        )
        azure_openai_api_version = os.environ.get(
            "AZURE_OPENAI_EMBEDDING_API_VERSION", None
        )

        config = cls(
            api_key=os.environ.get(
                "OPENAI_API_KEY",
                None,
            ),
            model=model,
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_deployment_name=azure_openai_deployment_name,
            azure_openai_api_version=azure_openai_api_version,
        )

        # Override with command-line arguments if provided
        if hasattr(args, "openai_api_key") and args.openai_api_key:
            config.api_key = args.openai_api_key
        if hasattr(args, "embedder_model") and args.embedder_model:
            config.model = args.embedder_model

        return config

    def create_embedder_client(self) -> EmbedderClient:
        """Create an embedder client based on this configuration."""
        from graphiti_core.embedder.azure_openai_embedder import (
            AzureOpenAIEmbedder,
        )
        from graphiti_core.embedder.openai_embedder import OpenAIEmbedder
        from openai import AsyncAzureOpenAI

        if (
            self.azure_openai_endpoint
            and self.azure_openai_deployment_name
            and self.azure_openai_api_version
        ):
            logger.info("Using Azure OpenAI for embedder.")
            # If you need to use a token provider, you can add it here.
            # See https://github.com/openai/openai-python#microsoft-azure-endpoints
            # for more details.
            token_provider = None
            if token_provider:
                return AzureOpenAIEmbedder(
                    azure_client=AsyncAzureOpenAI(
                        azure_endpoint=self.azure_openai_endpoint,
                        azure_deployment=self.azure_openai_deployment_name,
                        api_version=self.azure_openai_api_version,
                        azure_ad_token_provider=token_provider,
                    ),
                    model=self.model,
                )
            else:
                return AzureOpenAIEmbedder(
                    azure_client=AsyncAzureOpenAI(
                        azure_endpoint=self.azure_openai_endpoint,
                        azure_deployment=self.azure_openai_deployment_name,
                        api_version=self.azure_openai_api_version,
                        api_key=self.api_key,
                    ),
                    model=self.model,
                )
        logger.info("Using OpenAI for embedder.")
        return OpenAIEmbedder(api_key=self.api_key, model=self.model)


# --- Graphiti Instance ---
graph: Optional[Graph] = None
llm_client: Optional[LLMClient] = None
embedder_client: Optional[EmbedderClient] = None
semaphore: Optional[asyncio.Semaphore] = None


# --- FastAPI App and Router ---
app = FastAPI()
router = APIRouter()
oauth_router = APIRouter(prefix="/oauth", tags=["oauth"])

GRAPHITI_MCP_INSTRUCTIONS = """
You are a memory assistant for a large language model. Your purpose is to enrich the user's conversation with relevant facts from a knowledge graph.

The user will provide you with a query. Your task is to:
1. Search for relevant facts in the knowledge graph using the `search_memory_facts` tool.
2. If the user is asking a question that can be answered by the facts, synthesize an answer from the facts and respond to the user.
3. If the user is making a statement or having a conversation, and the facts can enrich the conversation, do so.
4. If no relevant facts are found, you MUST respond with "I have no facts on this topic."
5. You can search for entities in the graph using the `search_memory_nodes` tool. This is useful for discovery.
"""


# --- Utility Functions ---
def format_fact_result(fact: Dict) -> Dict:
    """Format a fact result for display."""
    if "fact_embedding" in fact:
        del fact["fact_embedding"]
    if "attributes" in fact and "fact_embedding" in fact["attributes"]:
        del fact["attributes"]["fact_embedding"]
    return fact

# --- OAuth Endpoints ---


def _metadata_base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _render_authorize_page(
    client: RegisteredClient,
    params: AuthorizationRequestParams,
    error: Optional[str] = None,
) -> str:
    hidden_fields = []
    for field in (
        "client_id",
        "redirect_uri",
        "response_type",
        "scope",
        "state",
        "code_challenge",
        "code_challenge_method",
    ):
        value = getattr(params, field)
        if value is None:
            continue
        hidden_fields.append(
            f'<input type="hidden" name="{html.escape(field)}" value="{html.escape(str(value))}">'
        )
    hidden_inputs = "\n        ".join(hidden_fields)
    error_block = (
        f"<div class='error'>{html.escape(error)}</div>" if error else ""
    )
    scope_display = html.escape(params.scope or "") or "mcp:read mcp:write"
    demo_user = _get_demo_user()
    demo_button = (
        ""
        if demo_user is None
        else """
          <button type="submit" name="action" value="demo" class="btn-secondary" formnovalidate>
            Use Demo Account
          </button>
        """
    )
    return f"""
<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\">
    <title>Authorize {html.escape(client.client_name or client.client_id)}</title>
    <style>
      body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f7fafc; margin: 0; padding: 0; }}
      .container {{ max-width: 420px; margin: 48px auto; background: #fff; border-radius: 12px; padding: 32px; box-shadow: 0 15px 35px rgba(50, 50, 93, 0.1); }}
      h1 {{ margin-top: 0; color: #1a202c; }}
      .subtitle {{ color: #4a5568; margin-bottom: 24px; }}
      .app-info {{ background: #edf2f7; border-radius: 10px; padding: 16px; margin-bottom: 24px; }}
      .app-name {{ font-weight: 600; color: #2d3748; }}
      label {{ display: block; font-weight: 600; margin-bottom: 8px; color: #2d3748; }}
      input[type='email'], input[type='password'] {{ width: 100%; padding: 12px; border-radius: 8px; border: 1px solid #cbd5e0; margin-bottom: 16px; }}
      .btn-primary {{ width: 100%; padding: 12px; background: #4c51bf; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 15px; }}
      .btn-secondary {{ width: 100%; padding: 12px; background: #edf2f7; color: #4c51bf; border: 1px solid #cbd5e0; border-radius: 8px; cursor: pointer; font-size: 15px; margin-top: 12px; }}
      .error {{ background: #fed7d7; color: #742a2a; padding: 12px; border-radius: 8px; margin-bottom: 16px; }}
      .scope {{ color: #4a5568; font-size: 14px; margin-top: 8px; }}
    </style>
  </head>
  <body>
    <div class=\"container\">
      <h1>Sign in</h1>
      <div class=\"subtitle\">Authorize access for {html.escape(client.client_name or client.client_id)}</div>
      <div class=\"app-info\">
        <div class=\"app-name\">Requested scopes</div>
        <div class=\"scope\">{scope_display}</div>
      </div>
      {error_block}
      <form method=\"post\" action=\"/oauth/authorize\">
        {hidden_inputs}
        <label for=\"email\">Email</label>
        <input type=\"email\" id=\"email\" name=\"email\" required placeholder=\"you@example.com\">
        <label for=\"password\">Password</label>
        <input type=\"password\" id=\"password\" name=\"password\" required placeholder=\"••••••••\">
        <button type=\"submit\" name=\"action\" value=\"approve\" class=\"btn-primary\">Sign in &amp; Authorize</button>
        {demo_button}
      </form>
    </div>
  </body>
</html>
"""
@app.get("/.well-known/oauth-authorization-server")
async def oauth_metadata(request: Request):
    base_url = _metadata_base_url(request)
    return {
        "issuer": base_url,
        "authorization_endpoint": f"{base_url}/oauth/authorize",
        "token_endpoint": f"{base_url}/oauth/token",
        "registration_endpoint": f"{base_url}/oauth/register",
        "response_types_supported": ["code"],
        "grant_types_supported": [
            "authorization_code",
            "refresh_token",
            "client_credentials",
        ],
        "token_endpoint_auth_methods_supported": [
            "client_secret_basic",
            "client_secret_post",
            "none",
        ],
        "scopes_supported": [""],
        "code_challenge_methods_supported": ["S256"],
    }


def _validate_authorization_request(
    params: AuthorizationRequestParams, client: RegisteredClient
) -> Optional[str]:
    if params.redirect_uri not in client.redirect_uris:
        return "Invalid redirect_uri"
    if "authorization_code" not in client.grant_types:
        return "Client is not allowed to use the authorization_code grant"
    if params.code_challenge_method and params.code_challenge_method.upper() != "S256":
        return "Unsupported PKCE code_challenge_method"
    if not _client_requires_secret(client) and not params.code_challenge:
        return "Public clients must supply a PKCE code_challenge"
    return None


def _parse_authorization_params(raw_params: Dict[str, Optional[str]]) -> AuthorizationRequestParams:
    normalized: Dict[str, Optional[str]] = {}
    for key, value in raw_params.items():
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped and key not in {"scope", "state"}:
                continue
            normalized[key] = stripped
        else:
            normalized[key] = value
    if "scope" not in normalized:
        normalized["scope"] = ""
    return AuthorizationRequestParams.model_validate(normalized)


@oauth_router.get("/authorize", response_class=HTMLResponse, include_in_schema=False)
async def oauth_authorize_get(request: Request):
    try:
        params = _parse_authorization_params(dict(request.query_params))
    except ValidationError as exc:
        logger.warning("Invalid authorization request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid authorization request",
        ) from exc

    client = _get_registered_client(params.client_id)
    if client is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown client_id")

    validation_error = _validate_authorization_request(params, client)
    if validation_error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=validation_error)

    page = _render_authorize_page(client, params)
    return HTMLResponse(page)


@oauth_router.post("/authorize", include_in_schema=False)
async def oauth_authorize_post(request: Request):
    form = await request.form()
    raw_params = {key: form.get(key) for key in (
        "client_id",
        "redirect_uri",
        "response_type",
        "scope",
        "state",
        "code_challenge",
        "code_challenge_method",
    )}
    try:
        params = _parse_authorization_params(raw_params)
    except ValidationError as exc:
        logger.warning("Invalid authorization request payload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid authorization request",
        ) from exc

    client = _get_registered_client(params.client_id)
    if client is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown client_id")

    validation_error = _validate_authorization_request(params, client)
    if validation_error:
        return HTMLResponse(
            _render_authorize_page(client, params, error=validation_error),
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    action = (form.get("action") or "approve").lower()
    user: Optional[AuthUser]
    if action == "demo":
        user = _get_demo_user()
        if user is None:
            return HTMLResponse(
                _render_authorize_page(
                    client,
                    params,
                    error="Demo account is disabled on this server",
                ),
                status_code=status.HTTP_400_BAD_REQUEST,
            )
    else:
        email = (form.get("email") or "").strip()
        password = form.get("password") or ""
        if not email or not password or not _authenticate_user(email, password):
            return HTMLResponse(
                _render_authorize_page(client, params, error="Invalid email or password"),
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        user = _get_auth_user(email)
        if user is None:
            return HTMLResponse(
                _render_authorize_page(client, params, error="Account is not available"),
                status_code=status.HTTP_400_BAD_REQUEST,
            )

    authorization_code = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(seconds=AUTHORIZATION_CODE_TTL_SECONDS)
    grant = AuthorizationCodeGrant(
        code=authorization_code,
        client_id=client.client_id,
        redirect_uri=params.redirect_uri,
        scope=params.scope or "",
        user_email=user.email,
        code_challenge=params.code_challenge,
        code_challenge_method=params.code_challenge_method,
        expires_at=expires_at,
    )
    _store_authorization_code(grant)
    logger.info(
        "Issued authorization code for client %s (expires in %s seconds)",
        client.client_id,
        AUTHORIZATION_CODE_TTL_SECONDS,
    )

    response_params = {"code": authorization_code}
    if params.state:
        response_params["state"] = params.state
    separator = "&" if "?" in params.redirect_uri else "?"
    redirect_target = f"{params.redirect_uri}{separator}{urlencode(response_params)}"
    return RedirectResponse(url=redirect_target, status_code=status.HTTP_303_SEE_OTHER)
@oauth_router.post(
    "/register",
    response_model=ClientRegistrationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_oauth_client(
    registration: ClientRegistrationRequest,
):
    issued_at = int(time.time())
    client_id = uuid.uuid4().hex
    secret_required = registration.token_endpoint_auth_method != "none"
    client_secret = secrets.token_urlsafe(32) if secret_required else None
    client_record = RegisteredClient(
        client_id=client_id,
        client_secret_hash=pwd_context.hash(client_secret)
        if secret_required and client_secret
        else None,
        client_name=registration.client_name,
        redirect_uris=registration.redirect_uris,
        grant_types=registration.grant_types,
        token_endpoint_auth_method=registration.token_endpoint_auth_method,
        scope=registration.scope,
        client_id_issued_at=issued_at,
        client_secret_expires_at=0,
    )
    _store_registered_client(client_record)
    logger.info("Registered new OAuth client '%s'", registration.client_name)
    response = ClientRegistrationResponse(
        client_id=client_id,
        client_secret=client_secret,
        client_id_issued_at=issued_at,
        client_secret_expires_at=0,
        client_name=registration.client_name,
        redirect_uris=registration.redirect_uris,
        grant_types=registration.grant_types,
        token_endpoint_auth_method=registration.token_endpoint_auth_method,
        scope=registration.scope,
    )
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content=response.model_dump(exclude_none=True),
    )


def _extract_basic_credentials(request: Request) -> Optional[Dict[str, str]]:
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None
    scheme, param = get_authorization_scheme_param(auth_header)
    if not scheme:
        return None
    if scheme.lower() != "basic":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported authorization scheme for token endpoint",
        )
    if not param:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing credentials in Authorization header",
        )
    try:
        decoded = base64.b64decode(param).decode()
        client_id, client_secret = decoded.split(":", 1)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to decode basic auth credentials: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid basic authentication credentials",
        ) from exc
    return {"client_id": client_id, "client_secret": client_secret}


@oauth_router.post("/token", response_model=Token)
async def issue_access_token(
    request: Request,
    grant_type: str = Form(..., alias="grant_type"),
    scope: str = Form("", alias="scope"),
    client_id_form: Optional[str] = Form(None, alias="client_id"),
    client_secret_form: Optional[str] = Form(None, alias="client_secret"),
    code: Optional[str] = Form(None, alias="code"),
    redirect_uri_form: Optional[str] = Form(None, alias="redirect_uri"),
    code_verifier: Optional[str] = Form(None, alias="code_verifier"),
    refresh_token_form: Optional[str] = Form(None, alias="refresh_token"),
):
    credentials = _extract_basic_credentials(request)
    client_id = None
    client_secret = None
    if credentials:
        client_id = credentials["client_id"]
        client_secret = credentials["client_secret"]

    if client_id is None or client_secret is None:
        client_id = client_id_form or client_id
        client_secret = client_secret_form or client_secret
    client: Optional[RegisteredClient] = None
    user_email: Optional[str] = None
    response_scope = scope or ""
    refresh_token_value: Optional[str] = None

    if grant_type == "client_credentials":
        if not client_id or not client_secret:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Client authentication failed",
                headers={"WWW-Authenticate": "Basic"},
            )

        client = _get_registered_client(client_id)
        if client is None or not _verify_client_secret(client, client_secret):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid client credentials",
                headers={"WWW-Authenticate": "Basic"},
            )

        if "client_credentials" not in client.grant_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Client is not allowed to use client_credentials grant",
            )
    elif grant_type == "authorization_code":
        if not client_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing client_id",
            )
        client = _get_registered_client(client_id)
        if client is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid client credentials",
            )
        if _client_requires_secret(client):
            if not client_secret:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Client authentication failed",
                    headers={"WWW-Authenticate": "Basic"},
                )
            if not _verify_client_secret(client, client_secret):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid client credentials",
                    headers={"WWW-Authenticate": "Basic"},
                )

        if "authorization_code" not in client.grant_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Client is not allowed to use authorization_code grant",
            )

        if not code:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing authorization code",
            )

        grant = _get_authorization_code(code)
        if grant is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired authorization code",
            )
        if grant.client_id != client.client_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Authorization code was not issued to this client",
            )
        if not redirect_uri_form:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing redirect_uri",
            )
        if grant.redirect_uri != redirect_uri_form:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="redirect_uri does not match original authorization request",
            )
        if not _verify_pkce(code_verifier, grant):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid PKCE verifier",
            )
        grant = _consume_authorization_code(code)
        if grant is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Authorization code is no longer valid",
            )
        response_scope = grant.scope
        user_email = grant.user_email

        refresh_token_value = secrets.token_urlsafe(48)
        refresh_expires = datetime.utcnow() + timedelta(seconds=REFRESH_TOKEN_TTL_SECONDS)
        refresh_grant = RefreshTokenGrant(
            token=refresh_token_value,
            client_id=client.client_id,
            scope=response_scope,
            user_email=user_email,
            expires_at=refresh_expires,
        )
        _store_refresh_token(refresh_grant)
    elif grant_type == "refresh_token":
        if not refresh_token_form:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing refresh_token",
            )
        grant = _get_refresh_token(refresh_token_form)
        if grant is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired refresh_token",
            )
        client = _get_registered_client(grant.client_id)
        if client is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid client credentials",
            )
        if _client_requires_secret(client):
            if not client_secret:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Client authentication failed",
                    headers={"WWW-Authenticate": "Basic"},
                )
            if not _verify_client_secret(client, client_secret):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid client credentials",
                    headers={"WWW-Authenticate": "Basic"},
                )
        elif client_id and client_id != grant.client_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Client mismatch for refresh_token",
            )

        consumed = _consume_refresh_token(refresh_token_form)
        if consumed is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="refresh_token is no longer valid",
            )
        response_scope = consumed.scope
        user_email = consumed.user_email

        refresh_token_value = secrets.token_urlsafe(48)
        refresh_expires = datetime.utcnow() + timedelta(seconds=REFRESH_TOKEN_TTL_SECONDS)
        _store_refresh_token(
            RefreshTokenGrant(
                token=refresh_token_value,
                client_id=consumed.client_id,
                scope=consumed.scope,
                user_email=consumed.user_email,
                expires_at=refresh_expires,
            )
        )
        client = _get_registered_client(consumed.client_id)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported grant_type",
        )

    if client is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid client",
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": client.client_id,
            "client_name": client.client_name,
            "scope": response_scope,
            "grant_type": grant_type,
            **({"user_email": user_email} if user_email else {}),
        },
        expires_delta=access_token_expires,
    )
    token_response = Token(
        access_token=access_token,
        token_type="Bearer",
        expires_in=int(access_token_expires.total_seconds()),
        scope=response_scope,
        refresh_token=refresh_token_value,
    )
    return JSONResponse(content=token_response.model_dump(exclude_none=True))


# --- API Endpoints ---


@router.post(
    "/v1/episodes",
    response_model=SuccessResponse,
    responses={500: {"model": ErrorResponse}},
)
async def add_memory(
    request: Request,
    payload: Message,
    principal: dict = Depends(get_authenticated_principal),
):
    """
    Add a memory to the graph. The payload should be a JSON string with the following format:
    {
        "role": "human",
        "content": "The user's message.",
        "metadata": {
            "episode_id": "A unique ID for this episode.",
            "timestamp": "The timestamp of the message.",
            // Other metadata
        }
    }
    """
    global graph
    if not graph:
        raise HTTPException(status_code=500, detail="Graph not initialized")

    async with semaphore:
        try:
            message = Message(**payload.dict())
            await graph.add_message(message)
            return SuccessResponse(success=True, message="Memory added successfully")
        except Exception as e:
            logger.exception("Failed to add memory")
            raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/v1/episodes/{episode_id}/sse",
    responses={500: {"model": ErrorResponse}},
)
async def get_episode_sse(
    episode_id: str,
    request: Request,
    principal: dict = Depends(get_authenticated_principal),
):
    """
    Get real-time updates for an episode.
    """
    global graph
    if not graph:
        raise HTTPException(status_code=500, detail="Graph not initialized")

    async def event_generator() -> AsyncGenerator[StreamEvent, None]:
        try:
            async for data in graph.get_episode_events(episode_id):
                if await request.is_disconnected():
                    logger.info("Client disconnected, closing SSE stream.")
                    break
                # Assuming data is already a StreamEvent or can be converted to one
                if isinstance(data, StreamEvent):
                    yield data
                else:
                    # You might need to format your data into a StreamEvent
                    # This is just an example
                    yield StreamEvent(
                        event="message", data=json.dumps(data, default=str)
                    )
        except Exception as e:
            logger.exception(f"Error in SSE event generator for episode {episode_id}")
            # Send an error event to the client
            yield StreamEvent(
                event="error", data=json.dumps({"error": str(e)}, default=str)
            )

    return EventSourceResponse(event_generator())


@router.get(
    "/v1/facts/search",
    response_model=FactSearchResponse,
    responses={500: {"model": ErrorResponse}},
)
async def search_memory_facts(
    query: str,
    principal: dict = Depends(get_authenticated_principal),
):
    """
    Search for facts in the graph using natural language.
    """
    global graph
    if not graph:
        raise HTTPException(status_code=500, detail="Graph not initialized")

    try:
        facts = await graph.search_facts(query)
        formatted_facts = [format_fact_result(fact) for fact in facts]
        return FactSearchResponse(facts=formatted_facts)
    except Exception as e:
        logger.exception("Failed to search facts")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/v1/nodes/search",
    response_model=EpisodeSearchResponse,
    responses={500: {"model": ErrorResponse}},
)
async def search_memory_nodes(
    query: str,
    principal: dict = Depends(get_authenticated_principal),
):
    """
    Search for nodes (entities) in the graph using natural language.
    """
    global graph
    if not graph:
        raise HTTPException(status_code=500, detail="Graph not initialized")

    try:
        nodes = await graph.search_nodes(query)
        return EpisodeSearchResponse(episodes=[node.to_dict() for node in nodes])
    except Exception as e:
        logger.exception("Failed to search nodes")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/v1/status",
    response_model=StatusResponse,
    responses={500: {"model": ErrorResponse}},
)
async def get_status():
    """
    Get the status of the server.
    """
    global graph
    if not graph:
        return StatusResponse(status="error", message="Graph not initialized")
    return StatusResponse(status="ok")


# --- MCP Server Integration ---
def setup_mcp_server() -> Server:
    """
    Set up the MCP server with the Graphiti toolset.
    """
    mcp_server = Server(
        name="graphiti-memory",
        instructions=GRAPHITI_MCP_INSTRUCTIONS,
        host="0.0.0.0",
        port=5555,
    )

    # Register the FastAPI router with the MCP server
    mcp_server.include_router(oauth_router)
    mcp_server.include_router(router, prefix="/v1")

    # Define tools available to the LLM
    @mcp_server.tool()
    async def add_memory(
        content: str,
        role: str = "human",
        group_id: str = "default",
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Adds a new memory to the graph memory. The memory is a piece of text
        associated with a role (e.g., 'human', 'ai').
        The new memory is processed asynchronously to extract entities and
        relationships.
        """
        global graph
        if not graph:
            return "Error: Graph not initialized"

        if metadata is None:
            metadata = {}

        message = Message(
            role=MessageRole(role),
            content=content,
            metadata={
                "group_id": group_id,
                "timestamp": datetime.now().strftime(DATETIME_FORMAT),
                **metadata,
            },
        )
        await graph.add_message(message)
        return "Memory added successfully. It will be processed asynchronously."

    @mcp_server.tool()
    async def search_memory_facts(query: str) -> str:
        """
        Searches for facts in the graph memory using natural language.
        Returns a list of facts related to the query.
        """
        global graph
        if not graph:
            return "Error: Graph not initialized"
        facts = await graph.search_facts(query)
        return json.dumps([format_fact_result(fact) for fact in facts], default=str)

    @mcp_server.tool()
    async def search_memory_nodes(query: str) -> str:
        """
        Searches for nodes (entities) in the graph memory using natural language.
        Returns a list of nodes that match the query.
        """
        global graph
        if not graph:
            return "Error: Graph not initialized"
        nodes = await graph.search_nodes(query)
        return json.dumps([node.to_dict() for node in nodes], default=str)

    return mcp_server


# --- Main Application ---
async def main():
    """
    Main function to initialize and run the server.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Graphiti MCP Server")
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL_NAME", DEFAULT_LLM_MODEL),
        help="LLM model name",
    )
    parser.add_argument(
        "--small-model",
        type=str,
        default=os.environ.get("SMALL_MODEL_NAME", SMALL_LLM_MODEL),
        help="Small LLM model name",
    )
    parser.add_argument(
        "--embedder-model",
        type=str,
        default=os.environ.get("EMBEDDER_MODEL_NAME", DEFAULT_EMBEDDER_MODEL),
        help="Embedder model name",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reloading"
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default="default",
        help="Default group ID for memories",
    )

    args = parser.parse_args()

    global graph, llm_client, embedder_client, semaphore
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)

    # Initialize configurations and clients
    llm_config = GraphitiLLMConfig.from_args(args)
    llm_client = llm_config.create_llm_client()
    logger.info(
        f"Initialized LLM client with model: {llm_config.model} and small"
        f" model: {llm_config.small_model}"
    )

    embedder_config = GraphitiEmbedderConfig.from_args(args)
    embedder_client = embedder_config.create_embedder_client()
    logger.info(f"Initialized embedder client with model: {embedder_config.model}")

    # Initialize Graph
    graph = await Graph.create(
        llm_client=llm_client,
        embedder=embedder_client,
    )
    logger.info("Graphiti graph initialized.")
    logger.info(f"Concurrency limit set to: {SEMAPHORE_LIMIT}")

    # Set up and run the MCP server in the background
    mcp_server_instance = setup_mcp_server()
    asyncio.create_task(mcp_server_instance.run())

    # Include the routers in the FastAPI app
    app.include_router(oauth_router)
    app.include_router(router)

    # Run the FastAPI server
    config = uvicorn.Config(
        app, host=args.host, port=args.port, log_level="info", reload=args.reload
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
