import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import AsyncGenerator, Dict, List, Optional, Union

import uvicorn
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.security import OAuth2PasswordBearer
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
from pydantic import BaseModel, Field, field_validator
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

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


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


# --- Authentication Functions ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, MCP_SERVER_SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    token_query: Optional[str] = Query(None, alias="token"),
):
    """
    Dependency to get the current user from the token.
    Token can be provided in the Authorization header or as a query parameter.
    """
    auth_token = token or token_query
    if auth_token is None:
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
        payload = jwt.decode(auth_token, MCP_SERVER_SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    # In a real app, you would fetch the user from the database here
    # For now, we'll just return the username
    return {"username": username}


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

# --- API Endpoints ---
@router.post("/token", response_model=Token)
async def login_for_access_token():
    """
    In a real application, you'd have username and password authentication here.
    For this example, we'll just grant a token.
    """
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": "mcp_client"}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post(
    "/v1/episodes",
    response_model=SuccessResponse,
    responses={500: {"model": ErrorResponse}},
)
async def add_memory(
    request: Request,
    payload: Message,
    current_user: dict = Depends(get_current_user),
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
    current_user: dict = Depends(get_current_user),
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
    current_user: dict = Depends(get_current_user),
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
    current_user: dict = Depends(get_current_user),
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

    # Include the main router in the FastAPI app
    app.include_router(router)

    # Run the FastAPI server
    config = uvicorn.Config(
        app, host=args.host, port=args.port, log_level="info", reload=args.reload
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
