"""
Configuration management for infinite memory chat backends.
"""

import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class BackendType(Enum):
    OPENAI = "openai"
    MONGODB = "mongodb"
    ASYNC_MONGODB = "async_mongodb"


@dataclass
class BackendConfig:
    """Configuration for vector storage backends."""
    
    # Backend selection
    backend_type: BackendType = BackendType.OPENAI
    
    # OpenAI configuration (used for both chat and embeddings)
    openai_api_key: Optional[str] = None
    
    # MongoDB configuration
    mongodb_connection_string: Optional[str] = None
    mongodb_database: str = "infinite_memory_chat"
    
    # Embedding configuration
    embedding_model: str = "text-embedding-3-small"
    
    # Chat model configuration  
    chat_model: str = "gpt-4o-mini"
    
    @classmethod
    def from_env(cls) -> 'BackendConfig':
        """Load configuration from environment variables."""
        # Determine backend type
        backend_env = os.getenv('VECTOR_BACKEND', 'openai').lower()
        try:
            backend_type = BackendType(backend_env)
        except ValueError:
            print(f"⚠️  Unknown backend type '{backend_env}', defaulting to OpenAI")
            backend_type = BackendType.OPENAI
        
        return cls(
            backend_type=backend_type,
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            mongodb_connection_string=os.getenv('MONGODB_CONNECTION_STRING'),
            mongodb_database=os.getenv('MONGODB_DATABASE', 'infinite_memory_chat'),
            embedding_model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
            chat_model=os.getenv('CHAT_MODEL', 'gpt-4o-mini')
        )
    
    def validate(self) -> bool:
        """Validate configuration and return True if valid."""
        issues = []
        
        # OpenAI API key is always required (for chat and/or embeddings)
        if not self.openai_api_key:
            issues.append("OPENAI_API_KEY is required")
        
        # Backend-specific validation
        if self.backend_type in [BackendType.MONGODB, BackendType.ASYNC_MONGODB]:
            if not self.mongodb_connection_string:
                issues.append("MONGODB_CONNECTION_STRING is required for MongoDB backend")
        
        if issues:
            print("❌ Configuration validation failed:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        return True
    
    def get_display_info(self) -> dict:
        """Get configuration info safe for display (without secrets)."""
        return {
            "backend_type": self.backend_type.value,
            "mongodb_database": self.mongodb_database,
            "embedding_model": self.embedding_model,
            "chat_model": self.chat_model,
            "openai_api_key_set": bool(self.openai_api_key),
            "mongodb_connection_set": bool(self.mongodb_connection_string)
        }


def create_backend(config: BackendConfig):
    """Factory function to create the appropriate synchronous backend."""
    from vector_backends.base import VectorBackend
    
    if config.backend_type == BackendType.OPENAI:
        from vector_backends.openai_backend import OpenAIBackend
        from openai import OpenAI
        
        client = OpenAI(api_key=config.openai_api_key)
        return OpenAIBackend(client=client)
    
    elif config.backend_type == BackendType.MONGODB:
        from vector_backends.mongodb_backend import MongoDBBackend
        from openai import OpenAI
        
        embedding_client = OpenAI(api_key=config.openai_api_key)
        return MongoDBBackend(
            connection_string=config.mongodb_connection_string,
            database_name=config.mongodb_database,
            embedding_client=embedding_client,
            embedding_model=config.embedding_model
        )
    
    elif config.backend_type == BackendType.ASYNC_MONGODB:
        # For async backends, we'll create a wrapped version for backward compatibility
        from vector_backends.mongodb_backend import MongoDBBackend
        from vector_backends.async_wrapper import AsyncBackendWrapper
        from openai import OpenAI
        
        embedding_client = OpenAI(api_key=config.openai_api_key)
        sync_backend = MongoDBBackend(
            connection_string=config.mongodb_connection_string,
            database_name=config.mongodb_database,
            embedding_client=embedding_client,
            embedding_model=config.embedding_model
        )
        # Return sync backend with note about async capabilities
        return sync_backend
    
    else:
        raise ValueError(f"Unsupported backend type: {config.backend_type}")


async def create_async_backend(config: BackendConfig):
    """Factory function to create the appropriate asynchronous backend."""
    from vector_backends.async_base import AsyncVectorBackend
    from vector_backends.async_wrapper import AsyncBackendFactory
    
    if config.backend_type == BackendType.ASYNC_MONGODB:
        return await AsyncBackendFactory.create_async_backend(
            "async_mongodb",
            connection_string=config.mongodb_connection_string,
            database_name=config.mongodb_database,
            embedding_model=config.embedding_model
        )
    
    elif config.backend_type == BackendType.MONGODB:
        # Create wrapped version of sync backend for async compatibility
        return await AsyncBackendFactory.create_async_backend(
            "mongodb",
            connection_string=config.mongodb_connection_string,
            database_name=config.mongodb_database,
            embedding_model=config.embedding_model
        )
    
    elif config.backend_type == BackendType.OPENAI:
        # OpenAI backend doesn't have native async support yet, so wrap it
        from vector_backends.openai_backend import OpenAIBackend
        from vector_backends.async_wrapper import AsyncBackendWrapper
        from openai import OpenAI
        
        client = OpenAI(api_key=config.openai_api_key)
        sync_backend = OpenAIBackend(client=client)
        return AsyncBackendWrapper(sync_backend)
    
    else:
        raise ValueError(f"Unsupported backend type: {config.backend_type}")


# Utility functions for easy access
def get_config() -> BackendConfig:
    """Get configuration from environment with validation."""
    config = BackendConfig.from_env()
    if not config.validate():
        raise ValueError("Invalid configuration")
    return config


def is_async_backend(config: BackendConfig) -> bool:
    """Check if the configured backend supports native async operations."""
    return config.backend_type == BackendType.ASYNC_MONGODB
