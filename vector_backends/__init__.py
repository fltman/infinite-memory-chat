"""
Vector backend interfaces for infinite memory chat.
"""

from .base import VectorBackend
from .openai_backend import OpenAIBackend
from .mongodb_backend import MongoDBBackend

# Async backends
from .async_base import AsyncVectorBackend, AsyncProgressTracker, AsyncBatchProcessor
from .async_mongodb_backend import AsyncMongoDBBackend
from .async_wrapper import AsyncBackendWrapper, AsyncBackendFactory, AsyncOperationManager

__all__ = [
    # Sync backends
    'VectorBackend', 'OpenAIBackend', 'MongoDBBackend',
    # Async backends
    'AsyncVectorBackend', 'AsyncProgressTracker', 'AsyncBatchProcessor',
    'AsyncMongoDBBackend', 'AsyncBackendWrapper', 'AsyncBackendFactory',
    'AsyncOperationManager'
]
