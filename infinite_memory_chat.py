"""
Infinite Memory Chat - Proof of Concept
========================================
A chatbot with "infinite" memory by combining:
- OpenAI Responses API for conversation
- Vector Store for long-term memory (OpenAI or MongoDB)
- Automatic archival of older messages
"""

import os
import json
import time
import asyncio
from datetime import datetime
from typing import Optional, Union
from openai import OpenAI
from vector_backends.base import VectorBackend
from vector_backends.openai_backend import OpenAIBackend
from vector_backends.mongodb_backend import MongoDBBackend
from config import BackendConfig, BackendType, create_backend, create_async_backend, is_async_backend

# Configuration
# =============================================================================
# LEVEL 1: Active conversation
# Max number of messages before archival is triggered.
# Lower value = less token consumption per request, but more archival operations.
MAX_MESSAGES = 20

# Number of messages to archive at a time.
# Tip: Overlap can improve context (e.g., archive 10, but keep 2 extra)
ARCHIVE_COUNT = 10

# =============================================================================
# LEVEL 2: Vector store archive
# OpenAI allows max 10,000 files per vector store.
# When MAX_ARCHIVE_FILES is reached, the oldest CONSOLIDATION_COUNT files are
# merged into a single file. All data is preserved - we only reduce file count.
# Example: 100 files → 50 merged into 1 → 51 files remaining
# This makes it take ~50x longer to reach the 10,000 file limit.
MAX_ARCHIVE_FILES = 100

# Number of files to merge into a single file during consolidation
CONSOLIDATION_COUNT = 50

# =============================================================================
# LEVEL 3: File size limit
# OpenAI has a limit of 512 MB per file.
# We set a safety margin at 500 MB.
# When a consolidated file approaches this limit, a new consolidated file is created.
# Theoretical max: ~50 files × 500 MB = 25 GB chat history
MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# =============================================================================
# Model
MODEL = "gpt-5-nano"  

SYSTEM_PROMPT = """You are a helpful assistant with long-term memory.

IMPORTANT ABOUT YOUR MEMORY:
- The active conversation only shows the most recent messages.
- Older conversation history is saved in your knowledge base (file_search).
- If the user references something you discussed earlier, or if you need
  context from earlier in the conversation, ALWAYS search the knowledge base first.
- Never assume you lack information – it may be in the knowledge base.

When searching, consider:
- The user's name, projects, or topics you've discussed
- Decisions or agreements that were made
- Specific details the user has shared

If you find relevant history, feel free to confirm this for the user."""


class InfiniteMemoryChat:
    def __init__(self, backend: Optional[VectorBackend] = None, config: Optional[BackendConfig] = None):
        # Load configuration
        self.config = config or BackendConfig.from_env()
        
        # Initialize OpenAI client for chat responses (always needed)
        self.client = OpenAI(api_key=self.config.openai_api_key)
        
        # Initialize backend with fallback logic
        if backend:
            self.backend = backend
            print(f"🔌 Using provided {type(backend).__name__}")
        else:
            try:
                self.backend = create_backend(self.config)
                print(f"🔌 Using {self.config.backend_type.value} backend")
            except Exception as e:
                print(f"⚠️ Error initializing {self.config.backend_type.value} backend: {e}")
                print("🔄 Falling back to OpenAI backend...")
                fallback_config = BackendConfig(
                    backend_type=BackendType.OPENAI,
                    openai_api_key=self.config.openai_api_key
                )
                self.backend = create_backend(fallback_config)
        
        # Initialize state
        self.conversation_history = []
        self.archive_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.store_id = None
        
    def setup_vector_store(self):
        """Create a new vector store for this session."""
        self.store_id = self.backend.setup_store(self.session_id)
        return self.store_id
    
    def archive_messages(self):
        """Archive the oldest messages using the selected backend."""
        if len(self.conversation_history) < MAX_MESSAGES:
            return
        
        # Extract the oldest messages
        messages_to_archive = self.conversation_history[:ARCHIVE_COUNT]
        self.conversation_history = self.conversation_history[ARCHIVE_COUNT:]
        
        self.archive_count += 1
        success = self.backend.archive_messages(
            messages_to_archive, 
            self.archive_count, 
            self.session_id
        )
        
        if success and self.backend.should_consolidate(MAX_ARCHIVE_FILES):
            print(f"\n🔄 Max archive files reached - consolidating...")
            self._trigger_consolidation()
            
    def _trigger_consolidation(self):
        """Delegate consolidation to backend."""
        # Get oldest archives to consolidate  
        archive_ids = []
        for i in range(min(CONSOLIDATION_COUNT, self.backend.get_archive_count())):
            archive_ids.append(str(i + 1))  # Simple sequential IDs for now
        
        if archive_ids:
            result = self.backend.consolidate_archives(archive_ids)
            if result:
                print(f"✅ Backend consolidation complete: {result}")
            else:
                print("⚠️ Backend consolidation failed")
    
    def chat(self, user_message: str) -> str:
        """Send a message and get a response with backend-aware memory search."""
        
        # Add user's message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Build messages array for API call
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # For MongoDB backend, search archives and add context if archives exist
        if isinstance(self.backend, MongoDBBackend) and self.archive_count > 0:
            try:
                # Search for relevant archived conversations
                search_results = self.backend.search_archives(user_message, limit=3)
                
                if search_results:
                    print(f"🔍 Found {len(search_results)} relevant archive(s)")
                    # Add retrieved context to system prompt
                    context_addon = "\n\nRelevant conversation history from your memory:\n"
                    for result in search_results:
                        context_addon += f"\n--- Archive #{result.get('archive_id', 'unknown')} (Score: {result.get('score', 0):.3f}) ---\n"
                        # Add messages from the archive
                        for msg in result.get('messages', []):
                            role = "User" if msg["role"] == "user" else "Assistant"
                            context_addon += f"{role}: {msg['content']}\n"
                    
                    # Update system prompt with context
                    enhanced_system_prompt = SYSTEM_PROMPT + context_addon
                    messages = [{"role": "system", "content": enhanced_system_prompt}]
                
            except Exception as e:
                print(f"⚠️ Error searching archives: {e}")
        
        # Add current conversation
        messages.extend(self.conversation_history)
        
        # Make API call - backend-specific handling
        if isinstance(self.backend, OpenAIBackend) and self.store_id and self.archive_count > 0:
            # OpenAI backend uses file_search tool
            response = self.client.responses.create(
                model=self.config.chat_model,
                input=messages,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [self.backend.vector_store_id]
                }]
            )
        else:
            # MongoDB backend or no archives - use standard completion
            response = self.client.responses.create(
                model=self.config.chat_model,
                input=messages
            )
        
        # Extract the response
        assistant_message = response.output_text
        
        # Add assistant's response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        # Check if we need to archive
        self.archive_messages()
        
        return assistant_message
    
    def get_status(self):
        """Get conversation status."""
        backend_status = self.backend.get_status()
        
        return {
            "active_messages": len(self.conversation_history),
            "archived_files": backend_status.get("archived_files", 0),
            "consolidations": backend_status.get("consolidations", 0),
            "total_messages": len(self.conversation_history) + (self.archive_count * ARCHIVE_COUNT),
            "consolidated_size_mb": backend_status.get("consolidated_size_mb", 0),
            "backend_type": backend_status.get("backend_type", "unknown"),
            "store_id": self.store_id
        }


class AsyncInfiniteMemoryChat:
    """Async-enabled version supporting both sync and async backends with progress tracking."""
    
    def __init__(self, backend=None, config: Optional[BackendConfig] = None):
        # Load configuration
        self.config = config or BackendConfig.from_env()
        
        # Initialize OpenAI client for chat responses (always needed)
        self.client = OpenAI(api_key=self.config.openai_api_key)
        
        # Backend will be initialized in async context
        self.backend = backend
        self._is_async_backend = False
        
        # Initialize state
        self.conversation_history = []
        self.archive_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.store_id = None
        
    async def initialize_backend(self):
        """Initialize backend in async context."""
        if self.backend:
            print(f"🔌 Using provided {type(self.backend).__name__}")
            self._is_async_backend = hasattr(self.backend, 'archive_messages') and asyncio.iscoroutinefunction(self.backend.archive_messages)
        else:
            try:
                if is_async_backend(self.config):
                    # Create async backend
                    self.backend = await create_async_backend(self.config)
                    self._is_async_backend = True
                    print(f"🚀 Using {self.config.backend_type.value} async backend")
                else:
                    # Create sync backend
                    self.backend = create_backend(self.config)
                    self._is_async_backend = False
                    print(f"🔌 Using {self.config.backend_type.value} sync backend")
            except Exception as e:
                print(f"⚠️ Error initializing {self.config.backend_type.value} backend: {e}")
                print("🔄 Falling back to OpenAI backend...")
                fallback_config = BackendConfig(
                    backend_type=BackendType.OPENAI,
                    openai_api_key=self.config.openai_api_key
                )
                self.backend = create_backend(fallback_config)
                self._is_async_backend = False
        
    async def setup_vector_store(self):
        """Create a new vector store for this session."""
        if self._is_async_backend:
            self.store_id = await self.backend.setup_store(self.session_id)
        else:
            self.store_id = self.backend.setup_store(self.session_id)
        return self.store_id
    
    async def archive_messages_with_progress(self):
        """Archive the oldest messages with progress tracking."""
        if len(self.conversation_history) < MAX_MESSAGES:
            return
        
        # Extract the oldest messages
        messages_to_archive = self.conversation_history[:ARCHIVE_COUNT]
        self.conversation_history = self.conversation_history[ARCHIVE_COUNT:]
        
        self.archive_count += 1
        
        # Progress callback for async operations
        progress_messages = []
        
        async def progress_callback(message: str, progress: float):
            progress_messages.append(message)
            print(f"📦 {message} ({progress*100:.0f}%)")
        
        # Archive with progress tracking
        if self._is_async_backend:
            success = await self.backend.archive_messages(
                messages_to_archive, 
                self.archive_count, 
                self.session_id,
                progress_callback=progress_callback
            )
        else:
            # For sync backends, just print simple message
            print(f"📦 Archiving {len(messages_to_archive)} messages...")
            success = self.backend.archive_messages(
                messages_to_archive, 
                self.archive_count, 
                self.session_id
            )
        
        if success:
            # Check if consolidation needed
            should_consolidate = await self._should_consolidate()
            if should_consolidate:
                print(f"\n🔄 Max archive files reached - consolidating...")
                await self._trigger_consolidation()
    
    async def _should_consolidate(self):
        """Check if consolidation is needed."""
        if self._is_async_backend:
            return await self.backend.should_consolidate(MAX_ARCHIVE_FILES)
        else:
            return self.backend.should_consolidate(MAX_ARCHIVE_FILES)
    
    async def _trigger_consolidation(self):
        """Delegate consolidation to backend with progress tracking."""
        # Get archive count
        if self._is_async_backend:
            archive_count = await self.backend.get_archive_count()
        else:
            archive_count = self.backend.get_archive_count()
        
        # Get oldest archives to consolidate  
        archive_ids = []
        for i in range(min(CONSOLIDATION_COUNT, archive_count)):
            archive_ids.append(str(i + 1))  # Simple sequential IDs for now
        
        if archive_ids:
            # Progress callback for consolidation
            async def consolidation_progress(message: str, progress: float):
                print(f"🔄 {message} ({progress*100:.0f}%)")
            
            if self._is_async_backend:
                result = await self.backend.consolidate_archives(
                    archive_ids, 
                    progress_callback=consolidation_progress
                )
            else:
                print(f"🔄 Consolidating {len(archive_ids)} archives...")
                result = self.backend.consolidate_archives(archive_ids)
            
            if result:
                print(f"✅ Backend consolidation complete: {result}")
            else:
                print("⚠️ Backend consolidation failed")
    
    async def search_archives_with_progress(self, query: str, limit: int = 3):
        """Search archives with progress tracking."""
        if self.archive_count == 0:
            return []
        
        # Progress callback for search
        async def search_progress(message: str, progress: float):
            print(f"🔍 {message} ({progress*100:.0f}%)")
        
        try:
            if self._is_async_backend:
                return await self.backend.search_archives(
                    query, 
                    limit=limit, 
                    progress_callback=search_progress
                )
            else:
                print(f"🔍 Searching archives for relevant context...")
                return self.backend.search_archives(query, limit=limit)
        except Exception as e:
            print(f"⚠️ Error searching archives: {e}")
            return []
    
    async def chat(self, user_message: str) -> str:
        """Send a message and get a response with async backend support."""
        
        # Add user's message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Build messages array for API call
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # For MongoDB backend, search archives and add context if archives exist
        if (isinstance(self.backend, MongoDBBackend) or 
            (hasattr(self.backend, 'sync_backend') and isinstance(self.backend.sync_backend, MongoDBBackend))) and self.archive_count > 0:
            
            # Search for relevant archived conversations
            search_results = await self.search_archives_with_progress(user_message)
            
            if search_results:
                print(f"🔍 Found {len(search_results)} relevant archive(s)")
                # Add retrieved context to system prompt
                context_addon = "\n\nRelevant conversation history from your memory:\n"
                for result in search_results:
                    context_addon += f"\n--- Archive #{result.get('archive_id', 'unknown')} (Score: {result.get('score', 0):.3f}) ---\n"
                    # Add messages from the archive
                    for msg in result.get('messages', []):
                        role = "User" if msg["role"] == "user" else "Assistant"
                        context_addon += f"{role}: {msg['content']}\n"
                
                # Update system prompt with context
                enhanced_system_prompt = SYSTEM_PROMPT + context_addon
                messages = [{"role": "system", "content": enhanced_system_prompt}]
        
        # Add current conversation
        messages.extend(self.conversation_history)
        
        # Make API call - backend-specific handling
        if isinstance(self.backend, OpenAIBackend) and self.store_id and self.archive_count > 0:
            # OpenAI backend uses file_search tool
            response = self.client.responses.create(
                model=self.config.chat_model,
                input=messages,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [self.backend.vector_store_id]
                }]
            )
        else:
            # MongoDB backend or no archives - use standard completion
            response = self.client.responses.create(
                model=self.config.chat_model,
                input=messages
            )
        
        # Extract the response
        assistant_message = response.output_text
        
        # Add assistant's response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        # Check if we need to archive with progress tracking
        await self.archive_messages_with_progress()
        
        return assistant_message
    
    async def get_status(self):
        """Get conversation status with async backend support."""
        if self._is_async_backend:
            backend_status = await self.backend.get_status()
        else:
            backend_status = self.backend.get_status()
        
        return {
            "active_messages": len(self.conversation_history),
            "archived_files": backend_status.get("archived_files", 0),
            "consolidations": backend_status.get("consolidations", 0),
            "total_messages": len(self.conversation_history) + (self.archive_count * ARCHIVE_COUNT),
            "consolidated_size_mb": backend_status.get("consolidated_size_mb", 0),
            "backend_type": backend_status.get("backend_type", "unknown"),
            "store_id": self.store_id,
            "is_async": self._is_async_backend
        }
    
    async def close(self):
        """Close backend connections."""
        if self._is_async_backend and hasattr(self.backend, 'close'):
            await self.backend.close()
        elif hasattr(self.backend, 'close'):
            self.backend.close()


async def async_main():
    """Async version of main function for async backends."""
    print("=" * 60)
    print("🚀 Infinite Memory Chat - Async Edition")
    print("=" * 60)
    
    # Load configuration from environment
    config = BackendConfig.from_env()
    
    # Display configuration
    print(f"Configuration:")
    print(f"  • Vector Backend: {config.backend_type.value.upper()}")
    print(f"  • Chat Model: {config.chat_model}")
    print(f"  • Embedding Model: {config.embedding_model}")
    print(f"  • Max {MAX_MESSAGES} active messages")
    print(f"  • Archive {ARCHIVE_COUNT} messages at a time")
    print(f"  • Consolidate at {MAX_ARCHIVE_FILES} files ({CONSOLIDATION_COUNT}→1)")
    print(f"  • Max file size: {MAX_FILE_SIZE_MB} MB")
    
    if is_async_backend(config):
        print(f"  • ⚡ Async operations enabled with progress tracking")
    else:
        print(f"  • 🔄 Sync backend (async wrapper available)")
    
    # Display backend-specific info
    if config.backend_type in [BackendType.MONGODB, BackendType.ASYNC_MONGODB]:
        db_name = config.mongodb_database
        # Mask connection string for security
        conn_str = config.mongodb_connection_string
        if conn_str:
            if '@' in conn_str:
                # Hide credentials: mongodb+srv://user:pass@host -> mongodb+srv://***@host
                masked_conn = conn_str.split('@')[0].split('//')[0] + '//***@' + conn_str.split('@')[1]
            else:
                masked_conn = conn_str
            print(f"  • MongoDB Database: {db_name}")
            print(f"  • Connection: {masked_conn}")
    
    print("Type 'quit' to exit, 'status' for statistics")
    print("=" * 60 + "\n")
    
    # Initialize async chat with configuration
    chat = None
    try:
        print("🔧 Initializing async backend...")
        chat = AsyncInfiniteMemoryChat(config=config)
        await chat.initialize_backend()
        await chat.setup_vector_store()
        
        # Display backend status
        status = await chat.get_status()
        backend_name = status.get("backend_type", "unknown").upper()
        async_indicator = "🚀 ASYNC" if status.get("is_async") else "🔄 SYNC"
        print(f"✅ {backend_name} backend initialized ({async_indicator})")
        print("\n🚀 Ready to chat with async capabilities!\n")
        
    except Exception as e:
        print(f"❌ Failed to initialize backend: {e}")
        print("💡 Check your configuration in .env file or environment variables")
        return
    
    try:
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == "quit":
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == "status":
                    status = await chat.get_status()
                    
                    print(f"\n📊 System Status:")
                    print(f"   Vector Backend: {status['backend_type'].upper()} {'(ASYNC)' if status.get('is_async') else '(SYNC)'}")
                    print(f"   Active messages: {status['active_messages']}/{MAX_MESSAGES}")
                    print(f"   Archive files in vector store: {status['archived_files']}/{MAX_ARCHIVE_FILES}")
                    print(f"   Consolidations performed: {status['consolidations']}")
                    print(f"   Consolidated data: {status['consolidated_size_mb']:.1f} MB")
                    print(f"   Total messages (history): {status['total_messages']}")
                    
                    # Backend-specific status
                    if config.backend_type in [BackendType.MONGODB, BackendType.ASYNC_MONGODB]:
                        # Get detailed MongoDB status
                        if hasattr(chat.backend, 'get_status'):
                            if chat._is_async_backend:
                                backend_status = await chat.backend.get_status()
                            else:
                                backend_status = chat.backend.get_status()
                            
                            print(f"\n🍃 MongoDB Status:")
                            print(f"   Connection: {backend_status.get('connection_status', 'unknown')}")
                            print(f"   Database: {backend_status.get('database_name', 'unknown')}")
                            print(f"   Collection: {backend_status.get('collection_name', 'not initialized')}")
                            print(f"   Total documents: {backend_status.get('total_documents', 0)}")
                            print(f"   Storage size: {backend_status.get('storage_size_mb', 0):.1f} MB")
                            print(f"   Embedding model: {backend_status.get('embedding_model', 'unknown')}")
                            if status.get('is_async'):
                                print(f"   Max concurrent ops: {backend_status.get('max_concurrent_operations', 'N/A')}")
                    
                    print("")
                    continue
                
                response = await chat.chat(user_input)
                print(f"\nAssistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                
    finally:
        # Clean up async resources
        if chat:
            await chat.close()


def main():
    print("=" * 60)
    print("🧠 Infinite Memory Chat - Proof of Concept")
    print("=" * 60)
    
    # Load configuration from environment
    config = BackendConfig.from_env()
    
    # Display configuration
    print(f"Configuration:")
    print(f"  • Vector Backend: {config.backend_type.value.upper()}")
    print(f"  • Chat Model: {config.chat_model}")
    print(f"  • Embedding Model: {config.embedding_model}")
    print(f"  • Max {MAX_MESSAGES} active messages")
    print(f"  • Archive {ARCHIVE_COUNT} messages at a time")
    print(f"  • Consolidate at {MAX_ARCHIVE_FILES} files ({CONSOLIDATION_COUNT}→1)")
    print(f"  • Max file size: {MAX_FILE_SIZE_MB} MB")
    print(f"  • Theoretical max: ~{CONSOLIDATION_COUNT * MAX_FILE_SIZE_MB / 1024:.0f} GB history")
    
    # Display backend-specific info
    if config.backend_type == BackendType.MONGODB:
        db_name = config.mongodb_database
        # Mask connection string for security
        conn_str = config.mongodb_connection_string
        if conn_str:
            if '@' in conn_str:
                # Hide credentials: mongodb+srv://user:pass@host -> mongodb+srv://***@host
                masked_conn = conn_str.split('@')[0].split('//')[0] + '//***@' + conn_str.split('@')[1]
            else:
                masked_conn = conn_str
            print(f"  • MongoDB Database: {db_name}")
            print(f"  • Connection: {masked_conn}")
    
    print("Type 'quit' to exit, 'status' for statistics")
    print("=" * 60 + "\n")
    
    # Initialize chat with configuration
    try:
        print("🔧 Initializing backend...")
        chat = InfiniteMemoryChat(config=config)
        chat.setup_vector_store()
        
        # Display backend status
        backend_status = chat.backend.get_status()
        backend_name = backend_status.get("backend_type", "unknown").upper()
        connection_status = backend_status.get("connection_status", "unknown")
        print(f"✅ {backend_name} backend initialized ({connection_status})")
        print("\n🚀 Ready to chat!\n")
        
    except Exception as e:
        print(f"❌ Failed to initialize backend: {e}")
        print("💡 Check your configuration in .env file or environment variables")
        return
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == "status":
                status = chat.get_status()
                backend_status = chat.backend.get_status()
                
                print(f"\n📊 System Status:")
                print(f"   Vector Backend: {status['backend_type'].upper()}")
                print(f"   Active messages: {status['active_messages']}/{MAX_MESSAGES}")
                print(f"   Archive files in vector store: {status['archived_files']}/{MAX_ARCHIVE_FILES}")
                print(f"   Consolidations performed: {status['consolidations']}")
                print(f"   Consolidated data: {status['consolidated_size_mb']:.1f} MB / {CONSOLIDATION_COUNT * MAX_FILE_SIZE_MB / 1024:.0f} GB theoretical max")
                print(f"   Total messages (history): {status['total_messages']}")
                
                # Backend-specific status
                if config.backend_type == BackendType.MONGODB:
                    print(f"\n🍃 MongoDB Status:")
                    print(f"   Connection: {backend_status.get('connection_status', 'unknown')}")
                    print(f"   Database: {backend_status.get('database_name', 'unknown')}")
                    print(f"   Collection: {backend_status.get('collection_name', 'not initialized')}")
                    print(f"   Total documents: {backend_status.get('total_documents', 0)}")
                    print(f"   Storage size: {backend_status.get('storage_size_mb', 0):.1f} MB")
                    print(f"   Embedding model: {backend_status.get('embedding_model', 'unknown')}")
                elif config.backend_type == BackendType.OPENAI:
                    print(f"\n🤖 OpenAI Status:")
                    print(f"   Vector store ID: {status.get('store_id', 'not initialized')}")
                    print(f"   Chat model: {config.chat_model}")
                    print(f"   Embedding model: {config.embedding_model}")
                
                print("")
                continue
            
            response = chat.chat(user_input)
            print(f"\nAssistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    import sys
    
    # Check for async mode
    if "--async" in sys.argv:
        asyncio.run(async_main())
    else:
        # Show mode selection
        config = BackendConfig.from_env()
        
        if is_async_backend(config):
            print("🚀 Async backend detected! Choose your experience:")
            print("  1. Async mode (with progress tracking): python infinite_memory_chat.py --async")
            print("  2. Sync mode (backward compatibility): python infinite_memory_chat.py")
            print()
            
            # Default to async for async backends
            if len(sys.argv) == 1:  # No arguments
                print("🚀 Starting in async mode (recommended for this backend type)...")
                print("💡 Use 'python infinite_memory_chat.py --sync' to force sync mode")
                print()
                asyncio.run(async_main())
            else:
                main()
        else:
            main()
