"""
Async MongoDB Vector Search backend implementation with high-performance async operations.
"""

import os
import json
import struct
import time
import asyncio
from typing import List, Dict, Any, Optional, Callable, Awaitable
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo.errors import PyMongoError, OperationFailure, ServerSelectionTimeoutError, AutoReconnect
from bson import Binary
from openai import AsyncOpenAI, RateLimitError, APIError
import logging

from .async_base import AsyncVectorBackend, AsyncProgressTracker, AsyncBatchProcessor
from input_validator import input_validator, security_audit_logger
from connection_monitor import MongoDBConnectionMonitor, connection_health_manager
from error_recovery import error_recovery_manager, graceful_degradation_manager
from rate_limit_manager import DEFAULT_CONFIGS, AdaptiveRateLimitManager

# Configure logging
logger = logging.getLogger("infinite_memory_chat")


class AsyncMongoDBBackend(AsyncVectorBackend):
    """High-performance async MongoDB Vector Search backend with full reliability features."""
    
    def __init__(self, 
                 connection_string: str, 
                 database_name: str = "infinite_memory_chat", 
                 embedding_client: Optional[AsyncOpenAI] = None,
                 embedding_model: str = "text-embedding-3-small",
                 max_concurrent_operations: int = 10):
        self.connection_string = connection_string
        self.database_name = database_name
        self.embedding_model = embedding_model
        self.embedding_client = embedding_client or AsyncOpenAI()
        self.max_concurrent_operations = max_concurrent_operations
        
        # Async connection will be established lazily
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.collection_name = None
        self.collection: Optional[AsyncIOMotorCollection] = None
        
        # State tracking
        self.archive_count = 0
        self.consolidated_archives = []
        
        # Initialize reliability features
        self.connection_monitor = None
        self.rate_limit_manager = AdaptiveRateLimitManager(DEFAULT_CONFIGS["moderate"])
        
        # Async utilities
        self.batch_processor = AsyncBatchProcessor(
            batch_size=5, 
            max_concurrent=max_concurrent_operations
        )
        
        # Connection semaphore to limit concurrent MongoDB operations
        self._mongo_semaphore = asyncio.Semaphore(max_concurrent_operations)
        
        logger.info(f"Async MongoDB backend initialized with {max_concurrent_operations} max concurrent operations")
        
    async def _connect(self):
        """Establish async MongoDB connection if not already connected."""
        if self.client is None:
            try:
                self.client = AsyncIOMotorClient(self.connection_string)
                self.db = self.client[self.database_name]
                # Test the connection
                await self.client.admin.command('hello')
                print(f"✅ Connected to MongoDB (async): {self.database_name}")
            except PyMongoError as e:
                print(f"❌ Failed to connect to MongoDB: {e}")
                raise
                
    async def setup_store(self, session_id: str) -> str:
        """Create MongoDB collection and set up vector search index."""
        await self._connect()
        
        print("📦 Setting up MongoDB collection and vector index (async)...")
        
        self.collection_name = f"chat_memory_{session_id}"
        self.collection = self.db[self.collection_name]
        
        # Create the collection explicitly if it doesn't exist
        collection_names = await self.db.list_collection_names()
        if self.collection_name not in collection_names:
            await self.db.create_collection(self.collection_name)
        
        print(f"✅ MongoDB collection created (async): {self.collection_name}")
        print("ℹ️  Vector search index required (create manually in MongoDB Atlas):")
        print("   Index name: vector_index")
        print("   Field: content_embedding (vector, 1536 dimensions, cosine)")
        print("   Filter fields: session_id, archive_id, type")
        
        return self.collection_name
    
    async def archive_messages(self, 
                              messages: List[Dict], 
                              archive_id: int, 
                              session_id: str,
                              progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None) -> bool:
        """Archive messages to MongoDB with async operations and comprehensive error handling."""
        if self.collection is None:
            error_info = error_recovery_manager.handle_error(
                ValueError("MongoDB collection not initialized"),
                {"operation": "archive_messages", "archive_id": archive_id}
            )
            print(f"❌ {error_info.user_message}")
            return False
            
        print(f"📁 Archiving {len(messages)} messages to MongoDB (async)...")
        
        # Initialize progress tracker
        progress = AsyncProgressTracker(6, progress_callback)  # 6 major steps
        
        try:
            # Step 1: Input validation
            await progress.update("Validating inputs...")
            try:
                session_id = input_validator.validate_session_id(session_id)
                
                # Validate messages
                for i, msg in enumerate(messages):
                    if not isinstance(msg, dict) or 'content' not in msg:
                        raise ValueError(f"Invalid message format at index {i}")
                    msg['content'] = input_validator.validate_user_message(msg['content'])
                    
            except Exception as validation_error:
                error_info = error_recovery_manager.handle_error(
                    validation_error,
                    {"operation": "archive_validation", "archive_id": archive_id}
                )
                print(f"❌ {error_info.user_message}")
                return False
            
            # Step 2: Create summary text for embedding
            await progress.update("Preparing content for embedding...")
            summary_text = f"Conversation archive #{archive_id}\n"
            summary_text += f"Timestamp: {datetime.now().isoformat()}\n"
            summary_text += f"Session: {session_id}\n"
            summary_text += f"Content:\n"
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                summary_text += f"\n{role}: {msg['content']}\n"
            
            # Step 3: Generate embedding using OpenAI with async and rate limiting
            await progress.update("Generating embedding...")
            print(f"🧠 Generating embedding for archive #{archive_id} (async)...")
            
            # Wait for rate limiting if needed
            wait_time = self.rate_limit_manager.wait_if_needed(estimated_tokens=len(summary_text) // 4)
            if wait_time > 0.1:
                print(f"⏱️ Waiting {wait_time:.1f}s for rate limiting...")
                await asyncio.sleep(wait_time)
            
            start_time = time.time()
            try:
                # Use async OpenAI client
                embedding_response = await self.embedding_client.embeddings.create(
                    model=self.embedding_model,
                    input=summary_text
                )
                
                # Validate API response
                validated_response = input_validator.validate_openai_response(embedding_response.model_dump())
                embedding_vector = validated_response['data'][0]['embedding']
                
                # Record successful API request
                response_time = (time.time() - start_time) * 1000
                self.rate_limit_manager.record_request(
                    tokens_used=len(summary_text) // 4,  # Rough estimate
                    response_time_ms=response_time,
                    was_rate_limited=False,
                    error_occurred=False
                )
                
            except (RateLimitError, APIError) as api_error:
                response_time = (time.time() - start_time) * 1000
                was_rate_limited = isinstance(api_error, RateLimitError)
                
                # Record failed API request
                self.rate_limit_manager.record_request(
                    tokens_used=0,
                    response_time_ms=response_time,
                    was_rate_limited=was_rate_limited,
                    error_occurred=True
                )
                
                # Handle the error with recovery
                error_info = error_recovery_manager.handle_error(
                    api_error,
                    {"operation": "embedding_generation", "archive_id": archive_id}
                )
                print(f"❌ {error_info.user_message}")
                print(f"💡 {error_info.suggestion}")
                return False
            
            # Step 4: Prepare document
            await progress.update("Preparing document for storage...")
            
            # Convert to BinData for efficient storage
            embedding_bindata = self._vector_to_bindata(embedding_vector)
            
            # Create document with comprehensive metadata
            archive_doc = {
                "_id": f"{session_id}_{archive_id}",
                "session_id": session_id,
                "archive_id": archive_id,
                "timestamp": datetime.now(),
                "messages": messages,
                "content_text": summary_text,
                "content_embedding": embedding_bindata,  # BinData format
                "content_embedding_array": embedding_vector,  # Array format for compatibility
                "message_count": len(messages),
                "type": "archive",
                "embedding_model": self.embedding_model,
                "created_at": datetime.now()
            }
            
            # Step 5: Document validation
            await progress.update("Validating document...")
            try:
                validated_doc = input_validator.validate_mongodb_document(archive_doc)
            except Exception as doc_error:
                error_info = error_recovery_manager.handle_error(
                    doc_error,
                    {"operation": "document_validation", "archive_id": archive_id}
                )
                print(f"❌ {error_info.user_message}")
                return False
            
            # Step 6: Insert document with async MongoDB operation
            await progress.update("Saving to database...")
            try:
                async with self._mongo_semaphore:  # Control concurrent MongoDB operations
                    result = await self.collection.insert_one(validated_doc)
                self.archive_count += 1
                
                await progress.complete(f"Archive #{archive_id} saved successfully")
                print(f"✅ Archive #{archive_id} saved to MongoDB (async) (ID: {result.inserted_id})")
                logger.info(f"Successfully archived {len(messages)} messages for archive #{archive_id} (async)")
                return True
                
            except (PyMongoError, ServerSelectionTimeoutError, AutoReconnect) as mongo_error:
                error_info = error_recovery_manager.handle_error(
                    mongo_error,
                    {"operation": "mongodb_insert", "archive_id": archive_id}
                )
                print(f"❌ {error_info.user_message}")
                print(f"💡 {error_info.suggestion}")
                
                # Trigger graceful degradation if needed
                if isinstance(mongo_error, ServerSelectionTimeoutError):
                    graceful_degradation_manager.degrade_to_mode(
                        "limited_memory", 
                        f"MongoDB connection timeout: {mongo_error}"
                    )
                
                return False
            
        except Exception as unexpected_error:
            error_info = error_recovery_manager.handle_error(
                unexpected_error,
                {"operation": "archive_messages", "archive_id": archive_id}
            )
            print(f"❌ {error_info.user_message}")
            logger.error(f"Unexpected error in async archive_messages: {unexpected_error}", exc_info=True)
            return False
    
    async def search_archives(self, 
                             query_text: str, 
                             limit: int = 10,
                             progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None) -> List[Dict]:
        """Search archived messages using async MongoDB vector search."""
        if self.collection is None:
            print("❌ MongoDB collection not initialized")
            return []
        
        progress = AsyncProgressTracker(4, progress_callback)
        
        try:
            await progress.update("Preparing search query...")
            print(f"🔍 Searching archives for: '{query_text[:50]}...' (async)")
            
            # Step 1: Generate query embedding
            await progress.update("Generating query embedding...")
            start_time = time.time()
            
            # Use async OpenAI client with rate limiting
            wait_time = self.rate_limit_manager.wait_if_needed(estimated_tokens=len(query_text) // 4)
            if wait_time > 0.1:
                await asyncio.sleep(wait_time)
            
            embedding_response = await self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=query_text
            )
            query_vector = embedding_response.data[0].embedding
            
            # Record API usage
            response_time = (time.time() - start_time) * 1000
            self.rate_limit_manager.record_request(
                tokens_used=len(query_text) // 4,
                response_time_ms=response_time,
                was_rate_limited=False,
                error_occurred=False
            )
            
            # Step 2: Prepare search pipeline
            await progress.update("Preparing search pipeline...")
            # MongoDB vector search aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",  # Must match the index name in Atlas
                        "path": "content_embedding",
                        "queryVector": query_vector,
                        "numCandidates": limit * 20,  # 20x limit for better recall
                        "limit": limit,
                        "filter": {"type": "archive"}
                    }
                },
                {
                    "$project": {
                        "messages": 1,
                        "archive_id": 1,
                        "timestamp": 1,
                        "session_id": 1,
                        "content_text": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            # Step 3: Execute search
            await progress.update("Executing vector search...")
            async with self._mongo_semaphore:  # Control concurrent operations
                cursor = self.collection.aggregate(pipeline)
                results = await cursor.to_list(length=limit)
            
            await progress.complete(f"Found {len(results)} relevant archives")
            print(f"✅ Found {len(results)} relevant archives (async)")
            
            return results
            
        except OperationFailure as e:
            if "vector search" in str(e).lower():
                print(f"⚠️  Vector search index not found. Create 'vector_index' in MongoDB Atlas")
                print(f"   Error: {e}")
            else:
                print(f"❌ MongoDB operation failed: {e}")
            return []
        
        except Exception as e:
            error_info = error_recovery_manager.handle_error(
                e, {"operation": "search_archives", "query": query_text[:100]}
            )
            print(f"❌ {error_info.user_message}")
            logger.error(f"Error in async search_archives: {e}", exc_info=True)
            return []
    
    async def consolidate_archives(self, 
                                  archive_ids: List[str],
                                  progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None) -> Optional[str]:
        """Consolidate multiple archives into a single unit with async operations."""
        if self.collection is None:
            print("❌ MongoDB collection not initialized")
            return None
        
        if not archive_ids:
            print("⚠️  No archive IDs provided for consolidation")
            return None
        
        progress = AsyncProgressTracker(5, progress_callback)
        consolidated_id = f"consolidated_{int(time.time())}"
        
        try:
            await progress.update(f"Starting consolidation of {len(archive_ids)} archives...")
            print(f"🔄 Consolidating {len(archive_ids)} archives (async)...")
            
            # Step 1: Fetch all archives to consolidate
            await progress.update("Fetching archives...")
            async with self._mongo_semaphore:
                cursor = self.collection.find({
                    "$or": [
                        {"_id": {"$in": archive_ids}},
                        {"archive_id": {"$in": [int(id_) if id_.isdigit() else id_ for id_ in archive_ids]}}
                    ]
                })
                archives_to_consolidate = await cursor.to_list(length=None)
            
            if not archives_to_consolidate:
                print("⚠️  No matching archives found for consolidation")
                return None
            
            # Step 2: Combine all messages
            await progress.update("Combining messages...")
            all_messages = []
            earliest_timestamp = None
            session_ids = set()
            
            for archive in archives_to_consolidate:
                all_messages.extend(archive.get('messages', []))
                session_ids.add(archive.get('session_id', 'unknown'))
                
                archive_timestamp = archive.get('timestamp')
                if earliest_timestamp is None or (archive_timestamp and archive_timestamp < earliest_timestamp):
                    earliest_timestamp = archive_timestamp
            
            # Step 3: Generate consolidated content and embedding
            await progress.update("Generating consolidated embedding...")
            consolidated_text = f"Consolidated Archive: {consolidated_id}\n"
            consolidated_text += f"Original Archives: {', '.join(map(str, archive_ids))}\n"
            consolidated_text += f"Sessions: {', '.join(session_ids)}\n"
            consolidated_text += f"Total Messages: {len(all_messages)}\n"
            consolidated_text += f"Consolidation Date: {datetime.now().isoformat()}\n\n"
            
            for i, msg in enumerate(all_messages):
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get('content', '')
                consolidated_text += f"\n[{i+1}] {role}: {content}\n"
            
            # Generate embedding for consolidated content
            start_time = time.time()
            wait_time = self.rate_limit_manager.wait_if_needed(estimated_tokens=len(consolidated_text) // 4)
            if wait_time > 0.1:
                await asyncio.sleep(wait_time)
            
            embedding_response = await self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=consolidated_text
            )
            consolidated_embedding = embedding_response.data[0].embedding
            
            response_time = (time.time() - start_time) * 1000
            self.rate_limit_manager.record_request(
                tokens_used=len(consolidated_text) // 4,
                response_time_ms=response_time,
                was_rate_limited=False,
                error_occurred=False
            )
            
            # Step 4: Create consolidated document
            await progress.update("Creating consolidated document...")
            consolidated_doc = {
                "_id": consolidated_id,
                "session_id": list(session_ids)[0] if len(session_ids) == 1 else "multi_session",
                "archive_id": consolidated_id,
                "timestamp": earliest_timestamp or datetime.now(),
                "messages": all_messages,
                "content_text": consolidated_text,
                "content_embedding": self._vector_to_bindata(consolidated_embedding),
                "content_embedding_array": consolidated_embedding,
                "message_count": len(all_messages),
                "type": "consolidated",
                "embedding_model": self.embedding_model,
                "original_archive_ids": archive_ids,
                "consolidated_at": datetime.now(),
                "created_at": datetime.now()
            }
            
            # Validate and insert consolidated document
            validated_doc = input_validator.validate_mongodb_document(consolidated_doc)
            
            # Step 5: Save consolidated and clean up originals
            await progress.update("Saving consolidated archive...")
            async with self._mongo_semaphore:
                # Insert consolidated document
                await self.collection.insert_one(validated_doc)
                
                # Delete original archives
                await self.collection.delete_many({
                    "$or": [
                        {"_id": {"$in": archive_ids}},
                        {"archive_id": {"$in": [int(id_) if id_.isdigit() else id_ for id_ in archive_ids]}}
                    ]
                })
            
            self.consolidated_archives.append(consolidated_id)
            
            await progress.complete(f"Consolidation complete: {consolidated_id}")
            print(f"✅ Consolidated {len(archives_to_consolidate)} archives into {consolidated_id} (async)")
            logger.info(f"Successfully consolidated archives {archive_ids} into {consolidated_id}")
            
            return consolidated_id
            
        except Exception as e:
            error_info = error_recovery_manager.handle_error(
                e, {"operation": "consolidate_archives", "archive_ids": archive_ids}
            )
            print(f"❌ {error_info.user_message}")
            logger.error(f"Error in async consolidate_archives: {e}", exc_info=True)
            return None
    
    async def get_archive_count(self) -> int:
        """Get the number of archived message groups."""
        if self.collection is None:
            return 0
        
        try:
            async with self._mongo_semaphore:
                count = await self.collection.count_documents({
                    "type": {"$in": ["archive", "consolidated"]}
                })
            return count
        except Exception as e:
            logger.error(f"Error getting archive count: {e}")
            return 0
    
    async def cleanup_archives(self, archive_ids: List[str]) -> bool:
        """Remove specified archives from storage."""
        if self.collection is None or not archive_ids:
            return False
        
        try:
            async with self._mongo_semaphore:
                result = await self.collection.delete_many({
                    "$or": [
                        {"_id": {"$in": archive_ids}},
                        {"archive_id": {"$in": archive_ids}}
                    ]
                })
            
            deleted_count = result.deleted_count
            print(f"✅ Deleted {deleted_count} archives (async)")
            logger.info(f"Cleaned up {deleted_count} archives: {archive_ids}")
            return deleted_count > 0
            
        except Exception as e:
            error_info = error_recovery_manager.handle_error(
                e, {"operation": "cleanup_archives", "archive_ids": archive_ids}
            )
            print(f"❌ {error_info.user_message}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get backend-specific status information."""
        try:
            status = {
                "backend_type": "async_mongodb",
                "database_name": self.database_name,
                "collection_name": self.collection_name or "Not initialized",
                "embedding_model": self.embedding_model,
                "archived_files": self.archive_count,
                "consolidations": len(self.consolidated_archives),
                "max_concurrent_operations": self.max_concurrent_operations,
            }
            
            if self.client and self.collection:
                try:
                    # Test connection
                    await self.client.admin.command('hello')
                    status["connection_status"] = "connected"
                    
                    # Get document counts
                    async with self._mongo_semaphore:
                        total_docs = await self.collection.count_documents({})
                        archive_docs = await self.collection.count_documents({"type": "archive"})
                        consolidated_docs = await self.collection.count_documents({"type": "consolidated"})
                    
                    status.update({
                        "total_documents": total_docs,
                        "archive_documents": archive_docs,
                        "consolidated_documents": consolidated_docs,
                    })
                    
                    # Get collection stats (storage size)
                    try:
                        stats = await self.db.command("collStats", self.collection_name)
                        status["storage_size_mb"] = round(stats.get("storageSize", 0) / 1024 / 1024, 2)
                    except:
                        status["storage_size_mb"] = 0
                        
                except Exception:
                    status["connection_status"] = "disconnected"
                    status["total_documents"] = 0
                    status["storage_size_mb"] = 0
            else:
                status["connection_status"] = "not_initialized"
                status["total_documents"] = 0
                status["storage_size_mb"] = 0
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting async backend status: {e}")
            return {
                "backend_type": "async_mongodb",
                "connection_status": "error",
                "error": str(e)
            }
    
    async def should_consolidate(self, max_archives: int) -> bool:
        """Check if consolidation should be triggered."""
        try:
            current_count = await self.get_archive_count()
            return current_count >= max_archives
        except Exception:
            return False
    
    def _vector_to_bindata(self, vector: List[float]) -> Binary:
        """Convert vector to BSON Binary format for efficient storage."""
        vector_bytes = struct.pack(f'{len(vector)}f', *vector)
        return Binary(vector_bytes, subtype=0)
    
    def _bindata_to_vector(self, bindata: Binary) -> List[float]:
        """Convert BSON Binary back to vector list."""
        vector_size = len(bindata) // 4  # 4 bytes per float
        return list(struct.unpack(f'{vector_size}f', bindata))
    
    async def close(self):
        """Clean up async connections and resources."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.collection = None
            print("✅ MongoDB connection closed (async)")
        
        # Close OpenAI client if it has a close method
        if hasattr(self.embedding_client, 'close'):
            await self.embedding_client.close()
