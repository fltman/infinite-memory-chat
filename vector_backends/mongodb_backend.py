"""
MongoDB Vector Search backend implementation with reliability and performance improvements.
"""

import os
import json
import struct
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError, OperationFailure, ServerSelectionTimeoutError, AutoReconnect
from bson import Binary
from openai import OpenAI, RateLimitError, APIError
import logging

from .base import VectorBackend
from input_validator import input_validator, security_audit_logger
from connection_monitor import MongoDBConnectionMonitor, connection_health_manager
from error_recovery import error_recovery_manager, graceful_degradation_manager
from rate_limit_manager import DEFAULT_CONFIGS, AdaptiveRateLimitManager

# Configure logging
logger = logging.getLogger("infinite_memory_chat")


class MongoDBBackend(VectorBackend):
    """MongoDB Vector Search backend implementation with enhanced reliability."""
    
    def __init__(self, 
                 connection_string: str, 
                 database_name: str = "infinite_memory_chat", 
                 embedding_client: Optional[OpenAI] = None,
                 embedding_model: str = "text-embedding-3-small"):
        self.connection_string = connection_string
        self.database_name = database_name
        self.embedding_model = embedding_model
        self.embedding_client = embedding_client or OpenAI()
        
        # Connection will be established lazily
        self.client: Optional[MongoClient] = None
        self.db = None
        self.collection_name = None
        self.collection: Optional[Collection] = None
        
        # State tracking
        self.archive_count = 0
        self.consolidated_archives = []
        
        # Initialize reliability features
        self.connection_monitor = None
        self.rate_limit_manager = AdaptiveRateLimitManager(DEFAULT_CONFIGS["moderate"])
        
        logger.info(f"MongoDB backend initialized with reliability features")
        
    def _connect(self):
        """Establish MongoDB connection if not already connected."""
        if self.client is None:
            try:
                self.client = MongoClient(self.connection_string)
                self.db = self.client[self.database_name]
                # Test the connection
                self.client.admin.command('hello')
                print(f"✅ Connected to MongoDB: {self.database_name}")
            except PyMongoError as e:
                print(f"❌ Failed to connect to MongoDB: {e}")
                raise
                
    def setup_store(self, session_id: str) -> str:
        """Create MongoDB collection and set up vector search index."""
        self._connect()
        
        print("📦 Setting up MongoDB collection and vector index...")
        
        self.collection_name = f"chat_memory_{session_id}"
        self.collection = self.db[self.collection_name]
        
        # Create the collection explicitly if it doesn't exist
        if self.collection_name not in self.db.list_collection_names():
            self.db.create_collection(self.collection_name)
        
        # Note: Vector search indexes may need to be created manually in MongoDB Atlas UI
        # or via the Atlas Admin API. For production, this would be automated.
        # For now, we'll document the required index structure.
        
        print(f"✅ MongoDB collection created: {self.collection_name}")
        print("ℹ️  Vector search index required (create manually in MongoDB CE 8.2.x or Atlas):")
        print("   Index name: vector_index")
        print("   Field: content_embedding (vector, 1536 dimensions, cosine)")
        print("   Filter fields: session_id, archive_id, type")
        
        return self.collection_name
    
    def archive_messages(self, messages: List[Dict], archive_id: int, session_id: str) -> bool:
        """Archive messages to MongoDB with vector embeddings and enhanced error handling."""
        if self.collection is None:
            error_info = error_recovery_manager.handle_error(
                ValueError("MongoDB collection not initialized"),
                {"operation": "archive_messages", "archive_id": archive_id}
            )
            print(f"❌ {error_info.user_message}")
            return False
            
        print(f"📁 Archiving {len(messages)} messages to MongoDB...")
        
        try:
            # Validate inputs first
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
            
            # Create summary text for embedding
            summary_text = f"Conversation archive #{archive_id}\n"
            summary_text += f"Timestamp: {datetime.now().isoformat()}\n"
            summary_text += f"Session: {session_id}\n"
            summary_text += f"Content:\n"
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                summary_text += f"\n{role}: {msg['content']}\n"
            
            # Generate embedding using OpenAI with rate limiting
            print(f"🧠 Generating embedding for archive #{archive_id}...")
            
            # Wait for rate limiting if needed
            wait_time = self.rate_limit_manager.wait_if_needed(estimated_tokens=len(summary_text) // 4)
            if wait_time > 0.1:
                print(f"⏱️ Waiting {wait_time:.1f}s for rate limiting...")
            
            start_time = time.time()
            try:
                embedding_response = self.embedding_client.embeddings.create(
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
            
            # Validate document before insertion
            try:
                validated_doc = input_validator.validate_mongodb_document(archive_doc)
            except Exception as doc_error:
                error_info = error_recovery_manager.handle_error(
                    doc_error,
                    {"operation": "document_validation", "archive_id": archive_id}
                )
                print(f"❌ {error_info.user_message}")
                return False
            
            # Insert document with MongoDB error handling
            try:
                result = self.collection.insert_one(validated_doc)
                self.archive_count += 1
                
                print(f"✅ Archive #{archive_id} saved to MongoDB (ID: {result.inserted_id})")
                logger.info(f"Successfully archived {len(messages)} messages for archive #{archive_id}")
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
            logger.error(f"Unexpected error in archive_messages: {unexpected_error}", exc_info=True)
            return False
    
    def search_archives(self, query_text: str, limit: int = 10) -> List[Dict]:
        """Search archived messages using MongoDB vector search."""
        if self.collection is None:
            print("❌ MongoDB collection not initialized")
            return []
        
        try:
            print(f"🔍 Searching archives for: '{query_text[:50]}...'")
            
            # Generate query embedding
            embedding_response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=query_text
            )
            query_vector = embedding_response.data[0].embedding
            
            # MongoDB vector search aggregation pipeline
            # Note: This requires a vector search index to be created in Atlas
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
            
            # Execute the search
            results = list(self.collection.aggregate(pipeline))
            
            print(f"✅ Found {len(results)} relevant archives")
            
            return results
            
        except OperationFailure as e:
            if "vector search" in str(e).lower():
                print(f"⚠️  Vector search index not found. Create 'vector_index' in MongoDB Atlas")
                print(f"   Error: {e}")
            else:
                print(f"❌ MongoDB operation failed: {e}")
            return []
        except Exception as e:
            print(f"❌ Error searching MongoDB: {e}")
            return []
    
    def consolidate_archives(self, archive_ids: List[str]) -> Optional[str]:
        """
        Consolidate multiple archives into a single document.
        Unlike file-based consolidation, this merges documents in the database.
        """
        if self.collection is None or not archive_ids:
            return None
            
        try:
            print(f"📦 Consolidating {len(archive_ids)} archives in MongoDB...")
            
            # Fetch archives to consolidate
            archives_to_merge = list(self.collection.find({
                "archive_id": {"$in": [int(aid) for aid in archive_ids if aid.isdigit()]},
                "type": "archive"
            }))
            
            if not archives_to_merge:
                print("⚠️  No archives found to consolidate")
                return None
            
            # Create consolidated document
            consolidated_id = f"consolidated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Collect all messages and create combined content
            all_messages = []
            combined_content = f"Consolidated Archive: {consolidated_id}\n"
            combined_content += f"Original Archives: {[a['archive_id'] for a in archives_to_merge]}\n"
            combined_content += f"Timestamp: {datetime.now().isoformat()}\n"
            combined_content += f"Content:\n"
            
            for archive in archives_to_merge:
                all_messages.extend(archive.get('messages', []))
                combined_content += f"\n--- Archive #{archive['archive_id']} ---\n"
                for msg in archive.get('messages', []):
                    role = "User" if msg["role"] == "user" else "Assistant"
                    combined_content += f"{role}: {msg['content']}\n"
            
            # Generate new embedding for consolidated content
            print(f"🧠 Generating embedding for consolidated archive...")
            embedding_response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=combined_content
            )
            consolidated_vector = embedding_response.data[0].embedding
            consolidated_bindata = self._vector_to_bindata(consolidated_vector)
            
            # Create consolidated document
            consolidated_doc = {
                "_id": consolidated_id,
                "session_id": archives_to_merge[0]['session_id'],
                "archive_id": consolidated_id,
                "timestamp": datetime.now(),
                "messages": all_messages,
                "content_text": combined_content,
                "content_embedding": consolidated_bindata,
                "content_embedding_array": consolidated_vector,
                "message_count": len(all_messages),
                "type": "consolidated",
                "original_archive_ids": [a['archive_id'] for a in archives_to_merge],
                "consolidation_count": len(archives_to_merge),
                "embedding_model": self.embedding_model,
                "created_at": datetime.now()
            }
            
            # Insert consolidated document
            result = self.collection.insert_one(consolidated_doc)
            
            # Remove original archives
            delete_result = self.collection.delete_many({
                "archive_id": {"$in": [int(aid) for aid in archive_ids if aid.isdigit()]},
                "type": "archive"
            })
            
            print(f"✅ Consolidated {len(archives_to_merge)} archives into {consolidated_id}")
            print(f"   Deleted {delete_result.deleted_count} original archives")
            
            self.consolidated_archives.append(consolidated_id)
            return consolidated_id
            
        except Exception as e:
            print(f"❌ Error during consolidation: {e}")
            return None
    
    def get_archive_count(self) -> int:
        """Get the number of archived message groups."""
        if self.collection is None:
            return 0
            
        try:
            count = self.collection.count_documents({"type": {"$in": ["archive", "consolidated"]}})
            return count
        except Exception as e:
            print(f"❌ Error getting archive count: {e}")
            return 0
    
    def cleanup_archives(self, archive_ids: List[str]) -> bool:
        """Remove specified archives from MongoDB."""
        if self.collection is None:
            return False
            
        try:
            # Convert string IDs to appropriate types
            numeric_ids = [int(aid) for aid in archive_ids if aid.isdigit()]
            string_ids = [aid for aid in archive_ids if not aid.isdigit()]
            
            # Delete documents
            result = self.collection.delete_many({
                "$or": [
                    {"archive_id": {"$in": numeric_ids}},
                    {"archive_id": {"$in": string_ids}}
                ]
            })
            
            print(f"✅ Deleted {result.deleted_count} archives from MongoDB")
            return result.deleted_count > 0
            
        except Exception as e:
            print(f"❌ Error cleaning up archives: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get backend-specific status information."""
        status = {
            "backend_type": "mongodb",
            "database_name": self.database_name,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "archived_files": 0,  # For compatibility with OpenAI backend
            "consolidations": len(self.consolidated_archives),
            "consolidated_size_mb": 0,  # MongoDB doesn't have direct file sizes
            "connection_status": "disconnected"
        }

        if self.collection is not None:
            try:
                # Get document counts by type
                archive_count = self.collection.count_documents({"type": "archive"})
                consolidated_count = self.collection.count_documents({"type": "consolidated"})
                
                # Get collection stats (approximate size)
                stats = self.db.command("collStats", self.collection_name)
                collection_size_mb = stats.get("size", 0) / (1024 * 1024)
                
                status.update({
                    "archived_files": archive_count,
                    "consolidated_archives": consolidated_count,
                    "total_documents": archive_count + consolidated_count,
                    "consolidated_size_mb": collection_size_mb,
                    "connection_status": "connected",
                    "index_count": len(stats.get("indexSizes", {})),
                    "storage_size_mb": stats.get("storageSize", 0) / (1024 * 1024)
                })
                
            except Exception as e:
                print(f"⚠️  Error getting detailed status: {e}")
        
        return status
    
    def should_consolidate(self, max_archives: int) -> bool:
        """Check if consolidation should be triggered."""
        if self.collection is None:
            return False
            
        try:
            archive_count = self.collection.count_documents({"type": "archive"})
            return archive_count >= max_archives
        except Exception:
            return False
    
    def _vector_to_bindata(self, vector: List[float]) -> Binary:
        """Convert float vector to MongoDB BinData for efficient storage."""
        try:
            # Pack as 32-bit floats
            byte_data = struct.pack(f'{len(vector)}f', *vector)
            return Binary(byte_data, subtype=4)  # float32 subtype
        except Exception as e:
            print(f"⚠️  Error converting vector to BinData: {e}")
            # Fallback to array storage
            return vector
    
    def _bindata_to_vector(self, bindata: Binary) -> List[float]:
        """Convert MongoDB BinData back to float vector."""
        try:
            if isinstance(bindata, Binary):
                # Unpack from bytes
                float_count = len(bindata) // 4  # 4 bytes per float
                return list(struct.unpack(f'{float_count}f', bindata))
            else:
                # Already an array
                return bindata
        except Exception as e:
            print(f"⚠️  Error converting BinData to vector: {e}")
            return []
    
    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.collection = None
            print("✅ MongoDB connection closed")
    
    def __del__(self):
        """Ensure connection is closed when object is destroyed."""
        self.close()
