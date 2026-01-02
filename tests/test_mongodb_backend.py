"""
Tests for MongoDB vector backend implementation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from vector_backends.mongodb_backend import MongoDBBackend
from config import BackendConfig, BackendType, create_backend


class TestMongoDBBackend:
    """Test MongoDB backend implementation."""
    
    @pytest.fixture
    def mock_mongo_client(self):
        """Mock MongoDB client for testing."""
        client = Mock()
        db = Mock()
        collection = Mock()
        
        # Mock collection methods
        collection.insert_one.return_value = Mock(inserted_id="test_doc_id")
        collection.count_documents.return_value = 5
        collection.find.return_value = []
        collection.aggregate.return_value = []
        collection.delete_many.return_value = Mock(deleted_count=3)
        
        # Mock database methods
        db.list_collection_names.return_value = []
        db.create_collection.return_value = None
        db.command.return_value = {
            "size": 1024 * 1024,  # 1MB
            "storageSize": 2 * 1024 * 1024,  # 2MB
            "indexSizes": {"_id_": 100}
        }
        # Mock database collection access (db[collection_name])
        def db_getitem(collection_name):
            return collection
        db.__getitem__ = Mock(side_effect=db_getitem)
        
        # Mock client methods
        client.admin = Mock()
        client.admin.command.return_value = {"ok": 1}  # hello command
        # Mock client database access (client[db_name])
        def client_getitem(db_name):
            return db
        client.__getitem__ = Mock(side_effect=client_getitem)
        client.close.return_value = None
        
        return client, db, collection
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for embedding generation."""
        client = Mock()
        embedding_response = Mock()
        embedding_response.data = [Mock(embedding=[0.1] * 1536)]  # Mock 1536-dim vector
        
        # Mock the model_dump() method to return a proper dictionary structure
        mock_response_dict = {
            "data": [{"embedding": [0.1] * 1536}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        }
        embedding_response.model_dump.return_value = mock_response_dict
        
        client.embeddings.create.return_value = embedding_response
        return client
    
    @pytest.fixture
    def backend(self, mock_mongo_client, mock_openai_client):
        """Create MongoDB backend with mocked dependencies."""
        mongo_client, db, collection = mock_mongo_client
        
        with patch('vector_backends.mongodb_backend.MongoClient') as mock_mongo_constructor:
            mock_mongo_constructor.return_value = mongo_client
            
            backend = MongoDBBackend(
                connection_string="mongodb://localhost:27017",
                database_name="test_db",
                embedding_client=mock_openai_client,
                embedding_model="text-embedding-3-small"
            )
            
            return backend, mongo_client, db, collection
    
    @patch('vector_backends.mongodb_backend.OpenAI')
    def test_initialization(self, mock_openai):
        """Test backend initialization."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        backend = MongoDBBackend(
            connection_string="mongodb://localhost:27017",
            database_name="test_db"
        )
        
        assert backend.connection_string == "mongodb://localhost:27017"
        assert backend.database_name == "test_db"
        assert backend.embedding_model == "text-embedding-3-small"
        assert backend.client is None  # Lazy connection
        assert backend.embedding_client == mock_client  # Should use mocked OpenAI client
    
    @patch('vector_backends.mongodb_backend.MongoClient')
    def test_setup_store(self, mock_mongo_constructor, backend, mock_openai_client):
        """Test vector store setup."""
        backend_obj, expected_mongo_client, db, collection = backend
        session_id = "test_session_123"
        
        # Make sure the MongoClient constructor returns our mocked client
        mock_mongo_constructor.return_value = expected_mongo_client
        
        store_id = backend_obj.setup_store(session_id)
        
        assert store_id == f"chat_memory_{session_id}"
        assert backend_obj.collection_name == f"chat_memory_{session_id}"
        expected_mongo_client.admin.command.assert_called_with('hello')
    
    def test_archive_messages_success(self, backend, mock_openai_client):
        """Test successful message archival."""
        backend_obj, mongo_client, db, collection = backend
        backend_obj.collection = collection
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        result = backend_obj.archive_messages(messages, 1, "test_session")
        
        assert result is True
        collection.insert_one.assert_called_once()
        mock_openai_client.embeddings.create.assert_called_once()
        
        # Check the document structure
        call_args = collection.insert_one.call_args[0][0]
        assert call_args["session_id"] == "test_session"
        assert call_args["archive_id"] == 1
        assert call_args["messages"] == messages
        assert call_args["type"] == "archive"
    
    def test_archive_messages_no_collection(self, backend):
        """Test archival fails without collection setup."""
        backend_obj, mongo_client, db, collection = backend
        # Don't set collection
        
        messages = [{"role": "user", "content": "Hello"}]
        result = backend_obj.archive_messages(messages, 1, "test_session")
        
        assert result is False
    
    def test_search_archives(self, backend, mock_openai_client):
        """Test vector search functionality."""
        backend_obj, mongo_client, db, collection = backend
        backend_obj.collection = collection
        
        # Mock search results
        mock_results = [
            {
                "archive_id": 1,
                "messages": [{"role": "user", "content": "test"}],
                "score": 0.95
            }
        ]
        collection.aggregate.return_value = mock_results
        
        results = backend_obj.search_archives("test query", limit=5)
        
        assert results == mock_results
        collection.aggregate.assert_called_once()
        mock_openai_client.embeddings.create.assert_called_once()
        
        # Verify the aggregation pipeline structure
        pipeline = collection.aggregate.call_args[0][0]
        assert len(pipeline) == 2  # $vectorSearch and $project stages
        assert "$vectorSearch" in pipeline[0]
        assert "$project" in pipeline[1]
    
    def test_search_archives_no_collection(self, backend):
        """Test search fails without collection setup."""
        backend_obj, mongo_client, db, collection = backend
        # Don't set collection
        
        results = backend_obj.search_archives("test query")
        
        assert results == []
    
    def test_consolidate_archives(self, backend, mock_openai_client):
        """Test archive consolidation."""
        backend_obj, mongo_client, db, collection = backend
        backend_obj.collection = collection
        
        # Mock existing archives to consolidate
        mock_archives = [
            {
                "archive_id": 1,
                "messages": [{"role": "user", "content": "msg1"}],
                "session_id": "test_session"
            },
            {
                "archive_id": 2, 
                "messages": [{"role": "user", "content": "msg2"}],
                "session_id": "test_session"
            }
        ]
        collection.find.return_value = mock_archives
        
        result = backend_obj.consolidate_archives(["1", "2"])
        
        assert result is not None
        assert result.startswith("consolidated_")
        collection.find.assert_called_once()
        collection.insert_one.assert_called_once()
        collection.delete_many.assert_called_once()
        mock_openai_client.embeddings.create.assert_called_once()
    
    def test_get_archive_count(self, backend):
        """Test getting archive count."""
        backend_obj, mongo_client, db, collection = backend
        backend_obj.collection = collection
        collection.count_documents.return_value = 10
        
        count = backend_obj.get_archive_count()
        
        assert count == 10
        collection.count_documents.assert_called_once_with({"type": {"$in": ["archive", "consolidated"]}})
    
    def test_cleanup_archives(self, backend):
        """Test archive cleanup."""
        backend_obj, mongo_client, db, collection = backend
        backend_obj.collection = collection
        
        result = backend_obj.cleanup_archives(["1", "2", "consolidated_abc"])
        
        assert result is True
        collection.delete_many.assert_called_once()
    
    @patch('vector_backends.mongodb_backend.MongoClient')
    def test_get_status(self, mock_mongo_constructor, backend):
        """Test status reporting."""
        backend_obj, mongo_client, db, collection = backend
        backend_obj.collection = collection
        backend_obj.collection_name = "test_collection"
        
        # Mock the client to be the expected one so status check works
        mock_mongo_constructor.return_value = mongo_client
        backend_obj.client = mongo_client
        backend_obj.db = db
        
        status = backend_obj.get_status()
        
        expected_keys = [
            "backend_type", "database_name", "collection_name", 
            "embedding_model", "archived_files", "consolidations",
            "connection_status", "total_documents", "storage_size_mb"
        ]
        
        for key in expected_keys:
            assert key in status
        
        assert status["backend_type"] == "mongodb"
        assert status["connection_status"] == "connected"
    
    def test_should_consolidate(self, backend):
        """Test consolidation trigger logic."""
        backend_obj, mongo_client, db, collection = backend
        backend_obj.collection = collection
        
        # Mock archive count
        collection.count_documents.return_value = 150
        
        assert backend_obj.should_consolidate(100) is True
        assert backend_obj.should_consolidate(200) is False
    
    def test_vector_bindata_conversion(self, backend):
        """Test vector to BinData conversion."""
        backend_obj, mongo_client, db, collection = backend
        
        test_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Convert to BinData
        bindata = backend_obj._vector_to_bindata(test_vector)
        
        # Convert back to vector
        restored_vector = backend_obj._bindata_to_vector(bindata)
        
        # Should be approximately equal (floating point precision)
        assert len(restored_vector) == len(test_vector)
        for i in range(len(test_vector)):
            assert abs(restored_vector[i] - test_vector[i]) < 0.001
    
    def test_close_connection(self, backend):
        """Test connection cleanup."""
        backend_obj, mongo_client, db, collection = backend
        
        # Set up the backend to have a client to close
        backend_obj.client = mongo_client
        backend_obj.db = db
        
        backend_obj.close()
        
        mongo_client.close.assert_called_once()
        assert backend_obj.client is None


class TestMongoDBConfiguration:
    """Test MongoDB backend configuration and factory."""
    
    @patch.dict('os.environ', {
        'VECTOR_BACKEND': 'mongodb',
        'OPENAI_API_KEY': 'test-key',
        'MONGODB_CONNECTION_STRING': 'mongodb://localhost:27017',
        'MONGODB_DATABASE': 'test_db'
    })
    def test_config_from_env_mongodb(self):
        """Test MongoDB configuration from environment."""
        from config import BackendConfig
        
        config = BackendConfig.from_env()
        
        assert config.backend_type == BackendType.MONGODB
        assert config.openai_api_key == 'test-key'
        assert config.mongodb_connection_string == 'mongodb://localhost:27017'
        assert config.mongodb_database == 'test_db'
    
    @patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key'
    })
    @patch('vector_backends.mongodb_backend.OpenAI')
    def test_create_mongodb_backend(self, mock_openai):
        """Test MongoDB backend creation via factory."""
        # Mock the OpenAI client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        config = BackendConfig(
            backend_type=BackendType.MONGODB,
            openai_api_key='test-key',
            mongodb_connection_string='mongodb://localhost:27017',
            mongodb_database='test_db'
        )
        
        backend = create_backend(config)
        
        assert isinstance(backend, MongoDBBackend)
        assert backend.database_name == 'test_db'
        # Since the backend is created, we can check its type and config
        assert backend.connection_string == 'mongodb://localhost:27017'
        assert backend.embedding_model == "text-embedding-3-small"
    
    def test_config_validation_mongodb_missing_connection(self):
        """Test validation fails when MongoDB connection string missing."""
        config = BackendConfig(
            backend_type=BackendType.MONGODB,
            openai_api_key='test-key'
            # mongodb_connection_string is missing
        )
        
        assert config.validate() is False


class TestMongoDBIntegration:
    """Integration tests for MongoDB backend with main application."""
    
    @patch('infinite_memory_chat.create_backend')
    def test_chat_with_mongodb_backend(self, mock_create_backend):
        """Test chat integration with MongoDB backend."""
        from infinite_memory_chat import InfiniteMemoryChat
        from config import BackendConfig, BackendType
        
        # Mock the backend
        mock_backend = Mock(spec=MongoDBBackend)
        mock_backend.setup_store.return_value = "test_collection"
        mock_backend.archive_messages.return_value = True
        mock_backend.should_consolidate.return_value = False
        mock_backend.search_archives.return_value = []
        mock_backend.get_status.return_value = {
            "backend_type": "mongodb",
            "archived_files": 0,
            "consolidations": 0,
            "consolidated_size_mb": 0
        }
        
        mock_create_backend.return_value = mock_backend
        
        # Test configuration
        config = BackendConfig(
            backend_type=BackendType.MONGODB,
            openai_api_key='test-key',
            mongodb_connection_string='mongodb://localhost:27017'
        )
        
        chat = InfiniteMemoryChat(config=config)
        
        assert isinstance(chat.backend, Mock)  # Our mocked backend
        assert chat.config.backend_type == BackendType.MONGODB
