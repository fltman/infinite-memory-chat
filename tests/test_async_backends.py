"""
Tests for async vector backends.
"""

import pytest
import pytest_asyncio
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import List, Dict

from vector_backends.async_base import AsyncVectorBackend, AsyncProgressTracker, AsyncBatchProcessor
from vector_backends.async_mongodb_backend import AsyncMongoDBBackend
from vector_backends.async_wrapper import AsyncBackendWrapper, AsyncBackendFactory, AsyncOperationManager


class TestAsyncProgressTracker:
    """Test async progress tracking functionality."""
    
    @pytest.mark.asyncio
    async def test_progress_tracker_without_callback(self):
        """Test progress tracker without callback."""
        tracker = AsyncProgressTracker(total_steps=5)
        
        await tracker.update("Step 1")
        assert tracker.current_step == 1
        
        await tracker.update("Step 2", step_increment=2)
        assert tracker.current_step == 3
        
        await tracker.complete("Done")
        assert tracker.current_step == 5
    
    @pytest.mark.asyncio
    async def test_progress_tracker_with_callback(self):
        """Test progress tracker with callback."""
        callback_calls = []
        
        async def mock_callback(message: str, progress: float):
            callback_calls.append((message, progress))
        
        tracker = AsyncProgressTracker(total_steps=4, callback=mock_callback)
        
        await tracker.update("Starting", 2)
        await tracker.complete("Finished")
        
        assert len(callback_calls) == 2
        assert callback_calls[0] == ("Starting", 0.5)  # 2/4 = 0.5
        assert callback_calls[1] == ("Finished", 1.0)


class TestAsyncBatchProcessor:
    """Test async batch processing functionality."""
    
    @pytest.mark.asyncio
    async def test_batch_processor_simple(self):
        """Test basic batch processing."""
        processor = AsyncBatchProcessor(batch_size=2, max_concurrent=2)
        
        async def process_item(item: int) -> int:
            await asyncio.sleep(0.01)  # Simulate work
            return item * 2
        
        items = [1, 2, 3, 4, 5]
        results = await processor.process_batch(items, process_item)
        
        assert len(results) == 5
        assert results == [2, 4, 6, 8, 10]
    
    @pytest.mark.asyncio
    async def test_batch_processor_with_callback(self):
        """Test batch processing with progress callback."""
        processor = AsyncBatchProcessor(batch_size=3, max_concurrent=2)
        callback_calls = []
        
        async def mock_callback(message: str, progress: float):
            callback_calls.append((message, progress))
        
        async def process_item(item: str) -> str:
            await asyncio.sleep(0.01)
            return item.upper()
        
        items = ["a", "b", "c", "d"]
        results = await processor.process_batch(items, process_item, mock_callback)
        
        assert results == ["A", "B", "C", "D"]
        assert len(callback_calls) >= 4  # At least one call per item
        assert callback_calls[-1][1] == 1.0  # Final progress should be 100%
    
    @pytest.mark.asyncio
    async def test_batch_processor_with_exceptions(self):
        """Test batch processing handles exceptions gracefully."""
        processor = AsyncBatchProcessor(batch_size=2, max_concurrent=2)
        
        async def process_item(item: int) -> int:
            if item == 3:
                raise ValueError(f"Error processing {item}")
            return item * 2
        
        items = [1, 2, 3, 4]
        results = await processor.process_batch(items, process_item)
        
        assert len(results) == 4
        assert results[0] == 2  # 1 * 2
        assert results[1] == 4  # 2 * 2
        assert results[2] is None  # Exception result
        assert results[3] == 8  # 4 * 2


class TestAsyncMongoDBBackend:
    """Test async MongoDB backend functionality."""
    
    @pytest_asyncio.fixture
    async def mock_motor_client(self):
        """Mock Motor (async MongoDB) client."""
        client = AsyncMock()
        db = AsyncMock()
        collection = AsyncMock()
        
        # Mock client methods
        client.admin = AsyncMock()
        client.admin.command = AsyncMock(return_value={"ok": 1})
        client.__getitem__ = Mock(return_value=db)
        client.close = AsyncMock()
        
        # Mock database methods
        db.list_collection_names = AsyncMock(return_value=[])
        db.create_collection = AsyncMock()
        db.__getitem__ = Mock(return_value=collection)
        db.command = AsyncMock(return_value={"storageSize": 1024})
        
        # Mock collection methods
        insert_result = Mock()
        insert_result.inserted_id = "mock_id_123"
        collection.insert_one = AsyncMock(return_value=insert_result)
        collection.count_documents = AsyncMock(return_value=10)
        
        # Mock find cursor
        find_cursor = AsyncMock()
        find_cursor.to_list = AsyncMock(return_value=[])
        collection.find = Mock(return_value=find_cursor)
        
        # Mock delete result
        delete_result = Mock()
        delete_result.deleted_count = 5
        collection.delete_many = AsyncMock(return_value=delete_result)
        
        # Mock aggregate cursor
        aggregate_cursor = AsyncMock()
        aggregate_cursor.to_list = AsyncMock(return_value=[])
        collection.aggregate = Mock(return_value=aggregate_cursor)
        
        return client, db, collection
    
    @pytest_asyncio.fixture
    async def mock_async_openai_client(self):
        """Mock async OpenAI client."""
        client = AsyncMock()
        embedding_response = AsyncMock()
        
        # Mock embedding data
        embedding_response.data = [Mock(embedding=[0.1] * 1536)]
        embedding_response.model_dump = Mock(return_value={
            "data": [{"embedding": [0.1] * 1536}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        })
        
        client.embeddings.create = AsyncMock(return_value=embedding_response)
        client.close = AsyncMock()
        
        return client
    
    @pytest_asyncio.fixture
    async def async_backend(self, mock_motor_client, mock_async_openai_client):
        """Create async MongoDB backend with mocked dependencies."""
        motor_client, db, collection = mock_motor_client
        
        with patch('vector_backends.async_mongodb_backend.AsyncIOMotorClient') as mock_motor_constructor:
            mock_motor_constructor.return_value = motor_client
            
            backend = AsyncMongoDBBackend(
                connection_string="mongodb://localhost:27017",
                database_name="test_async_db",
                embedding_client=mock_async_openai_client,
                embedding_model="text-embedding-3-small",
                max_concurrent_operations=5
            )
            
            # Manually set up the mocked connections for testing
            backend.client = motor_client
            backend.db = db
            backend.collection = collection
            backend.collection_name = "test_collection"
            
            return backend, motor_client, db, collection
    
    @pytest.mark.asyncio
    async def test_async_backend_initialization(self, mock_async_openai_client):
        """Test async backend initialization."""
        backend = AsyncMongoDBBackend(
            connection_string="mongodb://localhost:27017",
            database_name="test_db",
            embedding_client=mock_async_openai_client,
            max_concurrent_operations=15
        )
        
        assert backend.connection_string == "mongodb://localhost:27017"
        assert backend.database_name == "test_db"
        assert backend.max_concurrent_operations == 15
        assert backend.embedding_client == mock_async_openai_client
        assert backend.client is None  # Lazy connection
    
    @pytest.mark.asyncio
    async def test_setup_store(self, async_backend):
        """Test async store setup."""
        backend, motor_client, db, collection = async_backend
        
        store_id = await backend.setup_store("test_session")
        
        assert store_id == "chat_memory_test_session"
        assert backend.collection_name == "chat_memory_test_session"
        db.list_collection_names.assert_called_once()
        # Note: admin.command('hello') is only called when establishing new connection,
        # but in tests the client is pre-mocked, so this assertion is not needed
    
    @pytest.mark.asyncio
    async def test_archive_messages_success(self, async_backend, mock_async_openai_client):
        """Test successful async message archival."""
        backend, motor_client, db, collection = async_backend
        
        messages = [
            {"role": "user", "content": "Hello async world"},
            {"role": "assistant", "content": "Hello! Async operations are great!"}
        ]
        
        # Track progress calls
        progress_calls = []
        
        async def progress_callback(msg: str, progress: float):
            progress_calls.append((msg, progress))
        
        result = await backend.archive_messages(
            messages=messages,
            archive_id=42,
            session_id="async_test_session",
            progress_callback=progress_callback
        )
        
        assert result is True
        collection.insert_one.assert_called_once()
        mock_async_openai_client.embeddings.create.assert_called_once()
        
        # Check progress was tracked
        assert len(progress_calls) >= 5  # Should have multiple progress updates
        assert progress_calls[-1][1] == 1.0  # Final progress should be 100%
        
        # Verify document structure
        call_args = collection.insert_one.call_args[0][0]
        assert call_args["session_id"] == "async_test_session"
        assert call_args["archive_id"] == 42
        assert call_args["messages"] == messages
        assert call_args["type"] == "archive"
    
    @pytest.mark.asyncio
    async def test_archive_messages_no_collection(self, async_backend):
        """Test archival fails without collection setup."""
        backend, motor_client, db, collection = async_backend
        backend.collection = None  # Simulate no collection
        
        messages = [{"role": "user", "content": "Test"}]
        result = await backend.archive_messages(messages, 1, "test_session")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_search_archives(self, async_backend, mock_async_openai_client):
        """Test async vector search."""
        backend, motor_client, db, collection = async_backend
        
        # Mock search results
        mock_results = [
            {
                "archive_id": 1,
                "messages": [{"role": "user", "content": "async test"}],
                "score": 0.95
            }
        ]
        collection.aggregate.return_value.to_list = AsyncMock(return_value=mock_results)
        
        progress_calls = []
        
        async def progress_callback(msg: str, progress: float):
            progress_calls.append((msg, progress))
        
        results = await backend.search_archives(
            "test query async",
            limit=5,
            progress_callback=progress_callback
        )
        
        assert results == mock_results
        collection.aggregate.assert_called_once()
        mock_async_openai_client.embeddings.create.assert_called_once()
        
        # Check progress tracking
        assert len(progress_calls) >= 3
        assert progress_calls[-1][1] == 1.0
    
    @pytest.mark.asyncio
    async def test_consolidate_archives(self, async_backend, mock_async_openai_client):
        """Test async archive consolidation."""
        backend, motor_client, db, collection = async_backend
        
        # Mock archives to consolidate
        mock_archives = [
            {
                "archive_id": 1,
                "messages": [{"role": "user", "content": "msg1"}],
                "session_id": "test_session",
                "timestamp": datetime.now()
            },
            {
                "archive_id": 2,
                "messages": [{"role": "user", "content": "msg2"}],
                "session_id": "test_session",
                "timestamp": datetime.now()
            }
        ]
        collection.find.return_value.to_list = AsyncMock(return_value=mock_archives)
        
        result = await backend.consolidate_archives(["1", "2"])
        
        assert result is not None
        assert result.startswith("consolidated_")
        collection.find.assert_called_once()
        collection.insert_one.assert_called_once()
        collection.delete_many.assert_called_once()
        mock_async_openai_client.embeddings.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_archive_count(self, async_backend):
        """Test async archive count."""
        backend, motor_client, db, collection = async_backend
        collection.count_documents = AsyncMock(return_value=42)
        
        count = await backend.get_archive_count()
        
        assert count == 42
        collection.count_documents.assert_called_once_with({
            "type": {"$in": ["archive", "consolidated"]}
        })
    
    @pytest.mark.asyncio
    async def test_cleanup_archives(self, async_backend):
        """Test async archive cleanup."""
        backend, motor_client, db, collection = async_backend
        collection.delete_many.return_value.deleted_count = 3
        
        result = await backend.cleanup_archives(["1", "2", "consolidated_abc"])
        
        assert result is True
        collection.delete_many.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_status(self, async_backend):
        """Test async status reporting."""
        backend, motor_client, db, collection = async_backend
        collection.count_documents.side_effect = [15, 10, 5]  # total, archives, consolidated
        
        status = await backend.get_status()
        
        expected_keys = [
            "backend_type", "database_name", "collection_name",
            "embedding_model", "max_concurrent_operations",
            "connection_status", "total_documents"
        ]
        
        for key in expected_keys:
            assert key in status
        
        assert status["backend_type"] == "async_mongodb"
        assert status["connection_status"] == "connected"
        assert status["total_documents"] == 15
    
    @pytest.mark.asyncio
    async def test_should_consolidate(self, async_backend):
        """Test async consolidation trigger logic."""
        backend, motor_client, db, collection = async_backend
        collection.count_documents = AsyncMock(return_value=150)
        
        should_consolidate_yes = await backend.should_consolidate(100)
        should_consolidate_no = await backend.should_consolidate(200)
        
        assert should_consolidate_yes is True
        assert should_consolidate_no is False
    
    @pytest.mark.asyncio
    async def test_close_connection(self, async_backend, mock_async_openai_client):
        """Test async connection cleanup."""
        backend, motor_client, db, collection = async_backend
        
        await backend.close()
        
        motor_client.close.assert_called_once()
        mock_async_openai_client.close.assert_called_once()
        assert backend.client is None


class TestAsyncBackendWrapper:
    """Test async wrapper for sync backends."""
    
    @pytest.fixture
    def mock_sync_backend(self):
        """Mock sync backend."""
        backend = Mock()
        backend.setup_store.return_value = "test_store"
        backend.archive_messages.return_value = True
        backend.search_archives.return_value = [{"test": "result"}]
        backend.consolidate_archives.return_value = "consolidated_123"
        backend.get_archive_count.return_value = 5
        backend.cleanup_archives.return_value = True
        backend.get_status.return_value = {"backend_type": "mock", "status": "ok"}
        backend.should_consolidate.return_value = False
        backend.close = Mock()
        return backend
    
    @pytest.mark.asyncio
    async def test_async_wrapper_setup_store(self, mock_sync_backend):
        """Test wrapped setup_store."""
        wrapper = AsyncBackendWrapper(mock_sync_backend, max_workers=2)
        
        result = await wrapper.setup_store("test_session")
        
        assert result == "test_store"
        mock_sync_backend.setup_store.assert_called_once_with("test_session")
        
        await wrapper.close()
    
    @pytest.mark.asyncio
    async def test_async_wrapper_archive_messages(self, mock_sync_backend):
        """Test wrapped archive_messages."""
        wrapper = AsyncBackendWrapper(mock_sync_backend)
        
        progress_calls = []
        
        async def progress_callback(msg: str, progress: float):
            progress_calls.append((msg, progress))
        
        messages = [{"role": "user", "content": "test"}]
        result = await wrapper.archive_messages(
            messages, 1, "test_session", progress_callback
        )
        
        assert result is True
        mock_sync_backend.archive_messages.assert_called_once_with(messages, 1, "test_session")
        
        # Should have start and end progress calls
        assert len(progress_calls) == 2
        assert progress_calls[0][1] == 0.0
        assert progress_calls[1][1] == 1.0
        
        await wrapper.close()
    
    @pytest.mark.asyncio
    async def test_async_wrapper_get_status(self, mock_sync_backend):
        """Test wrapped get_status adds wrapper info."""
        wrapper = AsyncBackendWrapper(mock_sync_backend)
        
        status = await wrapper.get_status()
        
        assert status["wrapped_backend"] is True
        assert status["original_type"] == "mock"
        assert status["backend_type"] == "async_wrapped_mock"
        
        await wrapper.close()


class TestAsyncBackendFactory:
    """Test async backend factory."""
    
    @pytest.mark.asyncio
    async def test_create_async_mongodb_backend(self):
        """Test creating native async MongoDB backend."""
        with patch('vector_backends.async_mongodb_backend.AsyncMongoDBBackend') as MockBackend:
            mock_backend = AsyncMock()
            MockBackend.return_value = mock_backend
            
            backend = await AsyncBackendFactory.create_async_backend(
                "async_mongodb",
                connection_string="mongodb://localhost:27017",
                database_name="test_db"
            )
            
            assert backend == mock_backend
            MockBackend.assert_called_once_with(
                connection_string="mongodb://localhost:27017",
                database_name="test_db"
            )
    
    @pytest.mark.asyncio
    async def test_create_wrapped_mongodb_backend(self):
        """Test wrapping sync MongoDB backend."""
        with patch('vector_backends.mongodb_backend.MongoDBBackend') as MockSyncBackend:
            mock_sync = Mock()
            MockSyncBackend.return_value = mock_sync
            
            backend = await AsyncBackendFactory.create_async_backend(
                "mongodb",
                connection_string="mongodb://localhost:27017"
            )
            
            assert isinstance(backend, AsyncBackendWrapper)
            assert backend.sync_backend == mock_sync
    
    @pytest.mark.asyncio
    async def test_create_unknown_backend_raises_error(self):
        """Test error handling for unknown backend type."""
        with pytest.raises(ValueError, match="Unknown backend type: unknown"):
            await AsyncBackendFactory.create_async_backend("unknown")


class TestAsyncOperationManager:
    """Test async operation manager."""
    
    @pytest.mark.asyncio
    async def test_run_operation_success(self):
        """Test running successful operation."""
        manager = AsyncOperationManager(max_concurrent=2)
        
        async def test_operation(value: int) -> int:
            await asyncio.sleep(0.01)
            return value * 2
        
        result = await manager.run_operation("test_op", test_operation, 5)
        
        assert result == 10
        assert "test_op" in manager.active_operations
        assert manager.active_operations["test_op"]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_run_operation_failure(self):
        """Test running failed operation."""
        manager = AsyncOperationManager(max_concurrent=2)
        
        async def failing_operation():
            await asyncio.sleep(0.01)
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            await manager.run_operation("fail_op", failing_operation)
        
        assert manager.active_operations["fail_op"]["status"] == "failed"
        assert "Test error" in manager.active_operations["fail_op"]["error"]
    
    @pytest.mark.asyncio
    async def test_batch_archive_messages(self):
        """Test batch archiving through operation manager."""
        manager = AsyncOperationManager(max_concurrent=3)
        
        # Mock async backend
        mock_backend = AsyncMock()
        mock_backend.archive_messages = AsyncMock(return_value=True)
        
        message_batches = [
            {"messages": [{"role": "user", "content": "msg1"}], "archive_id": 1},
            {"messages": [{"role": "user", "content": "msg2"}], "archive_id": 2},
            {"messages": [{"role": "user", "content": "msg3"}], "archive_id": 3}
        ]
        
        results = await manager.batch_archive_messages(
            mock_backend, message_batches, "test_session"
        )
        
        assert len(results) == 3
        assert all(results)  # All should be True
        assert mock_backend.archive_messages.call_count == 3
    
    @pytest.mark.asyncio
    async def test_get_operation_status(self):
        """Test operation status reporting."""
        manager = AsyncOperationManager(max_concurrent=5)
        
        # Add some mock operations
        manager.active_operations = {
            "op1": {"status": "running", "start_time": 1000},
            "op2": {"status": "completed", "start_time": 999},
            "op3": {"status": "failed", "start_time": 998, "error": "test error"}
        }
        
        status = manager.get_operation_status()
        
        assert status["active_operations"] == 1  # Only "running" operations
        assert status["total_operations"] == 3
        assert status["semaphore_available"] == 5  # All slots available
        assert len(status["operations"]) == 3


# Integration test
class TestAsyncIntegration:
    """Integration tests for async backends."""
    
    @pytest.mark.asyncio
    async def test_async_backend_integration(self):
        """Test basic async backend integration without external dependencies."""
        # This test uses mocked dependencies but tests the full async flow
        
        with patch('vector_backends.async_mongodb_backend.AsyncIOMotorClient') as MockClient, \
             patch('vector_backends.async_mongodb_backend.AsyncOpenAI') as MockOpenAI:
            
            # Setup mocks
            mock_client = AsyncMock()
            mock_client.admin.command = AsyncMock(return_value={"ok": 1})
            mock_db = AsyncMock()
            mock_db.list_collection_names = AsyncMock(return_value=[])
            mock_db.create_collection = AsyncMock()
            mock_collection = AsyncMock()
            mock_collection.insert_one = AsyncMock()
            mock_collection.insert_one.return_value.inserted_id = "test_id"
            
            mock_client.__getitem__ = Mock(return_value=mock_db)
            mock_db.__getitem__ = Mock(return_value=mock_collection)
            MockClient.return_value = mock_client
            
            # Mock OpenAI client
            mock_openai = AsyncMock()
            mock_embedding_response = AsyncMock()
            mock_embedding_response.data = [Mock(embedding=[0.1] * 1536)]
            mock_embedding_response.model_dump = Mock(return_value={
                "data": [{"embedding": [0.1] * 1536}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 10, "total_tokens": 10}
            })
            mock_openai.embeddings.create = AsyncMock(return_value=mock_embedding_response)
            MockOpenAI.return_value = mock_openai
            
            # Test the integration
            backend = AsyncMongoDBBackend(
                connection_string="mongodb://localhost:27017",
                database_name="integration_test"
            )
            
            # Setup store
            store_id = await backend.setup_store("integration_session")
            assert store_id == "chat_memory_integration_session"
            
            # Archive messages
            messages = [
                {"role": "user", "content": "Integration test message"},
                {"role": "assistant", "content": "Integration test response"}
            ]
            
            result = await backend.archive_messages(messages, 1, "integration_session")
            assert result is True
            
            # Verify calls were made
            mock_collection.insert_one.assert_called_once()
            mock_openai.embeddings.create.assert_called_once()
            
            await backend.close()
