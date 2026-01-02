#!/usr/bin/env python3
"""
Performance benchmark comparing sync vs async backend operations.
"""

import asyncio
import time
import os
from typing import List, Dict
from unittest.mock import Mock, AsyncMock, patch

# Import both sync and async backends
from vector_backends.mongodb_backend import MongoDBBackend
from vector_backends.async_mongodb_backend import AsyncMongoDBBackend
from vector_backends.async_wrapper import AsyncBackendWrapper


class AsyncBenchmark:
    """Benchmark async vs sync MongoDB backend performance."""
    
    def __init__(self):
        self.test_messages = [[
            {"role": "user", "content": f"Test message {i} about performance benchmarking"},
            {"role": "assistant", "content": f"Response {i} explaining async benefits and performance gains"}
        ] for i in range(1, 21)]  # 20 message pairs
        
        self.batch_count = 10  # Number of concurrent operations
        
    def create_mock_clients(self):
        """Create consistent mocks for both sync and async backends."""
        # Sync mocks
        sync_mongo_client = Mock()
        sync_db = Mock()
        sync_collection = Mock()
        sync_openai = Mock()
        
        sync_mongo_client.admin.command.return_value = {"ok": 1}
        sync_mongo_client.__getitem__ = Mock(return_value=sync_db)
        sync_db.list_collection_names.return_value = []
        sync_db.create_collection = Mock()
        sync_db.__getitem__ = Mock(return_value=sync_collection)
        sync_db.command.return_value = {"storageSize": 1024}
        
        sync_collection.insert_one = Mock()
        sync_collection.insert_one.return_value.inserted_id = "sync_mock_id"
        sync_collection.count_documents.return_value = 10
        
        # Mock OpenAI response
        sync_embedding_response = Mock()
        sync_embedding_response.data = [Mock(embedding=[0.1] * 1536)]
        sync_embedding_response.model_dump.return_value = {
            "data": [{"embedding": [0.1] * 1536}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        }
        sync_openai.embeddings.create.return_value = sync_embedding_response
        
        # Async mocks (identical structure)
        async_mongo_client = AsyncMock()
        async_db = AsyncMock()
        async_collection = AsyncMock()
        async_openai = AsyncMock()
        
        async_mongo_client.admin.command = AsyncMock(return_value={"ok": 1})
        async_mongo_client.__getitem__ = Mock(return_value=async_db)
        async_db.list_collection_names = AsyncMock(return_value=[])
        async_db.create_collection = AsyncMock()
        async_db.__getitem__ = Mock(return_value=async_collection)
        async_db.command = AsyncMock(return_value={"storageSize": 1024})
        
        async_collection.insert_one = AsyncMock()
        async_collection.insert_one.return_value.inserted_id = "async_mock_id"
        async_collection.count_documents = AsyncMock(return_value=10)
        
        # Mock OpenAI response (async)
        async_embedding_response = AsyncMock()
        async_embedding_response.data = [Mock(embedding=[0.1] * 1536)]
        async_embedding_response.model_dump = Mock(return_value={
            "data": [{"embedding": [0.1] * 1536}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        })
        async_openai.embeddings.create = AsyncMock(return_value=async_embedding_response)
        
        return (
            (sync_mongo_client, sync_db, sync_collection, sync_openai),
            (async_mongo_client, async_db, async_collection, async_openai)
        )
    
    async def benchmark_sync_sequential(self) -> Dict:
        """Benchmark sync backend with sequential operations."""
        print(f"\n🐌 Benchmarking SYNC Sequential Operations ({self.batch_count} operations)")
        
        sync_mocks, _ = self.create_mock_clients()
        sync_mongo_client, sync_db, sync_collection, sync_openai = sync_mocks
        
        with patch('vector_backends.mongodb_backend.MongoClient') as MockClient, \
             patch('vector_backends.mongodb_backend.OpenAI') as MockOpenAI:
            
            MockClient.return_value = sync_mongo_client
            MockOpenAI.return_value = sync_openai
            
            backend = MongoDBBackend(
                connection_string="mongodb://localhost:27017",
                database_name="benchmark_sync",
                embedding_client=sync_openai
            )
            
            # Setup
            backend.client = sync_mongo_client
            backend.db = sync_db
            backend.collection = sync_collection
            backend.collection_name = "benchmark_collection"
            
            # Benchmark sequential operations
            start_time = time.time()
            successful_operations = 0
            
            for i in range(self.batch_count):
                try:
                    # Add small delay to simulate I/O
                    time.sleep(0.01)  # 10ms delay per operation
                    
                    result = backend.archive_messages(
                        messages=self.test_messages[i % len(self.test_messages)],
                        archive_id=i + 1,
                        session_id="sync_benchmark"
                    )
                    if result:
                        successful_operations += 1
                except Exception as e:
                    print(f"Sync operation {i+1} failed: {e}")
            
            end_time = time.time()
            total_time = end_time - start_time
            
            return {
                "type": "sync_sequential",
                "operations": self.batch_count,
                "successful": successful_operations,
                "total_time": total_time,
                "ops_per_second": self.batch_count / total_time,
                "avg_time_per_op": total_time / self.batch_count
            }
    
    async def benchmark_sync_wrapped(self) -> Dict:
        """Benchmark sync backend wrapped for async (ThreadPoolExecutor)."""
        print(f"\n🔄 Benchmarking SYNC-Wrapped Concurrent Operations ({self.batch_count} operations)")
        
        sync_mocks, _ = self.create_mock_clients()
        sync_mongo_client, sync_db, sync_collection, sync_openai = sync_mocks
        
        with patch('vector_backends.mongodb_backend.MongoClient') as MockClient, \
             patch('vector_backends.mongodb_backend.OpenAI') as MockOpenAI:
            
            MockClient.return_value = sync_mongo_client
            MockOpenAI.return_value = sync_openai
            
            sync_backend = MongoDBBackend(
                connection_string="mongodb://localhost:27017",
                database_name="benchmark_wrapped",
                embedding_client=sync_openai
            )
            
            # Setup
            sync_backend.client = sync_mongo_client
            sync_backend.db = sync_db
            sync_backend.collection = sync_collection
            sync_backend.collection_name = "benchmark_collection"
            
            # Wrap in async wrapper
            async_wrapper = AsyncBackendWrapper(sync_backend, max_workers=10)
            
            # Benchmark concurrent operations
            start_time = time.time()
            
            # Create tasks for concurrent execution
            tasks = []
            for i in range(self.batch_count):
                async def archive_operation(operation_id: int):
                    # Add small delay to simulate I/O
                    await asyncio.sleep(0.01)  # 10ms delay per operation
                    
                    return await async_wrapper.archive_messages(
                        messages=self.test_messages[operation_id % len(self.test_messages)],
                        archive_id=operation_id + 1,
                        session_id="wrapped_benchmark"
                    )
                
                tasks.append(archive_operation(i))
            
            # Execute all operations concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            total_time = end_time - start_time
            
            successful_operations = sum(1 for r in results if r is True)
            
            await async_wrapper.close()
            
            return {
                "type": "sync_wrapped_concurrent",
                "operations": self.batch_count,
                "successful": successful_operations,
                "total_time": total_time,
                "ops_per_second": self.batch_count / total_time,
                "avg_time_per_op": total_time / self.batch_count
            }
    
    async def benchmark_async_concurrent(self) -> Dict:
        """Benchmark native async backend with concurrent operations."""
        print(f"\n⚡ Benchmarking ASYNC Concurrent Operations ({self.batch_count} operations)")
        
        _, async_mocks = self.create_mock_clients()
        async_mongo_client, async_db, async_collection, async_openai = async_mocks
        
        with patch('vector_backends.async_mongodb_backend.AsyncIOMotorClient') as MockClient, \
             patch('vector_backends.async_mongodb_backend.AsyncOpenAI') as MockOpenAI:
            
            MockClient.return_value = async_mongo_client
            MockOpenAI.return_value = async_openai
            
            backend = AsyncMongoDBBackend(
                connection_string="mongodb://localhost:27017",
                database_name="benchmark_async",
                embedding_client=async_openai,
                max_concurrent_operations=10
            )
            
            # Setup
            backend.client = async_mongo_client
            backend.db = async_db
            backend.collection = async_collection
            backend.collection_name = "benchmark_collection"
            
            # Benchmark concurrent operations
            start_time = time.time()
            
            # Create tasks for concurrent execution
            tasks = []
            for i in range(self.batch_count):
                async def archive_operation(operation_id: int):
                    # Add small delay to simulate I/O
                    await asyncio.sleep(0.01)  # 10ms delay per operation
                    
                    return await backend.archive_messages(
                        messages=self.test_messages[operation_id % len(self.test_messages)],
                        archive_id=operation_id + 1,
                        session_id="async_benchmark"
                    )
                
                tasks.append(archive_operation(i))
            
            # Execute all operations concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            total_time = end_time - start_time
            
            successful_operations = sum(1 for r in results if r is True)
            
            await backend.close()
            
            return {
                "type": "async_concurrent",
                "operations": self.batch_count,
                "successful": successful_operations,
                "total_time": total_time,
                "ops_per_second": self.batch_count / total_time,
                "avg_time_per_op": total_time / self.batch_count
            }
    
    def print_results(self, results: List[Dict]):
        """Print formatted benchmark results."""
        print("\n" + "="*80)
        print("🏁 PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        
        for result in results:
            print(f"\n📊 {result['type'].upper().replace('_', ' ')}")
            print(f"   Operations: {result['operations']}")
            print(f"   Successful: {result['successful']} ({result['successful']/result['operations']:.1%})")
            print(f"   Total Time: {result['total_time']:.3f}s")
            print(f"   Throughput: {result['ops_per_second']:.2f} ops/sec")
            print(f"   Avg Time/Op: {result['avg_time_per_op']:.4f}s")
        
        print("\n" + "="*80)
        print("📈 PERFORMANCE COMPARISON")
        print("="*80)
        
        if len(results) >= 3:
            sync_sequential = results[0]
            sync_wrapped = results[1] 
            async_concurrent = results[2]
            
            # Calculate improvements
            wrapped_improvement = sync_wrapped['ops_per_second'] / sync_sequential['ops_per_second']
            async_improvement = async_concurrent['ops_per_second'] / sync_sequential['ops_per_second']
            async_vs_wrapped = async_concurrent['ops_per_second'] / sync_wrapped['ops_per_second']
            
            print(f"🔄 Sync-Wrapped vs Sync-Sequential: {wrapped_improvement:.2f}x faster")
            print(f"⚡ Async vs Sync-Sequential: {async_improvement:.2f}x faster")
            print(f"🚀 Native Async vs Wrapped: {async_vs_wrapped:.2f}x faster")
            
            # Time savings
            sync_time = sync_sequential['total_time']
            async_time = async_concurrent['total_time']
            time_saved = sync_time - async_time
            time_saved_percent = (time_saved / sync_time) * 100
            
            print(f"⏱️  Time Saved: {time_saved:.3f}s ({time_saved_percent:.1f}% reduction)")
            print(f"🎯 Best Performance: {max(results, key=lambda r: r['ops_per_second'])['type']}")
        
        print("\n" + "="*80)
    
    async def run_benchmark(self):
        """Run the complete benchmark suite."""
        print("🚀 Starting MongoDB Backend Performance Benchmark")
        print(f"📝 Test Configuration:")
        print(f"   • Operations per test: {self.batch_count}")
        print(f"   • Message pairs per operation: {len(self.test_messages[0])} messages")
        print(f"   • Simulated I/O delay: 10ms per operation")
        
        results = []
        
        try:
            # Run sync sequential benchmark
            sync_result = await self.benchmark_sync_sequential()
            results.append(sync_result)
            
            # Run sync wrapped benchmark
            wrapped_result = await self.benchmark_sync_wrapped()
            results.append(wrapped_result)
            
            # Run async concurrent benchmark
            async_result = await self.benchmark_async_concurrent()
            results.append(async_result)
            
            # Print results
            self.print_results(results)
            
            print("\n✅ Benchmark completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Run the benchmark."""
    benchmark = AsyncBenchmark()
    await benchmark.run_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
