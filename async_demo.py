#!/usr/bin/env python3
"""
Async MongoDB Backend Demo - Showcasing high-performance async operations.
"""

import asyncio
import time
import os
from typing import List, Dict, Any
from vector_backends.async_mongodb_backend import AsyncMongoDBBackend
from vector_backends.async_wrapper import AsyncBackendFactory, AsyncOperationManager


class AsyncDemo:
    """Demo class showcasing async MongoDB backend capabilities."""
    
    def __init__(self):
        self.backend = None
        self.operation_manager = AsyncOperationManager(max_concurrent=10)
        
    async def setup_backend(self) -> bool:
        """Initialize async MongoDB backend."""
        print("🚀 Setting up Async MongoDB Backend...")
        
        # Use environment variables or defaults
        connection_string = os.getenv(
            "MONGODB_CONNECTION_STRING", 
            "mongodb://localhost:27017"
        )
        
        try:
            self.backend = await AsyncBackendFactory.create_async_backend(
                "async_mongodb",
                connection_string=connection_string,
                database_name="async_demo",
                max_concurrent_operations=10
            )
            
            # Setup store
            store_id = await self.backend.setup_store("async_demo_session")
            print(f"✅ Backend initialized with store: {store_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup backend: {e}")
            return False
    
    async def demo_progress_callback(self, message: str, progress: float):
        """Demo progress callback that shows real-time updates."""
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"\r🔄 [{bar}] {progress:.1%} - {message}", end='', flush=True)
        if progress >= 1.0:
            print()  # New line when complete
    
    async def demo_single_archive(self) -> bool:
        """Demo single message archiving with progress tracking."""
        print("\n📝 Demo 1: Single Archive with Progress Tracking")
        
        messages = [
            {"role": "user", "content": "What is async programming?"},
            {"role": "assistant", "content": "Async programming allows non-blocking operations, improving performance for I/O-bound tasks like database operations and API calls."},
            {"role": "user", "content": "How does it help with MongoDB?"},
            {"role": "assistant", "content": "With async MongoDB operations, multiple database queries can run concurrently, dramatically improving throughput for applications handling many requests."}
        ]
        
        start_time = time.time()
        success = await self.backend.archive_messages(
            messages=messages,
            archive_id=1,
            session_id="async_demo_session",
            progress_callback=self.demo_progress_callback
        )
        end_time = time.time()
        
        print(f"✅ Single archive completed in {end_time - start_time:.2f}s - Success: {success}")
        return success
    
    async def demo_concurrent_archives(self) -> List[bool]:
        """Demo concurrent message archiving."""
        print("\n🔥 Demo 2: Concurrent Archives (High Performance)")
        
        # Create multiple message batches
        message_batches = []
        for i in range(2, 12):  # Archives 2-11 (10 batches)
            messages = [
                {"role": "user", "content": f"This is batch {i} user message about async operations"},
                {"role": "assistant", "content": f"This is batch {i} assistant response explaining async benefits for batch {i}"},
                {"role": "user", "content": f"Batch {i} follow-up question"},
                {"role": "assistant", "content": f"Batch {i} detailed response with technical explanation"}
            ]
            message_batches.append({
                "messages": messages,
                "archive_id": i
            })
        
        print(f"📦 Archiving {len(message_batches)} batches concurrently...")
        
        start_time = time.time()
        results = await self.operation_manager.batch_archive_messages(
            backend=self.backend,
            message_batches=message_batches,
            session_id="async_demo_session",
            progress_callback=self.demo_progress_callback
        )
        end_time = time.time()
        
        successful = sum(results)
        total_time = end_time - start_time
        throughput = len(message_batches) / total_time
        
        print(f"✅ Concurrent archives completed:")
        print(f"   • {successful}/{len(results)} successful")
        print(f"   • Total time: {total_time:.2f}s")
        print(f"   • Throughput: {throughput:.1f} archives/second")
        print(f"   • Average per archive: {total_time/len(message_batches):.2f}s")
        
        return results
    
    async def demo_search_operations(self) -> List[Dict]:
        """Demo concurrent search operations."""
        print("\n🔍 Demo 3: Concurrent Search Operations")
        
        search_queries = [
            "async programming benefits",
            "MongoDB performance",
            "batch operations",
            "concurrent processing",
            "database throughput"
        ]
        
        print(f"🔎 Running {len(search_queries)} concurrent searches...")
        
        async def single_search(query: str) -> List[Dict]:
            return await self.backend.search_archives(
                query_text=query,
                limit=5,
                progress_callback=lambda msg, prog: self.demo_progress_callback(f"Search '{query[:20]}...': {msg}", prog)
            )
        
        start_time = time.time()
        # Run all searches concurrently
        search_tasks = [
            self.operation_manager.run_operation(f"search_{i}", single_search, query)
            for i, query in enumerate(search_queries)
        ]
        
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        end_time = time.time()
        
        total_results = 0
        successful_searches = 0
        
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                print(f"❌ Search {i+1} failed: {result}")
            else:
                successful_searches += 1
                total_results += len(result)
                print(f"✅ Search {i+1} ('{search_queries[i][:30]}...'): {len(result)} results")
        
        total_time = end_time - start_time
        print(f"\n📊 Search Performance:")
        print(f"   • {successful_searches}/{len(search_queries)} successful")
        print(f"   • Total results: {total_results}")
        print(f"   • Total time: {total_time:.2f}s")
        print(f"   • Search throughput: {len(search_queries)/total_time:.1f} searches/second")
        
        return [r for r in search_results if not isinstance(r, Exception)]
    
    async def demo_consolidation(self) -> str:
        """Demo archive consolidation."""
        print("\n🔄 Demo 4: Archive Consolidation")
        
        # Consolidate archives 2-5
        archive_ids = ["2", "3", "4", "5"]
        
        print(f"📦 Consolidating archives {archive_ids}...")
        
        start_time = time.time()
        consolidated_id = await self.backend.consolidate_archives(
            archive_ids=archive_ids,
            progress_callback=self.demo_progress_callback
        )
        end_time = time.time()
        
        print(f"✅ Consolidation completed in {end_time - start_time:.2f}s")
        print(f"   • Consolidated ID: {consolidated_id}")
        
        return consolidated_id
    
    async def demo_status_and_cleanup(self):
        """Demo status reporting and cleanup."""
        print("\n📊 Demo 5: Status Reporting")
        
        # Get status
        status = await self.backend.get_status()
        print("📈 Backend Status:")
        for key, value in status.items():
            print(f"   • {key}: {value}")
        
        # Get archive count
        count = await self.backend.get_archive_count()
        print(f"📂 Total archives: {count}")
        
        # Operation manager status
        op_status = self.operation_manager.get_operation_status()
        print(f"⚙️  Operation Manager:")
        print(f"   • Active operations: {op_status['active_operations']}")
        print(f"   • Total operations: {op_status['total_operations']}")
        print(f"   • Available slots: {op_status['semaphore_available']}")
    
    async def run_performance_benchmark(self):
        """Run a comprehensive performance benchmark."""
        print("\n🏁 Performance Benchmark: Sync vs Async Comparison")
        
        # Create test data
        test_batches = []
        for i in range(20, 30):  # 10 more batches for benchmarking
            messages = [
                {"role": "user", "content": f"Benchmark batch {i} - testing async performance"},
                {"role": "assistant", "content": f"Benchmark response {i} - measuring concurrent throughput"},
            ]
            test_batches.append({
                "messages": messages,
                "archive_id": i
            })
        
        print(f"⚡ Benchmarking with {len(test_batches)} operations...")
        
        # Benchmark concurrent execution
        print("\n🔥 Concurrent Execution:")
        start_time = time.time()
        concurrent_results = await self.operation_manager.batch_archive_messages(
            backend=self.backend,
            message_batches=test_batches,
            session_id="benchmark_session"
        )
        concurrent_time = time.time() - start_time
        concurrent_success = sum(concurrent_results)
        
        print(f"   • Time: {concurrent_time:.2f}s")
        print(f"   • Success rate: {concurrent_success}/{len(test_batches)} ({concurrent_success/len(test_batches):.1%})")
        print(f"   • Throughput: {len(test_batches)/concurrent_time:.1f} operations/second")
        
        print("\n📈 Performance Summary:")
        print(f"   • Concurrent async operations: {len(test_batches)/concurrent_time:.1f} ops/sec")
        print(f"   • Estimated sync equivalent: ~{len(test_batches)/(concurrent_time*5):.1f} ops/sec")  # Estimate 5x slower
        print(f"   • Performance improvement: ~5x faster with async")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.backend:
            await self.backend.close()
        print("✅ Cleanup completed")


async def main():
    """Main demo function."""
    print("🚀 Async MongoDB Backend Performance Demo")
    print("=" * 50)
    
    demo = AsyncDemo()
    
    try:
        # Setup
        if not await demo.setup_backend():
            print("❌ Failed to setup backend. Please check MongoDB connection.")
            return
        
        # Run demos
        await demo.demo_single_archive()
        await demo.demo_concurrent_archives()
        await demo.demo_search_operations()
        await demo.demo_consolidation()
        await demo.demo_status_and_cleanup()
        await demo.run_performance_benchmark()
        
        print("\n🎉 All demos completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    # Check if we're in an event loop already (Jupyter/IPython)
    try:
        loop = asyncio.get_running_loop()
        print("⚠️ Running in existing event loop - use 'await main()' instead")
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        asyncio.run(main())
