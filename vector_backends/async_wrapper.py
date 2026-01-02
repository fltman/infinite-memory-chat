"""
Async wrapper for sync vector backends to provide async compatibility.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable, Awaitable
from concurrent.futures import ThreadPoolExecutor

from .base import VectorBackend
from .async_base import AsyncVectorBackend


class AsyncBackendWrapper(AsyncVectorBackend):
    """
    Wrapper to make sync backends async-compatible using ThreadPoolExecutor.
    This allows gradual migration from sync to async operations.
    """
    
    def __init__(self, 
                 sync_backend: VectorBackend, 
                 max_workers: int = 10):
        self.sync_backend = sync_backend
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="AsyncWrapper")
        
    async def setup_store(self, session_id: str) -> str:
        """Setup store using sync backend in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.sync_backend.setup_store,
            session_id
        )
    
    async def archive_messages(self, 
                              messages: List[Dict], 
                              archive_id: int, 
                              session_id: str,
                              progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None) -> bool:
        """Archive messages using sync backend in executor."""
        if progress_callback:
            await progress_callback("Starting archive operation...", 0.0)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.sync_backend.archive_messages,
            messages,
            archive_id,
            session_id
        )
        
        if progress_callback:
            await progress_callback("Archive operation completed", 1.0)
        
        return result
    
    async def search_archives(self, 
                             query_text: str, 
                             limit: int = 10,
                             progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None) -> List[Dict]:
        """Search archives using sync backend in executor."""
        if progress_callback:
            await progress_callback("Starting search operation...", 0.0)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.sync_backend.search_archives,
            query_text,
            limit
        )
        
        if progress_callback:
            await progress_callback("Search operation completed", 1.0)
        
        return result
    
    async def consolidate_archives(self, 
                                  archive_ids: List[str],
                                  progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None) -> Optional[str]:
        """Consolidate archives using sync backend in executor."""
        if progress_callback:
            await progress_callback("Starting consolidation...", 0.0)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.sync_backend.consolidate_archives,
            archive_ids
        )
        
        if progress_callback:
            await progress_callback("Consolidation completed", 1.0)
        
        return result
    
    async def get_archive_count(self) -> int:
        """Get archive count using sync backend in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.sync_backend.get_archive_count
        )
    
    async def cleanup_archives(self, archive_ids: List[str]) -> bool:
        """Cleanup archives using sync backend in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.sync_backend.cleanup_archives,
            archive_ids
        )
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status using sync backend in executor."""
        loop = asyncio.get_event_loop()
        status = await loop.run_in_executor(
            self.executor,
            self.sync_backend.get_status
        )
        # Mark this as wrapped
        status["wrapped_backend"] = True
        status["original_type"] = status.get("backend_type", "unknown")
        status["backend_type"] = f"async_wrapped_{status.get('backend_type', 'unknown')}"
        return status
    
    async def should_consolidate(self, max_archives: int) -> bool:
        """Check if consolidation should be triggered using sync backend in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.sync_backend.should_consolidate,
            max_archives
        )
    
    async def close(self):
        """Close the wrapper and sync backend."""
        # Close sync backend if it has a close method
        if hasattr(self.sync_backend, 'close'):
            if asyncio.iscoroutinefunction(self.sync_backend.close):
                await self.sync_backend.close()
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self.executor, self.sync_backend.close)
        
        # Shutdown the executor
        self.executor.shutdown(wait=True)


class AsyncBackendFactory:
    """Factory for creating async backends or wrapping sync backends."""
    
    @staticmethod
    async def create_async_backend(backend_type: str, **kwargs) -> AsyncVectorBackend:
        """
        Create an async backend or wrap a sync backend.
        
        Args:
            backend_type: Type of backend ("async_mongodb", "mongodb", "openai")
            **kwargs: Backend-specific configuration
            
        Returns:
            AsyncVectorBackend instance
        """
        if backend_type == "async_mongodb":
            from .async_mongodb_backend import AsyncMongoDBBackend
            return AsyncMongoDBBackend(**kwargs)
        
        elif backend_type == "mongodb":
            # Wrap sync MongoDB backend
            from .mongodb_backend import MongoDBBackend
            sync_backend = MongoDBBackend(**kwargs)
            return AsyncBackendWrapper(sync_backend)
        
        elif backend_type == "openai":
            # Wrap sync OpenAI backend
            from .openai_backend import OpenAIBackend
            sync_backend = OpenAIBackend(**kwargs)
            return AsyncBackendWrapper(sync_backend)
        
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")


class AsyncOperationManager:
    """
    Manager for coordinating multiple async operations with progress tracking.
    """
    
    def __init__(self, max_concurrent: int = 5):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_operations = {}
        
    async def run_operation(self, 
                           operation_id: str,
                           operation_func: Callable[..., Awaitable[Any]],
                           *args,
                           progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None,
                           **kwargs) -> Any:
        """
        Run an async operation with progress tracking and concurrency control.
        
        Args:
            operation_id: Unique identifier for the operation
            operation_func: Async function to execute
            *args: Arguments for the operation function
            progress_callback: Optional progress callback
            **kwargs: Keyword arguments for the operation function
            
        Returns:
            Result of the operation
        """
        async with self.semaphore:
            if progress_callback:
                await progress_callback(f"Starting operation: {operation_id}", 0.0)
            
            try:
                self.active_operations[operation_id] = {
                    "status": "running",
                    "start_time": asyncio.get_event_loop().time()
                }
                
                # Execute the operation
                result = await operation_func(*args, **kwargs)
                
                self.active_operations[operation_id]["status"] = "completed"
                if progress_callback:
                    await progress_callback(f"Completed operation: {operation_id}", 1.0)
                
                return result
                
            except Exception as e:
                self.active_operations[operation_id]["status"] = "failed"
                self.active_operations[operation_id]["error"] = str(e)
                if progress_callback:
                    await progress_callback(f"Failed operation: {operation_id}", 1.0)
                raise
                
            finally:
                # Clean up old operations (keep last 100)
                if len(self.active_operations) > 100:
                    oldest_ops = sorted(
                        self.active_operations.items(),
                        key=lambda x: x[1].get("start_time", 0)
                    )[:50]  # Remove oldest 50
                    for op_id, _ in oldest_ops:
                        del self.active_operations[op_id]
    
    async def batch_archive_messages(self,
                                   backend: AsyncVectorBackend,
                                   message_batches: List[Dict],
                                   session_id: str,
                                   progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None) -> List[bool]:
        """
        Archive multiple message batches concurrently.
        
        Args:
            backend: Async backend to use
            message_batches: List of message batch dicts with 'messages' and 'archive_id'
            session_id: Session identifier
            progress_callback: Optional overall progress callback
            
        Returns:
            List of success/failure results for each batch
        """
        if not message_batches:
            return []
        
        total_batches = len(message_batches)
        completed = 0
        results = []
        
        async def archive_single_batch(batch: Dict) -> bool:
            nonlocal completed
            
            # Individual batch progress callback
            async def batch_progress(msg: str, progress: float):
                nonlocal completed
                if progress >= 1.0:
                    completed += 1
                
                if progress_callback:
                    overall_progress = completed / total_batches
                    await progress_callback(
                        f"Batch {completed}/{total_batches}: {msg}",
                        overall_progress
                    )
            
            result = await backend.archive_messages(
                messages=batch['messages'],
                archive_id=batch['archive_id'],
                session_id=session_id,
                progress_callback=batch_progress
            )
            
            return result
        
        # Create tasks for all batches
        tasks = []
        for i, batch in enumerate(message_batches):
            operation_id = f"archive_batch_{session_id}_{i}"
            task = self.run_operation(
                operation_id,
                archive_single_batch,
                batch
            )
            tasks.append(task)
        
        # Execute all batches concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to False
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    print(f"⚠️ Batch failed: {result}")
                    processed_results.append(False)
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            print(f"❌ Batch archiving failed: {e}")
            return [False] * total_batches
    
    def get_operation_status(self) -> Dict[str, Any]:
        """Get status of all operations."""
        return {
            "active_operations": len([
                op for op in self.active_operations.values() 
                if op.get("status") == "running"
            ]),
            "total_operations": len(self.active_operations),
            "semaphore_available": self.semaphore._value,
            "operations": dict(list(self.active_operations.items())[-10:])  # Last 10 operations
        }
