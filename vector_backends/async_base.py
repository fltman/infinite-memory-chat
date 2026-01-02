"""
Abstract base class for async vector storage backends.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Awaitable
from datetime import datetime
import asyncio


class AsyncVectorBackend(ABC):
    """Abstract base class for async vector storage backends."""
    
    @abstractmethod
    async def setup_store(self, session_id: str) -> str:
        """
        Create a new vector store for the session.
        
        Args:
            session_id: Unique identifier for the chat session
            
        Returns:
            Store identifier (vector store ID for OpenAI, collection name for MongoDB)
        """
        pass
    
    @abstractmethod
    async def archive_messages(self, 
                              messages: List[Dict], 
                              archive_id: int, 
                              session_id: str,
                              progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None) -> bool:
        """
        Archive messages to vector store with async operations.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            archive_id: Sequential archive number
            session_id: Session identifier
            progress_callback: Optional callback for progress updates (message, progress_0_to_1)
            
        Returns:
            True if archival was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def search_archives(self, 
                             query_text: str, 
                             limit: int = 10,
                             progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None) -> List[Dict]:
        """
        Search archived messages for relevant content with async operations.
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results to return
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of relevant archive results with score information
        """
        pass
    
    @abstractmethod
    async def consolidate_archives(self, 
                                  archive_ids: List[str],
                                  progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None) -> Optional[str]:
        """
        Consolidate multiple archives into a single unit with async operations.
        
        Args:
            archive_ids: List of archive identifiers to consolidate
            progress_callback: Optional callback for progress updates
            
        Returns:
            Consolidated archive identifier, or None if consolidation failed
        """
        pass
    
    @abstractmethod
    async def get_archive_count(self) -> int:
        """
        Get the number of archived message groups.
        
        Returns:
            Number of separate archives stored
        """
        pass
    
    @abstractmethod
    async def cleanup_archives(self, archive_ids: List[str]) -> bool:
        """
        Remove specified archives from storage.
        
        Args:
            archive_ids: List of archive identifiers to remove
            
        Returns:
            True if cleanup was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """
        Get backend-specific status information.
        
        Returns:
            Dictionary with status information (archive counts, storage size, etc.)
        """
        pass
    
    @abstractmethod
    async def should_consolidate(self, max_archives: int) -> bool:
        """
        Check if consolidation should be triggered.
        
        Args:
            max_archives: Maximum number of archives before consolidation
            
        Returns:
            True if consolidation should be performed
        """
        pass

    async def close(self):
        """
        Clean up connections and resources.
        Override in subclasses if needed.
        """
        pass


class AsyncProgressTracker:
    """Helper class for tracking async operation progress."""
    
    def __init__(self, 
                 total_steps: int, 
                 callback: Optional[Callable[[str, float], Awaitable[None]]] = None):
        self.total_steps = total_steps
        self.current_step = 0
        self.callback = callback
        
    async def update(self, message: str, step_increment: int = 1):
        """Update progress and notify callback if provided."""
        self.current_step = min(self.current_step + step_increment, self.total_steps)
        progress = self.current_step / self.total_steps if self.total_steps > 0 else 1.0
        
        if self.callback:
            await self.callback(message, progress)
    
    async def complete(self, message: str = "Operation completed"):
        """Mark operation as complete."""
        self.current_step = self.total_steps
        if self.callback:
            await self.callback(message, 1.0)


class AsyncBatchProcessor:
    """Helper class for processing items in async batches."""
    
    def __init__(self, batch_size: int = 10, max_concurrent: int = 5):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(self, 
                           items: List[Any], 
                           process_func: Callable[[Any], Awaitable[Any]],
                           progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None) -> List[Any]:
        """
        Process items in concurrent batches.
        
        Args:
            items: List of items to process
            process_func: Async function to process each item
            progress_callback: Optional progress callback
            
        Returns:
            List of processed results
        """
        if not items:
            return []
        
        results = []
        total_items = len(items)
        processed_count = 0
        
        # Process items in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            # Process batch with concurrency control
            batch_tasks = []
            for item in batch:
                task = self._process_with_semaphore(process_func, item)
                batch_tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    # Log error but don't fail entire batch
                    print(f"⚠️ Error processing item {i + j}: {result}")
                    results.append(None)
                else:
                    results.append(result)
                
                processed_count += 1
                
                # Update progress
                if progress_callback:
                    progress = processed_count / total_items
                    await progress_callback(f"Processed {processed_count}/{total_items} items", progress)
        
        return results
    
    async def _process_with_semaphore(self, process_func: Callable[[Any], Awaitable[Any]], item: Any) -> Any:
        """Process single item with semaphore control."""
        async with self._semaphore:
            return await process_func(item)
