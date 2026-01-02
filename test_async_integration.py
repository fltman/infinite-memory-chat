#!/usr/bin/env python3
"""
Quick test script to verify async backend integration
"""
import asyncio
import os
from config import BackendConfig, BackendType, create_async_backend, is_async_backend

async def test_async_integration():
    """Test the async backend integration."""
    print("🧪 Testing Async Backend Integration")
    print("=" * 50)
    
    # Test 1: Configuration loading
    print("\n1. Testing configuration loading...")
    config = BackendConfig.from_env()
    print(f"   Current backend type: {config.backend_type.value}")
    print(f"   Is async backend: {is_async_backend(config)}")
    
    # Test 2: Create async backend
    print("\n2. Testing async backend creation...")
    try:
        # Temporarily set to async_mongodb for testing
        test_config = BackendConfig(
            backend_type=BackendType.ASYNC_MONGODB,
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            mongodb_connection_string=os.getenv('MONGODB_CONNECTION_STRING'),
            mongodb_database='test_async_integration'
        )
        
        if test_config.validate():
            backend = await create_async_backend(test_config)
            print(f"   ✅ Created: {type(backend).__name__}")
            
            # Test 3: Basic operations
            print("\n3. Testing basic async operations...")
            
            # Setup store
            store_id = await backend.setup_store("test_session")
            print(f"   ✅ Store setup: {store_id}")
            
            # Get status
            status = await backend.get_status()
            print(f"   ✅ Status check: {status['backend_type']}")
            
            # Test archival (will likely fail without real messages, but tests interface)
            test_messages = [
                {"role": "user", "content": "Hello async world!"},
                {"role": "assistant", "content": "Hello! This is a test."}
            ]
            
            progress_calls = []
            async def test_progress(message: str, progress: float):
                progress_calls.append((message, progress))
                print(f"     📦 {message} ({progress*100:.0f}%)")
            
            try:
                result = await backend.archive_messages(
                    test_messages, 1, "test_session", progress_callback=test_progress
                )
                print(f"   ✅ Archive test: {result}")
                print(f"   ✅ Progress calls: {len(progress_calls)}")
            except Exception as e:
                print(f"   ⚠️  Archive test failed (expected): {e}")
            
            # Cleanup
            await backend.close()
            print("   ✅ Backend closed")
            
        else:
            print("   ❌ Configuration validation failed")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n4. Testing sync backend fallback...")
    try:
        # Test with OpenAI backend (sync)
        fallback_config = BackendConfig(
            backend_type=BackendType.OPENAI,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        backend = await create_async_backend(fallback_config)
        print(f"   ✅ Fallback backend: {type(backend).__name__}")
        await backend.close()
        
    except Exception as e:
        print(f"   ❌ Fallback error: {e}")
    
    print("\n🎉 Async integration test complete!")

if __name__ == "__main__":
    asyncio.run(test_async_integration())
