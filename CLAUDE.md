# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Infinite Memory Chat is a proof-of-concept chatbot with virtually unlimited conversation memory using OpenAI's Responses API and pluggable vector backends (OpenAI Vector Store or MongoDB Vector Search). It implements a self-populating RAG system that automatically archives conversation history, creating seamless long-term memory without manual intervention.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
source venv/bin/activate  # or: source .venv/bin/activate
python3 infinite_memory_chat.py
```

### Testing
```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_backends.py
pytest tests/test_mongodb_backend.py

# Run with verbose output
pytest -v
```

### Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv

# Install dependencies
pip install -r requirements.txt

# Configuration (copy and edit)
cp env.example .env
```

## Architecture

### Three-Level Memory Hierarchy

The system implements a cascading memory architecture to handle unlimited conversation length:

1. **Level 1: Active Conversation** (max 20 messages)
   - Maintained in `conversation_history` list in `InfiniteMemoryChat` class
   - Full detail, immediate context for chat responses
   - Triggers archival when reaching `MAX_MESSAGES` threshold

2. **Level 2: Vector Store Archive** (max 100 files/documents)
   - Archives created in batches of `ARCHIVE_COUNT` messages (default: 10)
   - Searchable via backend-specific vector search
   - Backend manages storage and indexing
   - Triggers consolidation when reaching `MAX_ARCHIVE_FILES`

3. **Level 3: Consolidated Archives**
   - Merges `CONSOLIDATION_COUNT` files (default: 50) into single units
   - All data preserved without summarization
   - Max ~500 MB per consolidated file

### Pluggable Backend Architecture

The codebase uses the Abstract Factory pattern for interchangeable vector storage:

**Base Interface**: `vector_backends/base.py`
- `VectorBackend` abstract class defines common interface
- All backends must implement: `setup_store()`, `archive_messages()`, `search_archives()`, `consolidate_archives()`, `get_archive_count()`, `cleanup_archives()`, `get_status()`, `should_consolidate()`

**Backend Implementations**:
- `vector_backends/openai_backend.py`: OpenAI Vector Store (default)
  - Uses file-based archival with temporary JSON files
  - Search handled via OpenAI's `file_search` tool in chat responses
  - Consolidation merges files to manage 10,000 file limit

- `vector_backends/mongodb_backend.py`: MongoDB Vector Search
  - Uses document-based archival with embeddings
  - Explicit vector search via `$vectorSearch` aggregation pipeline
  - Requires vector search index: "vector_index" on "content_embedding" field (1536 dimensions, cosine similarity)
  - Embeddings generated via OpenAI API

**Configuration**: `config.py`
- `BackendConfig` dataclass manages all configuration
- `BackendType` enum for backend selection
- `create_backend()` factory function instantiates correct backend
- Loads from environment variables via `BackendConfig.from_env()`

### Key Design Patterns

**Backend Switching Logic** (infinite_memory_chat.py:75-98):
- Injection via constructor parameter or automatic creation from config
- Graceful fallback to OpenAI backend on MongoDB connection failure
- Environment variable `VECTOR_BACKEND` controls selection

**Backend-Aware Chat Flow** (infinite_memory_chat.py:144-213):
- MongoDB backend: explicitly searches archives and injects context into system prompt
- OpenAI backend: uses `file_search` tool in responses.create() call
- Different search integration strategies but same user experience

**Archive Trigger** (infinite_memory_chat.py:110-143):
- Checks message count after each chat interaction
- Automatically archives oldest messages when threshold reached
- Delegates consolidation check to backend via `should_consolidate()`

## Important Configuration Constants

Located in `infinite_memory_chat.py` (lines 22-52):
- `MAX_MESSAGES = 20`: Active conversation window size
- `ARCHIVE_COUNT = 10`: Messages per archive batch
- `MAX_ARCHIVE_FILES = 100`: Trigger for consolidation
- `CONSOLIDATION_COUNT = 50`: Files to merge per consolidation
- `MAX_FILE_SIZE_MB = 500`: Safety limit per consolidated file
- `MODEL = "gpt-5-nano"`: Chat model (note: verify this model name)

## MongoDB-Specific Requirements

When working with MongoDB backend:

1. **Vector Search Index Required** (manually created in Atlas or via Admin API):
   - Index name: `vector_index`
   - Path: `content_embedding`
   - Dimensions: 1536 (text-embedding-3-small)
   - Similarity: cosine
   - Filter fields: `session_id`, `archive_id`, `type`

2. **Document Structure**:
   - `content_embedding`: Vector stored as both BinData and array format
   - `type`: "archive" or "consolidated" for filtering
   - Helper methods: `_vector_to_bindata()` and `_bindata_to_vector()` for efficient storage

3. **Vector Search Pipeline** (mongodb_backend.py:154-178):
   - Uses `$vectorSearch` aggregation stage
   - `numCandidates` set to 20x limit for better recall
   - Returns documents with `$meta: "vectorSearchScore"`

## Testing Approach

The test suite (tests/test_backends.py) focuses on:
- Backend interface compliance
- OpenAI backend operations with mocked client
- Integration tests verify backend injection into main chat class
- MongoDB tests require live database connection (tests/test_mongodb_backend.py)

When adding features:
- Mock OpenAI client for unit tests to avoid API costs
- Test both backends for feature parity
- Verify fallback logic for MongoDB connection failures

## Session Management

Sessions identified by timestamp: `YYYYMMDD_HHMMSS` format
- Created at `InfiniteMemoryChat` initialization (infinite_memory_chat.py:102)
- Used for: vector store naming, archive document IDs, file naming
- No persistence between sessions currently (session state lost on restart)

## Cost Optimization Notes

From README.md and env.example:
- OpenAI backend: $3-33/month for small scale (<10K messages/month)
- MongoDB backend: More cost-effective at scale (>50K messages/month) but requires infrastructure
- Embedding costs apply to both backends (OpenAI API used for embeddings)
- Small active context window (20 messages) minimizes tokens per request
