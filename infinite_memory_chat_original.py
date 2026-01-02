"""
Infinite Memory Chat - Proof of Concept
========================================
A chatbot with "infinite" memory by combining:
- OpenAI Responses API for conversation
- Vector Store for long-term memory
- Automatic archival of older messages
"""

import os
import json
import time
from datetime import datetime
from openai import OpenAI

# Configuration
# =============================================================================
# LEVEL 1: Active conversation
# Max number of messages before archival is triggered.
# Lower value = less token consumption per request, but more archival operations.
MAX_MESSAGES = 20

# Number of messages to archive at a time.
# Tip: Overlap can improve context (e.g., archive 10, but keep 2 extra)
ARCHIVE_COUNT = 10

# =============================================================================
# LEVEL 2: Vector store archive
# OpenAI allows max 10,000 files per vector store.
# When MAX_ARCHIVE_FILES is reached, the oldest CONSOLIDATION_COUNT files are
# merged into a single file. All data is preserved - we only reduce file count.
# Example: 100 files → 50 merged into 1 → 51 files remaining
# This makes it take ~50x longer to reach the 10,000 file limit.
MAX_ARCHIVE_FILES = 100

# Number of files to merge into a single file during consolidation
CONSOLIDATION_COUNT = 50

# =============================================================================
# LEVEL 3: File size limit
# OpenAI has a limit of 512 MB per file.
# We set a safety margin at 500 MB.
# When a consolidated file approaches this limit, a new consolidated file is created.
# Theoretical max: ~50 files × 500 MB = 25 GB chat history
MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# =============================================================================
# Model
MODEL = "gpt-5-nano"  

SYSTEM_PROMPT = """You are a helpful assistant with long-term memory.

IMPORTANT ABOUT YOUR MEMORY:
- The active conversation only shows the most recent messages.
- Older conversation history is saved in your knowledge base (file_search).
- If the user references something you discussed earlier, or if you need
  context from earlier in the conversation, ALWAYS search the knowledge base first.
- Never assume you lack information – it may be in the knowledge base.

When searching, consider:
- The user's name, projects, or topics you've discussed
- Decisions or agreements that were made
- Specific details the user has shared

If you find relevant history, feel free to confirm this for the user."""


class InfiniteMemoryChat:
    def __init__(self):
        self.client = OpenAI()
        self.vector_store_id = None
        self.conversation_history = []
        self.archive_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.archived_files = []  # Track file IDs in order
        self.consolidation_count = 0  # Number of times we've consolidated
        
    def setup_vector_store(self):
        """Create a new vector store for this session."""
        print("📦 Creating vector store...")
        vector_store = self.client.vector_stores.create(
            name=f"chat_memory_{self.session_id}"
        )
        self.vector_store_id = vector_store.id
        print(f"✅ Vector store created: {self.vector_store_id}")
        return self.vector_store_id
    
    def archive_messages(self):
        """Archive the oldest messages to vector store."""
        if len(self.conversation_history) < MAX_MESSAGES:
            return
        
        # Extract the oldest messages
        messages_to_archive = self.conversation_history[:ARCHIVE_COUNT]
        self.conversation_history = self.conversation_history[ARCHIVE_COUNT:]
        
        # Create JSON file with metadata
        self.archive_count += 1
        archive_data = {
            "archive_id": self.archive_count,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "message_range": f"messages {(self.archive_count-1)*ARCHIVE_COUNT + 1}-{self.archive_count*ARCHIVE_COUNT}",
            "messages": messages_to_archive
        }
        
        # Create summary text for better embeddings
        summary_text = f"Conversation archive #{self.archive_count}\n"
        summary_text += f"Timestamp: {archive_data['timestamp']}\n"
        summary_text += f"Content:\n"
        for msg in messages_to_archive:
            role = "User" if msg["role"] == "user" else "Assistant"
            summary_text += f"\n{role}: {msg['content']}\n"
        
        # Save to file
        filename = f"/tmp/archive_{self.session_id}_{self.archive_count}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"summary": summary_text, "data": archive_data}, f, ensure_ascii=False, indent=2)
        
        print(f"\n📁 Archiving {ARCHIVE_COUNT} messages to vector store...")
        
        # Upload to vector store
        with open(filename, "rb") as f:
            file_obj = self.client.files.create(file=f, purpose="assistants")
        
        self.client.vector_stores.files.create(
            vector_store_id=self.vector_store_id,
            file_id=file_obj.id
        )
        
        # Save file ID for later consolidation
        self.archived_files.append({
            "file_id": file_obj.id,
            "archive_id": self.archive_count,
            "filename": filename
        })
        
        # Wait for indexing
        self._wait_for_indexing(file_obj.id)
        print(f"✅ Archive #{self.archive_count} saved and indexed")
        
        # Check if we need to consolidate
        self._check_consolidation()
        
    def _wait_for_indexing(self, file_id, timeout=30):
        """Wait until the file is indexed in vector store."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            file_status = self.client.vector_stores.files.retrieve(
                vector_store_id=self.vector_store_id,
                file_id=file_id
            )
            if file_status.status == "completed":
                return True
            time.sleep(1)
        print("⚠️ Timeout while indexing")
        return False
    
    def _check_consolidation(self):
        """Check if we need to consolidate older files."""
        if len(self.archived_files) >= MAX_ARCHIVE_FILES:
            print(f"\n🔄 Max archive files ({MAX_ARCHIVE_FILES}) reached - consolidating...")
            self._consolidate_files()
    
    def _consolidate_files(self):
        """
        Merge the oldest files into a single large file.
        No summarization - all data is preserved intact.
        50 files → 1 file = takes 50x longer to reach 10,000 file limit.
        
        If the resulting file would exceed MAX_FILE_SIZE_BYTES,
        we skip already large consolidated files.
        """
        files_to_consolidate = self.archived_files[:CONSOLIDATION_COUNT]
        
        # Separate already consolidated files that are near the size limit
        large_consolidated = []
        files_to_merge = []
        
        for file_info in files_to_consolidate:
            if file_info.get("is_consolidated"):
                # Check file size
                try:
                    file_size = os.path.getsize(file_info["filename"])
                    if file_size > MAX_FILE_SIZE_BYTES * 0.8:  # 80% of limit
                        large_consolidated.append(file_info)
                        print(f"⚠️ Skipping large file ({file_size / 1024 / 1024:.1f} MB)")
                        continue
                except FileNotFoundError:
                    pass
            files_to_merge.append(file_info)
        
        if not files_to_merge:
            print("⚠️ No files to consolidate (all are too large)")
            return
        
        # Collect all content from files to merge
        all_archives = []
        for file_info in files_to_merge:
            try:
                with open(file_info["filename"], "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # If it's a consolidated file, extract its archives
                    if file_info.get("is_consolidated") and "archives" in data:
                        all_archives.extend(data["archives"])
                    else:
                        all_archives.append(data)
            except FileNotFoundError:
                print(f"⚠️ Could not read {file_info['filename']}")
                continue
        
        self.consolidation_count += 1
        
        print(f"📦 Merging {len(files_to_merge)} files into 1...")
        
        # Create merged file with all data
        consolidated_data = {
            "type": "consolidated_archive",
            "consolidation_id": self.consolidation_count,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "original_archive_ids": [f["archive_id"] for f in files_to_merge],
            "file_count": len(all_archives),
            "archives": all_archives  # All original data preserved
        }
        
        filename = f"/tmp/consolidated_{self.session_id}_{self.consolidation_count}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(consolidated_data, f, ensure_ascii=False, indent=2)
        
        # Check resulting file size
        file_size = os.path.getsize(filename)
        file_size_mb = file_size / 1024 / 1024
        print(f"📏 Consolidated file size: {file_size_mb:.1f} MB / {MAX_FILE_SIZE_MB} MB")
        
        if file_size > MAX_FILE_SIZE_BYTES:
            print(f"❌ File too large! Aborting consolidation.")
            os.remove(filename)
            return
        
        # Remove old files from vector store
        print(f"🗑️ Removing {len(files_to_merge)} old archive files...")
        for file_info in files_to_merge:
            try:
                self.client.vector_stores.files.delete(
                    vector_store_id=self.vector_store_id,
                    file_id=file_info["file_id"]
                )
                self.client.files.delete(file_id=file_info["file_id"])
            except Exception as e:
                print(f"⚠️ Could not delete file: {e}")
        
        # Upload consolidated file
        print(f"📤 Uploading merged file...")
        with open(filename, "rb") as f:
            file_obj = self.client.files.create(file=f, purpose="assistants")
        
        self.client.vector_stores.files.create(
            vector_store_id=self.vector_store_id,
            file_id=file_obj.id
        )
        
        self._wait_for_indexing(file_obj.id)
        
        # Update our list of archived files
        # Keep large consolidated files, remove merged ones, add new one
        merged_ids = {f["archive_id"] for f in files_to_merge}
        self.archived_files = [f for f in self.archived_files if f["archive_id"] not in merged_ids]
        
        self.archived_files.insert(0, {
            "file_id": file_obj.id,
            "archive_id": f"consolidated_{self.consolidation_count}",
            "filename": filename,
            "is_consolidated": True
        })
        
        print(f"✅ Consolidation #{self.consolidation_count} complete!")
        print(f"   {len(files_to_merge)} files → 1 file (all data preserved)")
        print(f"   Archive files now: {len(self.archived_files)}")
    
    def chat(self, user_message: str) -> str:
        """Send a message and get a response."""
        
        # Add user's message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Build messages array for API call
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.conversation_history)
        
        # Make API call with file_search if vector store exists and has content
        if self.vector_store_id and self.archive_count > 0:
            response = self.client.responses.create(
                model=MODEL,
                input=messages,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [self.vector_store_id]
                }]
            )
        else:
            response = self.client.responses.create(
                model=MODEL,
                input=messages
            )
        
        # Extract the response
        assistant_message = response.output_text
        
        # Add assistant's response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        # Check if we need to archive
        self.archive_messages()
        
        return assistant_message
    
    def get_status(self):
        """Get conversation status."""
        # Calculate total size of consolidated files
        total_consolidated_size = 0
        for f in self.archived_files:
            if f.get("is_consolidated"):
                try:
                    total_consolidated_size += os.path.getsize(f["filename"])
                except FileNotFoundError:
                    pass
        
        return {
            "active_messages": len(self.conversation_history),
            "archived_files": len(self.archived_files),
            "consolidations": self.consolidation_count,
            "total_messages": len(self.conversation_history) + (self.archive_count * ARCHIVE_COUNT),
            "consolidated_size_mb": total_consolidated_size / 1024 / 1024
        }


def main():
    print("=" * 60)
    print("🧠 Infinite Memory Chat - Proof of Concept")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  • Max {MAX_MESSAGES} active messages")
    print(f"  • Archive {ARCHIVE_COUNT} messages at a time")
    print(f"  • Consolidate at {MAX_ARCHIVE_FILES} files ({CONSOLIDATION_COUNT}→1)")
    print(f"  • Max file size: {MAX_FILE_SIZE_MB} MB")
    print(f"  • Theoretical max: ~{CONSOLIDATION_COUNT * MAX_FILE_SIZE_MB / 1024:.0f} GB history")
    print("Type 'quit' to exit, 'status' for statistics")
    print("=" * 60 + "\n")
    
    chat = InfiniteMemoryChat()
    chat.setup_vector_store()
    
    print("\n🚀 Ready to chat!\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == "status":
                status = chat.get_status()
                print(f"\n📊 Status:")
                print(f"   Active messages: {status['active_messages']}/{MAX_MESSAGES}")
                print(f"   Archive files in vector store: {status['archived_files']}/{MAX_ARCHIVE_FILES}")
                print(f"   Consolidations performed: {status['consolidations']}")
                print(f"   Consolidated data: {status['consolidated_size_mb']:.1f} MB / {CONSOLIDATION_COUNT * MAX_FILE_SIZE_MB / 1024:.0f} GB theoretical max")
                print(f"   Total messages (history): {status['total_messages']}\n")
                continue
            
            response = chat.chat(user_input)
            print(f"\nAssistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
