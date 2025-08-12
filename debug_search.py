"""
Debug script to check what content is actually retrieved for Hero Wars queries
"""

import json
from pathlib import Path
from config import config

def debug_hero_wars_content():
    """Check what Hero Wars content is actually processed"""
    
    # Load processed documents
    processed_file = config.processed_data_dir / config.processed_docs_file
    
    if not processed_file.exists():
        print("❌ No processed documents found. Run ingestion first.")
        return
    
    with open(processed_file, 'r', encoding='utf-8') as f:
        all_docs = json.load(f)
    
    print(f"📊 Total processed documents: {len(all_docs)}")
    
    # Find Hero Wars document
    hero_wars_docs = []
    for doc_id, doc_data in all_docs.items():
        file_name = doc_data.get("file_name", "")
        if "hero-wars" in file_name.lower():
            hero_wars_docs.append((doc_id, doc_data))
    
    if not hero_wars_docs:
        print("❌ No Hero Wars documents found in processed data")
        print("Available documents:")
        for doc_id, doc_data in all_docs.items():
            print(f"  - {doc_data.get('file_name', 'unknown')}")
        return
    
    # Analyze Hero Wars content
    for doc_id, doc_data in hero_wars_docs:
        print(f"\n📄 Document: {doc_data.get('file_name')}")
        print(f"🆔 Doc ID: {doc_id}")
        print(f"📊 Chunk count in metadata: {doc_data.get('chunk_count', 0)}")
        
        chunks = doc_data.get("chunks", [])
        print(f"📊 Actual chunks found: {len(chunks)}")
        
        # Check chunk structure
        valid_chunks = 0
        for i, chunk in enumerate(chunks):
            if chunk and isinstance(chunk, dict):
                if chunk.get("content"):
                    valid_chunks += 1
                    if i < 3:  # Show first 3 chunks
                        content = chunk["content"]
                        metadata = chunk.get("metadata", {})
                        print(f"\n--- Chunk {i} ---")
                        print(f"Content length: {len(content)}")
                        print(f"Has metadata: {bool(metadata)}")
                        print(f"Content preview: {content[:150]}...")
                        
                        # Check for keywords
                        keywords = ["dau", "revenue", "live event", "strategy", "increase", "august", "june"]
                        content_lower = content.lower()
                        matches = [kw for kw in keywords if kw in content_lower]
                        if matches:
                            print(f"Keyword matches: {matches}")
                else:
                    print(f"❌ Chunk {i}: No content")
            else:
                print(f"❌ Chunk {i}: Invalid structure - {type(chunk)}")
        
        print(f"\n✅ Valid chunks: {valid_chunks}/{len(chunks)}")
        
        # If no valid chunks, check raw file
        if valid_chunks == 0:
            print(f"\n🔍 Checking raw HTML file...")
            raw_file = config.raw_data_dir / doc_data.get('file_name', '')
            if raw_file.exists():
                try:
                    with open(raw_file, 'r', encoding='utf-8') as f:
                        raw_content = f.read()
                    print(f"Raw file size: {len(raw_content)} characters")
                    print(f"Contains 'Hero Wars': {'Hero Wars' in raw_content}")
                    print(f"Contains 'DAU': {'DAU' in raw_content}")
                    print(f"Contains 'revenue': {'revenue' in raw_content.lower()}")
                except Exception as e:
                    print(f"❌ Could not read raw file: {e}")
            else:
                print(f"❌ Raw file not found: {raw_file}")
    
    # Test search directly if chunks exist
    if valid_chunks > 0:
        print("\n" + "="*60)
        print("🔍 TESTING DIRECT SEARCH")
        print("="*60)
        
        try:
            from retriever import HybridRetriever
            
            retriever = HybridRetriever(config)
            
            test_queries = [
                "Hero Wars live events DAU revenue",
                "Hero Wars strategies increase revenue",
                "live event strategies DAU",
                "Hero Wars August June 2024 2025"
            ]
            
            for query in test_queries:
                print(f"\n🔍 Query: '{query}'")
                results = retriever.search(query, mode="hybrid", k=3)
                
                if results:
                    print(f"   Found {len(results)} results:")
                    for i, result in enumerate(results):
                        score = result.get("score", 0)
                        content = result.get("content", "")[:100] + "..."
                        metadata = result.get("metadata", {})
                        file_name = metadata.get("file_name", "unknown")
                        print(f"     {i+1}. Score: {score:.3f} | File: {file_name}")
                        print(f"        Content: {content}")
                else:
                    print("   ❌ No results found")
                    
        except Exception as e:
            print(f"❌ Search test failed: {e}")
    else:
        print("\n❌ No valid chunks to test search with")

if __name__ == "__main__":
    debug_hero_wars_content()