"""
Document Processing Pipeline for RAG System
Handles HTML and PDF processing with structure-aware chunking
"""

import json
import re
import unicodedata
import hashlib
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import logging

# HTML Processing
try:
    import trafilatura
    from readability import Document as ReadabilityDocument
    from bs4 import BeautifulSoup
except ImportError as e:
    logging.warning(f"HTML processing libraries not available: {e}")

# PDF Processing
try:
    import fitz  # PyMuPDF
    import pdfplumber
except ImportError as e:
    logging.warning(f"PDF processing libraries not available: {e}")

# Text Processing
try:
    from langdetect import detect as detect_language
    from langdetect.lang_detect_exception import LangDetectException
except ImportError:
    def detect_language(text):
        return "en"  # Fallback
    LangDetectException = Exception

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Document processing pipeline with HTML and PDF support"""

    def __init__(self, processed_data_dir: Path, processed_docs_file: str = "processed_docs.json"):
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_docs_file = processed_docs_file
        self.processed_docs = {}
        self._load_existing_docs()

    def _load_existing_docs(self):
        """Load previously processed documents"""
        processed_file = self.processed_data_dir / self.processed_docs_file
        if processed_file.exists():
            try:
                with open(processed_file, 'r', encoding='utf-8') as f:
                    self.processed_docs = json.load(f)
                logger.info(f"Loaded {len(self.processed_docs)} existing processed documents")
            except Exception as e:
                logger.warning(f"Could not load existing documents: {e}")

    def _save_processed_docs(self):
        """Save processed documents to disk"""
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        processed_file = self.processed_data_dir / self.processed_docs_file
        try:
            with open(processed_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_docs, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.processed_docs)} processed documents to {processed_file}")
        except Exception as e:
            logger.error(f"Failed to save processed documents: {e}")

    def process_directory(self, directory: Path, force_reprocess: bool = False) -> Dict:
        """Process all documents in a directory recursively"""
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        # Find all supported files recursively
        supported_extensions = {'.html', '.htm', '.pdf', '.txt'}
        files = [p for p in directory.rglob("*") if p.suffix.lower() in supported_extensions]

        logger.info(f"📂 Found {len(files)} files in {directory} to process")

        processed_count = 0
        document_ids = []

        for file_path in files:
            try:
                doc_id = self._generate_doc_id(file_path)

                if not force_reprocess and doc_id in self.processed_docs:
                    logger.info(f"⏭ Skipping already processed: {file_path.name}")
                    document_ids.append(doc_id)
                    continue

                logger.info(f"⚙️ Processing: {file_path.name}")
                chunks = self.process_document(file_path)

                if chunks:
                    self.processed_docs[doc_id] = {
                        "doc_id": doc_id,
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "processed_at": datetime.now().isoformat(),
                        "chunk_count": len(chunks),
                        "chunks": chunks
                    }
                    processed_count += 1
                    document_ids.append(doc_id)
                    logger.info(f"✅ Processed {file_path.name}: {len(chunks)} chunks")
                else:
                    logger.warning(f"⚠️ No chunks extracted from {file_path.name}")

            except Exception as e:
                logger.error(f"❌ Failed to process {file_path}: {e}")

        # Save processed documents
        self._save_processed_docs()

        result = {
            "processed_count": processed_count,
            "total_count": len(files),
            "document_ids": document_ids
        }
        
        logger.info(f"✅ Processing complete: {processed_count}/{len(files)} files processed")
        return result

    def process_document(self, file_path: Path) -> List[Dict]:
        """Process a single document"""
        ext = file_path.suffix.lower()
        if ext in ['.html', '.htm']:
            return self._process_html(file_path)
        elif ext == '.pdf':
            return self._process_pdf(file_path)
        elif ext == '.txt':
            return self._process_text(file_path)
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return []

    def _process_html(self, file_path: Path) -> List[Dict]:
        """Process HTML document"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
        except Exception as e:
            logger.error(f"Failed to read HTML file {file_path}: {e}")
            return []

        extracted_text = None
        
        # Try trafilatura first (best for content extraction)
        if 'trafilatura' in globals():
            try:
                extracted_text = trafilatura.extract(html_content)
                if extracted_text and len(extracted_text.strip()) > 100:
                    logger.debug(f"Trafilatura extracted {len(extracted_text)} chars from {file_path.name}")
                else:
                    extracted_text = None
            except Exception as e:
                logger.debug(f"Trafilatura failed for {file_path.name}: {e}")

        # Try Readability as fallback
        if not extracted_text and 'ReadabilityDocument' in globals():
            try:
                doc = ReadabilityDocument(html_content)
                extracted_text = BeautifulSoup(doc.summary(), 'html.parser').get_text()
                if extracted_text and len(extracted_text.strip()) > 100:
                    logger.debug(f"Readability extracted {len(extracted_text)} chars from {file_path.name}")
                else:
                    extracted_text = None
            except Exception as e:
                logger.debug(f"Readability failed for {file_path.name}: {e}")

        # Final fallback to BeautifulSoup
        if not extracted_text:
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                extracted_text = soup.get_text()
                logger.debug(f"BeautifulSoup extracted {len(extracted_text)} chars from {file_path.name}")
            except Exception as e:
                logger.error(f"BeautifulSoup failed for {file_path.name}: {e}")
                return []

        if not extracted_text:
            logger.warning(f"No text extracted from {file_path.name}")
            return []

        cleaned_text = self._clean_text(extracted_text)
        
        # Check if we have enough content
        if len(cleaned_text.split()) < 20:
            logger.warning(f"Insufficient content in {file_path.name}: {len(cleaned_text.split())} words")
            return []

        metadata = {"file_name": file_path.name, "source_type": "html"}
        chunks = self._create_chunks(cleaned_text, metadata)
        
        logger.debug(f"Created {len(chunks)} chunks from {file_path.name}")
        return chunks

    def _process_pdf(self, file_path: Path) -> List[Dict]:
        """Process PDF with PyMuPDF"""
        chunks = []
        
        if 'fitz' not in globals():
            logger.warning("PyMuPDF not available for PDF processing")
            return []
            
        try:
            doc = fitz.open(file_path)
            logger.debug(f"Processing PDF {file_path.name} with {len(doc)} pages")
            
            for page_num in range(len(doc)):
                try:
                    text = doc[page_num].get_text()
                    cleaned_text = self._clean_text(text)
                    
                    if len(cleaned_text.split()) > 20:
                        metadata = {
                            "file_name": file_path.name,
                            "page": page_num + 1,
                            "source_type": "pdf"
                        }
                        page_chunks = self._create_chunks(cleaned_text, metadata)
                        chunks.extend(page_chunks)
                        logger.debug(f"Page {page_num + 1}: {len(page_chunks)} chunks")
                    else:
                        logger.debug(f"Page {page_num + 1}: insufficient content ({len(cleaned_text.split())} words)")
                        
                except Exception as e:
                    logger.warning(f"Failed to process page {page_num + 1} of {file_path.name}: {e}")
                    
            doc.close()
            
        except Exception as e:
            logger.error(f"Failed to process PDF {file_path.name}: {e}")
            
        return chunks

    def _process_text(self, file_path: Path) -> List[Dict]:
        """Process plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = self._clean_text(f.read())
        except Exception as e:
            logger.error(f"Failed to read text file {file_path}: {e}")
            return []
            
        if len(content.split()) < 20:
            logger.warning(f"Insufficient content in {file_path.name}: {len(content.split())} words")
            return []
            
        metadata = {"file_name": file_path.name, "source_type": "txt"}
        return self._create_chunks(content, metadata)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
            
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common HTML entities that might remain
        text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
        
        # Clean up whitespace again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _create_chunks(self, text: str, base_metadata: Dict) -> List[Dict]:
        """Create chunks from text with improved strategy"""
        if not text or len(text.strip()) < 50:
            return []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        target_chunk_size = 200  # words per chunk
        max_chunk_size = 300     # maximum words per chunk
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed max size, save current chunk
            if current_word_count + sentence_words > max_chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text.split()) >= 50:  # Minimum viable chunk size
                    chunks.append({
                        "content": chunk_text,
                        "metadata": {**base_metadata, "chunk_id": len(chunks)}
                    })
                current_chunk = [sentence]
                current_word_count = sentence_words
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words
                
                # If we've reached target size, save chunk
                if current_word_count >= target_chunk_size:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "content": chunk_text,
                        "metadata": {**base_metadata, "chunk_id": len(chunks)}
                    })
                    current_chunk = []
                    current_word_count = 0
        
        # Handle remaining content
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text.split()) >= 30:  # Lower threshold for final chunk
                chunks.append({
                    "content": chunk_text,
                    "metadata": {**base_metadata, "chunk_id": len(chunks)}
                })
        
        logger.debug(f"Created {len(chunks)} chunks with avg {sum(len(c['content'].split()) for c in chunks) / len(chunks) if chunks else 0:.1f} words each")
        return chunks

    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate unique document ID from file path"""
        return hashlib.md5(str(file_path).encode()).hexdigest()[:16]

    def get_processed_documents(self) -> Dict:
        """Get all processed documents"""
        return self.processed_docs

    def get_document_by_id(self, doc_id: str) -> Dict:
        """Get specific document by ID"""
        return self.processed_docs.get(doc_id, {})

    def clear_processed_documents(self):
        """Clear all processed documents"""
        self.processed_docs = {}
        self._save_processed_docs()
        logger.info("Cleared all processed documents")

    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        if not self.processed_docs:
            return {"total_documents": 0, "total_chunks": 0}
        
        total_chunks = sum(doc.get("chunk_count", 0) for doc in self.processed_docs.values())
        
        return {
            "total_documents": len(self.processed_docs),
            "total_chunks": total_chunks,
            "documents_by_type": self._get_type_breakdown(),
            "avg_chunks_per_doc": total_chunks / len(self.processed_docs) if self.processed_docs else 0
        }

    def _get_type_breakdown(self) -> Dict:
        """Get breakdown of documents by type"""
        type_counts = {}
        for doc in self.processed_docs.values():
            chunks = doc.get("chunks", [])
            if chunks and chunks[0].get("metadata"):
                source_type = chunks[0]["metadata"].get("source_type", "unknown")
                type_counts[source_type] = type_counts.get(source_type, 0) + 1
        return type_counts