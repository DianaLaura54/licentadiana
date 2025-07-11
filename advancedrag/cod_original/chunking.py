import os
import re
from collections import Counter

import numpy as np
import spacy

from embeddings import DEFAULT_EMBEDDING_MODEL

nlp = spacy.load("en_core_web_sm")

def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]



def chunk_documents_semantic(documents_with_pages, file_sources, embedding_model=None):
    if embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        except (ImportError, NameError):
            return chunk_documents(documents_with_pages, file_sources)
    all_chunks, all_metadata = [], []
    doc_index = 0
    for src_idx, doc_tuples in enumerate(documents_with_pages):
        source_file = file_sources[src_idx]
        for page_content, page_num in doc_tuples:
            if not page_content.strip():
                continue
            content = re.sub(r'\s+', ' ', page_content).strip()
            try:
                sentences = split_into_sentences(content)
            except Exception:
                sentences = re.split(r'(?<=[.!?])\s+', content)
            if not sentences:
                continue
            try:
                sentence_embeddings = embedding_model.encode(sentences)
            except Exception as e:
                print(f"Error generating embeddings: {str(e)}")
                sentence_embeddings = None
            min_chunk_size, max_chunk_size = 50, 250
            similarity_threshold = 0.5
            i = 0
            while i < len(sentences):
                while i < len(sentences):
                    if sentences[i] and re.match(r'^[A-Z]', sentences[i]):
                        break
                    i += 1
                if i >= len(sentences):
                    break
                current_chunk_sentences = [sentences[i]]
                current_words = len(sentences[i].split())
                current_idx = i
                i += 1
                while i < len(sentences):
                    sentence = sentences[i]
                    sentence_words = len(sentence.split())
                    if current_words + sentence_words > max_chunk_size:
                        break
                    if sentence_embeddings is not None:
                        try:
                            prev_embedding = sentence_embeddings[i - 1]
                            curr_embedding = sentence_embeddings[i]
                            similarity = np.dot(prev_embedding, curr_embedding) / (
                                np.linalg.norm(prev_embedding) * np.linalg.norm(curr_embedding) + 1e-8)
                            if similarity < similarity_threshold and current_words >= min_chunk_size:
                                break
                        except Exception:
                            pass
                    current_chunk_sentences.append(sentence)
                    current_words += sentence_words
                    i += 1
                if current_chunk_sentences and current_words >= min_chunk_size:
                    chunk_text = " ".join(current_chunk_sentences)
                    all_chunks.append(chunk_text)
                    all_metadata.append({
                        "source": source_file,
                        "page": page_num,
                        "doc_index": doc_index,
                        "sentences": len(current_chunk_sentences),
                        "words": current_words,
                        "chunking_method": "semantic"
                    })
                    doc_index += 1
    return all_chunks, all_metadata





def chunk_documents(documents_with_pages, file_sources):
    all_chunks, all_metadata = [], []
    doc_index = 0
    for src_idx, doc_tuples in enumerate(documents_with_pages):
        source_file = file_sources[src_idx]
        for content, page_info in doc_tuples:
            content = re.sub(r'\s+', ' ', content).strip()
            try:
                sentences = split_into_sentences(content)
            except Exception:
                raw_sentences = re.split(r'\.(?=\s|$)', content)
                sentences = [s.strip() + '.' if not s.strip().endswith('.') else s.strip() for s in raw_sentences if s.strip()]
            if not sentences:
                continue
            sentence_pages = []
            if isinstance(page_info, list):
                page_markers = page_info
                content_length = len(content)
                sentence_start_pos = 0
                for sentence in sentences:
                    sentence_length = len(sentence)
                    sentence_middle_pos = sentence_start_pos + (sentence_length // 2)
                    page_num = 1
                    for pos, page in page_markers:
                        if sentence_middle_pos >= pos:
                            page_num = page
                        else:
                            break
                    sentence_pages.append(page_num)
                    sentence_start_pos += sentence_length + 1
            else:
                page_num = page_info
                sentence_pages = [page_num] * len(sentences)
            current_chunk, current_sentences, current_pages = "", [], []
            min_chunk_words, max_chunk_words, ideal_chunk_words = 50, 250, 150
            for i, sentence in enumerate(sentences):
                sentence_words = len(sentence.split())
                current_words = len(current_chunk.split()) if current_chunk else 0
                if current_words > 0 and current_words + sentence_words > max_chunk_words:
                    if current_words >= min_chunk_words:
                        all_chunks.append(current_chunk)
                        if current_pages:
                            page_counter = Counter(current_pages)
                            most_common_page = page_counter.most_common(1)[0][0]
                        else:
                            most_common_page = 1
                        all_metadata.append({
                            "source": source_file,
                            "page": most_common_page,
                            "doc_index": doc_index,
                            "sentences": len(current_sentences),
                            "words": current_words,
                            "chunking_method": "standard"
                        })
                        doc_index += 1
                        current_chunk = sentence
                        current_sentences = [sentence]
                        current_pages = [sentence_pages[i]]
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
                        current_sentences.append(sentence)
                        current_pages.append(sentence_pages[i])
                else:
                    current_chunk = current_chunk + " " + sentence if current_chunk else sentence
                    current_sentences.append(sentence)
                    current_pages.append(sentence_pages[i])
                if len(current_chunk.split()) > ideal_chunk_words and current_chunk.endswith('.'):
                    all_chunks.append(current_chunk)
                    if current_pages:
                        page_counter = Counter(current_pages)
                        most_common_page = page_counter.most_common(1)[0][0]
                    else:
                        most_common_page = 1
                    all_metadata.append({
                        "source": source_file,
                        "page": most_common_page,
                        "doc_index": doc_index,
                        "sentences": len(current_sentences),
                        "words": len(current_chunk.split()),
                        "chunking_method": "standard"
                    })
                    doc_index += 1
                    current_chunk, current_sentences, current_pages = "", [], []
            if current_chunk:
                all_chunks.append(current_chunk)
                if current_pages:
                    page_counter = Counter(current_pages)
                    most_common_page = page_counter.most_common(1)[0][0]
                else:
                    most_common_page = 1
                all_metadata.append({
                    "source": source_file,
                    "page": most_common_page,
                    "doc_index": doc_index,
                    "sentences": len(current_sentences),
                    "words": len(current_chunk.split()),
                    "chunking_method": "standard"
                })
                doc_index += 1
    return all_chunks, all_metadata
