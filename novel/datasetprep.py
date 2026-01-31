"""
Dataset Preprocessing for Latent Predictor SentenceFormer
- NLTK-based sentence tokenization (better than naive period splitting)
- Merges adjacent short sentences when combined tokens < MAX_TOKENS
- Filters rows where any sentence exceeds token limits
- Estimates token count using word count ratio before tokenization
- Post-tokenization verification and filtering
"""

import torch
import os
import re
import nltk
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from tqdm import tqdm
import math

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
DATASET_NAME = "roneneldan/TinyStories"  # HuggingFace dataset
OUTPUT_DIR = "processed_shards"
NUM_SHARDS = 5
MAX_TRAIN_ROWS = 25000
MAX_VAL_ROWS = 5000

# Architecture Constraints (Must match SentenceFormer config)
MAX_SENTS_PER_STORY = 16
MAX_TOKENS_PER_SENT = 48  # Including <compress> token

# Token Estimation Ratio (word_count * RATIO ≈ token_count)
# Empirically, GPT-2 tokenizer: 1 word ≈ 1.3 tokens on average
TOKEN_ESTIMATION_RATIO = 1.3

# ============================================================================
# 2. SETUP
# ============================================================================
print(">>> Initializing Resources...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

# Tokenizer Setup
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<compress>"]})
COMPRESS_TOKEN_ID = tokenizer.convert_tokens_to_ids("<compress>")

print(f"Compress Token ID: {COMPRESS_TOKEN_ID}")
print(f"Pad Token ID: {tokenizer.pad_token_id}")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 3. HELPER FUNCTIONS
# ============================================================================

def clean_sentence(text):
    """Removes newlines, collapses whitespace, fixes punctuation."""
    text = text.replace('\n', ' ')
    text = text.replace('.,', ',')
    text = text.replace('..', '.')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def estimate_tokens(text):
    """Estimates token count using word count."""
    word_count = len(text.split())
    return int(word_count * TOKEN_ESTIMATION_RATIO)

def merge_short_sentences(sentences):
    """
    Merges adjacent sentences when combined estimated tokens <= MAX_TOKENS_PER_SENT.
    Only merges pairs (not chains).
    """
    merged = []
    i = 0
    
    while i < len(sentences):
        current = sentences[i].strip()
        current_est = estimate_tokens(current)
        
        # Try to merge with next sentence
        if i + 1 < len(sentences):
            next_sent = sentences[i + 1].strip()
            next_est = estimate_tokens(next_sent)
            
            # Check if merging is beneficial (combined estimate fits)
            combined_est = current_est + next_est
            if combined_est <= (MAX_TOKENS_PER_SENT-5):
                # Merge and skip next
                merged.append(current + " " + next_sent)
                i += 2
                continue
        
        # No merge - keep current
        merged.append(current)
        i += 1
    
    return merged

def process_single_story(raw_text):
    """
    Processes a single text into tokenized sentences.
    Returns: List of token lists, or None if story is invalid.
    """
    try:
        # 1. Sentence Tokenization (NLTK)
        raw_sentences = sent_tokenize(raw_text)
    except Exception:
        return None
    
    if not raw_sentences:
        return None
    
    # 2. Merge Short Adjacent Sentences
    merged_sentences = merge_short_sentences(raw_sentences)
    
    # 3. Tokenize & Validate Each Sentence
    processed_story = []
    
    for sent in merged_sentences:
        cleaned = clean_sentence(sent)
        if not cleaned:
            continue
        
        try:
            tokens = tokenizer.encode(cleaned)
        except Exception:
            continue
        
        # Filter: too short or too long (need room for <compress>)
        # Must have: 3 <= len(tokens) < MAX_TOKENS_PER_SENT
        # After adding <compress>: 4 <= len(tokens) <= MAX_TOKENS_PER_SENT
        if len(tokens) < 3:
            continue  # Too short
        
        if len(tokens) >= MAX_TOKENS_PER_SENT:
            # Sentence too long even before <compress>
            # Skip this sentence but continue with story
            continue
        
        # Append compress token
        tokens.append(COMPRESS_TOKEN_ID)
        processed_story.append(tokens)
    
    # 4. Validate Story Length
    if len(processed_story) == 0:
        return None  # No valid sentences
    
    if len(processed_story) > MAX_SENTS_PER_STORY:
        return None  # Too many sentences
    
    return processed_story

# ============================================================================
# 4. MAIN PROCESSING LOOP
# ============================================================================

def process_dataset_from_huggingface(dataset_name, split, max_rows, output_dir, num_shards):
    """
    Processes HuggingFace dataset and splits into shards.
    """
    print(f">>> Loading {split} dataset from HuggingFace: {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")
    
    total_rows = max_rows
    
    # Calculate shard size
    shard_size = math.ceil(total_rows / num_shards)
    print(f"Target Rows: {total_rows} | Shards: {num_shards} | Rows per Shard: ~{shard_size}")
    
    # Processing state
    current_shard_data = []
    shard_index = 0
    total_processed = 0
    skipped_count = 0
    
    print(">>> Starting Processing...")
    
    dataset_iter = iter(dataset)
    count = 0
    
    while count < total_rows:
        try:
            row = next(dataset_iter)
            text = row.get("text", "")
            
            if not text or len(text) < 50 or len(text) > 2000:
                skipped_count += 1
                continue
            
        except StopIteration:
            print("Dataset exhausted before reaching target rows")
            break
        except Exception:
            skipped_count += 1
            continue
        
        # Process story
        processed_story = process_single_story(text)
        
        if processed_story is None:
            skipped_count += 1
            continue
        
        # Add to current shard
        current_shard_data.append(processed_story)
        total_processed += 1
        count += 1
        
        if count % 1000 == 0:
            print(f"Processed {count}/{total_rows} stories...")
        
        # Save shard when full
        if len(current_shard_data) >= shard_size:
            shard_path = os.path.join(output_dir, f"{split}_shard_{shard_index}.pt")
            torch.save(current_shard_data, shard_path)
            print(f"\n✅ Saved {split}_shard_{shard_index}.pt ({len(current_shard_data)} stories)")
            
            current_shard_data = []
            shard_index += 1
    
    # Save remaining data
    if current_shard_data:
        shard_path = os.path.join(output_dir, f"{split}_shard_{shard_index}.pt")
        torch.save(current_shard_data, shard_path)
        print(f"\n✅ Saved {split}_shard_{shard_index}.pt ({len(current_shard_data)} stories)")
    
    # Summary
    print("\n" + "="*60)
    print(f"✅ {split.upper()} Processing Complete!")
    print(f"Target Rows: {total_rows}")
    print(f"Successfully Processed: {total_processed}")
    print(f"Skipped (invalid): {skipped_count}")
    print(f"Shards Created: {shard_index + 1}")
    print("="*60)

# ============================================================================
# 5. VALIDATION FUNCTION
# ============================================================================

def validate_shards(output_dir):
    """
    Validates processed shards for correctness.
    """
    print("\n>>> Validating Shards...")
    
    shard_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".pt")])
    
    total_stories = 0
    total_sentences = 0
    max_story_sents = 0
    max_sent_tokens = 0
    
    for shard_file in shard_files:
        shard_path = os.path.join(output_dir, shard_file)
        shard_data = torch.load(shard_path)
        
        total_stories += len(shard_data)
        
        for story in shard_data:
            num_sents = len(story)
            total_sentences += num_sents
            max_story_sents = max(max_story_sents, num_sents)
            
            for sentence_tokens in story:
                sent_len = len(sentence_tokens)
                max_sent_tokens = max(max_sent_tokens, sent_len)
                
                # Validate compress token
                if sentence_tokens[-1] != COMPRESS_TOKEN_ID:
                    print(f"⚠️ Warning: Sentence missing <compress> token!")
                
                # Validate length
                if sent_len > MAX_TOKENS_PER_SENT:
                    print(f"⚠️ Warning: Sentence length {sent_len} exceeds {MAX_TOKENS_PER_SENT}!")
    
    print(f"\n📊 Validation Results:")
    print(f"  Total Stories: {total_stories}")
    print(f"  Total Sentences: {total_sentences}")
    print(f"  Avg Sentences/Story: {total_sentences/total_stories:.2f}")
    print(f"  Max Sentences in Story: {max_story_sents} (limit: {MAX_SENTS_PER_STORY})")
    print(f"  Max Tokens in Sentence: {max_sent_tokens} (limit: {MAX_TOKENS_PER_SENT})")
    
    if max_story_sents <= MAX_SENTS_PER_STORY and max_sent_tokens <= MAX_TOKENS_PER_SENT:
        print(f"✅ All constraints satisfied!")
    else:
        print(f"❌ Constraint violations detected!")

# ============================================================================
# 6. EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Process train split
    print("\n" + "="*60)
    print("PROCESSING TRAIN SPLIT")
    print("="*60)
    process_dataset_from_huggingface(
        DATASET_NAME, 
        "train", 
        MAX_TRAIN_ROWS, 
        OUTPUT_DIR, 
        NUM_SHARDS
    )
    
    # Process validation split
    print("\n" + "="*60)
    print("PROCESSING VALIDATION SPLIT")
    print("="*60)
    process_dataset_from_huggingface(
        DATASET_NAME, 
        "validation", 
        MAX_VAL_ROWS, 
        OUTPUT_DIR, 
        NUM_SHARDS
    )
    
    # Validate output
    validate_shards(OUTPUT_DIR)
    
    print("\n✅ Preprocessing complete! Shards ready for training.")