import os
import math
import torch
import torch.nn.functional as F
import pandas as pd
import re
from transformers import AutoTokenizer
from model import TransformerSummarizer

# ------------------------------------------------------
# Paths & Hyperparameters
# ------------------------------------------------------
CLEANED_FILE = "news_90k_clean.csv"
BEST_MODEL_PATH = os.path.join("checkpoints", "best_model.pth")

# Decoding constraints
ENC_MAX_LEN = 256     # max tokens for input article or extract
MAX_GEN_LEN = 32      # maximum generated tokens
NO_REPEAT_NGRAM_SIZE = 2
REPETITION_PENALTY = 2.0

# For beam search & sampling
BEAM_SIZE = 3
TOP_K = 50
TOP_P = 0.9
TEMPERATURE = 1.0

# Model hyperparameters (should match training)
D_MODEL = 256
NHEAD = 4
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------
# 1) Load Tokenizer & Model
# ------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("abhinand/malayalam-llama-7b-instruct-v0.1")
tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = tokenizer.vocab_size

model = TransformerSummarizer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    bos_id=tokenizer.bos_token_id,
    eos_id=tokenizer.eos_token_id,
    pad_id=tokenizer.pad_token_id
).to(DEVICE)

if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(f"Best model not found at {BEST_MODEL_PATH}")
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Loaded best model from {BEST_MODEL_PATH}")

# ------------------------------------------------------
# Helper: Apply No-Repeat N-Gram & Repetition Penalty
# ------------------------------------------------------
def apply_constraints(logits, tokens, no_repeat_ngram_size, repetition_penalty):
    # Repetition penalty
    if repetition_penalty != 1.0:
        for token_id in set(tokens):
            logits[token_id] /= repetition_penalty

    # No-repeat n-gram constraint
    if no_repeat_ngram_size > 0 and len(tokens) >= no_repeat_ngram_size:
        generated_ngrams = set()
        for i in range(len(tokens) - no_repeat_ngram_size + 1):
            ngram_tuple = tuple(tokens[i : i + no_repeat_ngram_size])
            generated_ngrams.add(ngram_tuple)
        for idx in range(len(logits)):
            potential_ngram = tuple(tokens[-(no_repeat_ngram_size - 1):] + [idx])
            if potential_ngram in generated_ngrams:
                logits[idx] = -float("inf")
    return logits

# ------------------------------------------------------
# 2) Decoding Methods for Abstractive Summarization
# ------------------------------------------------------
def greedy_decode(model, src_ids):
    model.eval()
    bos_id = model.bos_id
    eos_id = model.eos_id
    pad_id = model.pad_id

    with torch.no_grad():
        memory, _ = model.encode(src_ids, src_key_padding_mask=(src_ids == pad_id))
        generated = [bos_id]
        for _ in range(MAX_GEN_LEN):
            tgt = torch.tensor([generated], dtype=torch.long, device=DEVICE)
            tgt_mask = model._generate_square_subsequent_mask(tgt.size(1)).to(DEVICE)
            dec_output, _, _ = model.decode(
                tgt, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=(tgt == pad_id),
                memory_key_padding_mask=(src_ids == pad_id)
            )
            last_hidden = dec_output[-1, 0, :]
            logits = model.output_proj(last_hidden)
            logits = apply_constraints(logits, generated, NO_REPEAT_NGRAM_SIZE, REPETITION_PENALTY)
            next_token = torch.argmax(logits, dim=-1).item()
            generated.append(next_token)
            if next_token == eos_id:
                break

    if generated and generated[0] == bos_id:
        generated = generated[1:]
    if generated and generated[-1] == eos_id:
        generated = generated[:-1]
    return tokenizer.decode(generated, skip_special_tokens=True)

def beam_search_decode(model, src_ids, beam_size=BEAM_SIZE):
    model.eval()
    bos_id = model.bos_id
    eos_id = model.eos_id
    pad_id = model.pad_id

    with torch.no_grad():
        memory, _ = model.encode(src_ids, src_key_padding_mask=(src_ids == pad_id))
    beams = [([bos_id], 0.0, False)]
    for _ in range(MAX_GEN_LEN):
        new_beams = []
        all_finished = True
        for tokens, log_prob, finished in beams:
            if finished:
                new_beams.append((tokens, log_prob, True))
                continue
            all_finished = False
            tgt = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
            tgt_mask = model._generate_square_subsequent_mask(tgt.size(1)).to(DEVICE)
            with torch.no_grad():
                dec_output, _, _ = model.decode(
                    tgt, memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=(tgt == pad_id),
                    memory_key_padding_mask=(src_ids == pad_id)
                )
            last_hidden = dec_output[-1, 0, :]
            logits = model.output_proj(last_hidden)
            logits = apply_constraints(logits, tokens, NO_REPEAT_NGRAM_SIZE, REPETITION_PENALTY)
            probs = F.softmax(logits, dim=-1)
            topk = torch.topk(probs, beam_size)
            for prob, token_id in zip(topk.values, topk.indices):
                new_tokens = tokens + [token_id.item()]
                new_log_prob = log_prob + math.log(prob.item() + 1e-12)
                done = (token_id.item() == eos_id)
                new_beams.append((new_tokens, new_log_prob, done))
        if all_finished:
            break
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
    best_tokens, best_score, _ = sorted(beams, key=lambda x: x[1], reverse=True)[0]
    if best_tokens and best_tokens[0] == bos_id:
        best_tokens = best_tokens[1:]
    if best_tokens and best_tokens[-1] == eos_id:
        best_tokens = best_tokens[:-1]
    return tokenizer.decode(best_tokens, skip_special_tokens=True)

def top_k_sampling_decode(model, src_ids, top_k=TOP_K, temperature=TEMPERATURE):
    model.eval()
    bos_id = model.bos_id
    eos_id = model.eos_id
    pad_id = model.pad_id
    with torch.no_grad():
        memory, _ = model.encode(src_ids, src_key_padding_mask=(src_ids == pad_id))
    generated = [bos_id]
    for _ in range(MAX_GEN_LEN):
        tgt = torch.tensor([generated], dtype=torch.long, device=DEVICE)
        tgt_mask = model._generate_square_subsequent_mask(tgt.size(1)).to(DEVICE)
        with torch.no_grad():
            dec_output, _, _ = model.decode(
                tgt, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=(tgt == pad_id),
                memory_key_padding_mask=(src_ids == pad_id)
            )
        last_hidden = dec_output[-1, 0, :]
        logits = model.output_proj(last_hidden)
        logits = logits / temperature
        logits = apply_constraints(logits, generated, NO_REPEAT_NGRAM_SIZE, REPETITION_PENALTY)
        probs = F.softmax(logits, dim=-1)
        current_top_k = min(top_k, probs.size(-1))
        topk_vals, topk_inds = torch.topk(probs, current_top_k)
        topk_probs = topk_vals / torch.sum(topk_vals)
        chosen_idx = torch.multinomial(topk_probs, 1).item()
        next_token = topk_inds[chosen_idx].item()
        generated.append(next_token)
        if next_token == eos_id:
            break
    if generated and generated[0] == bos_id:
        generated = generated[1:]
    if generated and generated[-1] == eos_id:
        generated = generated[:-1]
    return tokenizer.decode(generated, skip_special_tokens=True)

def nucleus_sampling_decode(model, src_ids, top_p=TOP_P, temperature=TEMPERATURE):
    model.eval()
    bos_id = model.bos_id
    eos_id = model.eos_id
    pad_id = model.pad_id
    with torch.no_grad():
        memory, _ = model.encode(src_ids, src_key_padding_mask=(src_ids == pad_id))
    generated = [bos_id]
    for _ in range(MAX_GEN_LEN):
        tgt = torch.tensor([generated], dtype=torch.long, device=DEVICE)
        tgt_mask = model._generate_square_subsequent_mask(tgt.size(1)).to(DEVICE)
        with torch.no_grad():
            dec_output, _, _ = model.decode(
                tgt, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=(tgt == pad_id),
                memory_key_padding_mask=(src_ids == pad_id)
            )
        last_hidden = dec_output[-1, 0, :]
        logits = model.output_proj(last_hidden)
        logits = logits / temperature
        logits = apply_constraints(logits, generated, NO_REPEAT_NGRAM_SIZE, REPETITION_PENALTY)
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = (cumulative_probs > top_p).nonzero(as_tuple=True)[0]
        if len(cutoff) > 0:
            first_cut = cutoff[0].item()
            sorted_probs = sorted_probs[:first_cut+1]
            sorted_indices = sorted_indices[:first_cut+1]
            sorted_probs = sorted_probs / torch.sum(sorted_probs)
        chosen_idx = torch.multinomial(sorted_probs, 1).item()
        next_token = sorted_indices[chosen_idx].item()
        generated.append(next_token)
        if next_token == eos_id:
            break
    if generated and generated[0] == bos_id:
        generated = generated[1:]
    if generated and generated[-1] == eos_id:
        generated = generated[:-1]
    return tokenizer.decode(generated, skip_special_tokens=True)

# ------------------------------------------------------
# 3) Extractive Summarization Step
# ------------------------------------------------------
def extract_summary(paragraph: str, max_words=40):
    """
    Extracts the most salient sentence from the paragraph using common Malayalam punctuation
    as delimiters. If no clear sentence boundaries are found, returns the first max_words.
    """
    sentence_delimiters = r'[।.!?]'
    sentences = re.split(sentence_delimiters, paragraph)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        # Choose the longest sentence by word count
        longest = max(sentences, key=lambda s: len(s.split()))
        words = longest.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words])
        return longest
    else:
        return ' '.join(paragraph.split()[:max_words])

def extractive_abstractive_decode(paragraph: str, decoding_fn):
    """
    Performs extractive summarization first to select a salient sentence,
    then uses the given abstractive decoding function to generate a headline.
    """
    extract = extract_summary(paragraph)
    print("Extracted Summary for Abstractive Input:\n", extract, "\n")
    enc_ids = tokenizer.encode(extract, add_special_tokens=False, truncation=True, max_length=ENC_MAX_LEN)
    enc_tensor = torch.tensor([enc_ids], dtype=torch.long, device=DEVICE)
    return decoding_fn(model, enc_tensor)

# ------------------------------------------------------
# 4) Testing All Methods
# ------------------------------------------------------
def test_all_methods(paragraph: str):
    # Direct abstractive decoding on full paragraph
    enc_ids = tokenizer.encode(paragraph, add_special_tokens=False, truncation=True, max_length=ENC_MAX_LEN)
    enc_tensor = torch.tensor([enc_ids], dtype=torch.long, device=DEVICE)
    
    print("==== Direct Abstractive Decoding (Full Paragraph) ====\n")
    print("Greedy Decoding:")
    print(greedy_decode(model, enc_tensor), "\n")
    
    print("Beam Search Decoding:")
    print(beam_search_decode(model, enc_tensor, beam_size=BEAM_SIZE), "\n")
    
    print("Top-k Sampling Decoding:")
    print(top_k_sampling_decode(model, enc_tensor, top_k=TOP_K, temperature=TEMPERATURE), "\n")
    
    print("Nucleus (Top-p) Sampling Decoding:")
    print(nucleus_sampling_decode(model, enc_tensor, top_p=TOP_P, temperature=TEMPERATURE), "\n")
    
    print("==== Extractive + Abstractive Decoding ====\n")
    print("Extractive + Greedy:")
    print(extractive_abstractive_decode(paragraph, greedy_decode), "\n")
    
    print("Extractive + Beam Search:")
    print(extractive_abstractive_decode(paragraph, beam_search_decode), "\n")
    
    print("Extractive + Top-k Sampling:")
    print(extractive_abstractive_decode(paragraph, top_k_sampling_decode), "\n")
    
    print("Extractive + Nucleus Sampling:")
    print(extractive_abstractive_decode(paragraph, nucleus_sampling_decode), "\n")

# ------------------------------------------------------
# Main Testing Code
# ------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(CLEANED_FILE):
        raise FileNotFoundError(f"{CLEANED_FILE} not found!")
    df = pd.read_csv(CLEANED_FILE)
    if df.empty:
        raise ValueError("Cleaned CSV is empty!")

    # Use the first row for demonstration
    row = df.iloc[0]
    paragraph_text = row["para"]
    actual_headline = row["headline"]

    print("📝 Input Paragraph:\n", paragraph_text)
    print("\n🎯 Actual Headline:\n", actual_headline)
    print("============================================\n")
    
    test_all_methods(paragraph_text)
