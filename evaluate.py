import os
import math
import torch
import torch.nn.functional as F
import pandas as pd
import re
from tqdm import tqdm
from transformers import AutoTokenizer
from model import TransformerSummarizer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sacrebleu
from bert_score import score as bert_score

# -------------------------------
# Hyperparameters & Paths
# -------------------------------
CLEANED_FILE = "news_90k_clean.csv"
BEST_MODEL_PATH = os.path.join("checkpoints", "best_model.pth")
EVAL_FRAC = 0.10  # use 10% of data for evaluation

# Decoding constraints
ENC_MAX_LEN = 256     # maximum tokens for source/extract
MAX_GEN_LEN = 32      # maximum tokens for generation
NO_REPEAT_NGRAM_SIZE = 2
REPETITION_PENALTY = 2.0

# Sampling parameters
BEAM_SIZE = 3
TOP_K = 50
TOP_P = 0.9
TEMPERATURE = 1.0

# Model hyperparameters (must match training)
D_MODEL = 256
NHEAD = 4
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 1) Load Tokenizer & Model
# -------------------------------
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

# -------------------------------
# 2) Decoding Methods (Functions)
# -------------------------------

def apply_constraints(logits, tokens, no_repeat_ngram_size, repetition_penalty):
    # Repetition penalty: penalize tokens already generated.
    if repetition_penalty != 1.0:
        for token_id in set(tokens):
            logits[token_id] /= repetition_penalty
    # No-repeat n-gram constraint
    if no_repeat_ngram_size > 0 and len(tokens) >= no_repeat_ngram_size:
        generated_ngrams = set()
        for i in range(len(tokens) - no_repeat_ngram_size + 1):
            ngram_tuple = tuple(tokens[i:i+no_repeat_ngram_size])
            generated_ngrams.add(ngram_tuple)
        for idx in range(len(logits)):
            potential_ngram = tuple(tokens[-(no_repeat_ngram_size-1):] + [idx])
            if potential_ngram in generated_ngrams:
                logits[idx] = -float("inf")
    return logits

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
# 3) Extractive Summarization Functions
# ------------------------------------------------------
def extract_summary(paragraph: str, max_words=40):
    """
    A simple extractive summarization: splits the paragraph using common Malayalam punctuation 
    delimiters and selects the longest sentence (or trims to max_words if necessary).
    """
    sentence_delimiters = r'[।.!?]'
    sentences = re.split(sentence_delimiters, paragraph)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        longest = max(sentences, key=lambda s: len(s.split()))
        words = longest.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words])
        return longest
    else:
        return ' '.join(paragraph.split()[:max_words])

def extractive_abstractive_decode(paragraph: str, decoding_fn):
    extract = extract_summary(paragraph)
    print("Extracted Summary for Abstractive Input:\n", extract, "\n")
    enc_ids = tokenizer.encode(extract, add_special_tokens=False, truncation=True, max_length=ENC_MAX_LEN)
    enc_tensor = torch.tensor([enc_ids], dtype=torch.long, device=DEVICE)
    return decoding_fn(model, enc_tensor)

# ------------------------------------------------------
# 4) Evaluation Metrics Functions
# ------------------------------------------------------
def compute_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

def compute_bleu(reference, hypothesis):
    # Using NLTK's sentence_bleu with smoothing
    smoothing_fn = SmoothingFunction().method1
    # Tokenize by whitespace
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    bleu_score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing_fn)
    return bleu_score * 100  # scale to percentage

def compute_indic_bleu(reference, hypothesis):
    # Using sacreBLEU at sentence level; note that sacreBLEU expects list of references.
    bleu = sacrebleu.sentence_bleu(hypothesis, [reference]).score
    return bleu

def compute_bert_score(reference, hypothesis):
    # Compute BERTScore using a multilingual model (lang "ml" for Malayalam)
    P, R, F1 = bert_score([hypothesis], [reference], lang="ml", verbose=False)
    # Return F1 score as a float
    return F1.item()

# ------------------------------------------------------
# 5) Evaluation Loop over 10% of Dataset
# ------------------------------------------------------
results = []  # list to store evaluation results
methods = {
    "Greedy": greedy_decode,
    "Beam": beam_search_decode,
    "Top-k": top_k_sampling_decode,
    "Nucleus": nucleus_sampling_decode,
    "Extractive+Greedy": lambda model, src: extractive_abstractive_decode(src, greedy_decode),
    "Extractive+Beam": lambda model, src: extractive_abstractive_decode(src, beam_search_decode),
    "Extractive+Top-k": lambda model, src: extractive_abstractive_decode(src, top_k_sampling_decode),
    "Extractive+Nucleus": lambda model, src: extractive_abstractive_decode(src, nucleus_sampling_decode),
}

# Load evaluation data (10% of the cleaned file)
if not os.path.exists(CLEANED_FILE):
    raise FileNotFoundError(f"{CLEANED_FILE} not found!")
df = pd.read_csv(CLEANED_FILE)
if df.empty:
    raise ValueError("Cleaned CSV is empty!")
eval_df = df.sample(frac=EVAL_FRAC, random_state=42).reset_index(drop=True)
print(f"Evaluating on {len(eval_df)} samples...")

# Iterate over evaluation samples with a progress bar
for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating Samples"):
    paragraph = row["para"]
    reference = row["headline"]
    
    # We'll store one row per method per sample.
    for method_name, decode_fn in methods.items():
        # For direct abstractive methods, use the full paragraph
        # For extractive methods, the function extractive_abstractive_decode will print the extract.
        try:
            if "Extractive" in method_name:
                generated = decode_fn(model, paragraph)
            else:
                enc_ids = tokenizer.encode(paragraph, add_special_tokens=False, truncation=True, max_length=ENC_MAX_LEN)
                enc_tensor = torch.tensor([enc_ids], dtype=torch.long, device=DEVICE)
                generated = decode_fn(model, enc_tensor)
        except Exception as e:
            generated = f"Error: {str(e)}"
        
        # Compute metrics
        rouge1, rouge2, rougel = compute_rouge(reference, generated)
        bleu = compute_bleu(reference, generated)
        indic_bleu = compute_indic_bleu(reference, generated)
        bert = compute_bert_score(reference, generated)
        
        results.append({
            "sample_id": idx,
            "method": method_name,
            "reference": reference,
            "generated": generated,
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougel": rougel,
            "bleu": bleu,
            "indic_bleu": indic_bleu,
            "indic_bert": bert,
        })

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
output_csv = "evaluation_results.csv"
results_df.to_csv(output_csv, index=False)
print(f"Evaluation results saved to {output_csv}")

# Optionally, compute average metrics per method and print them
avg_results = results_df.groupby("method").mean().reset_index()
print("Average Evaluation Metrics per Method:")
print(avg_results[["method", "rouge1", "rouge2", "rougel", "bleu", "indic_bleu", "indic_bert"]])
