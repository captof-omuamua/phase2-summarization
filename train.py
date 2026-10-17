import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model import TransformerSummarizer

# -------------------------------
# Paths and Hyperparameters
# -------------------------------
CLEANED_FILE = "news_90k_clean.csv"    # cleaned file from preprocess.py
CHECKPOINT_DIR = "checkpoints"
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# Initialize the abhinand Malayalam LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained("abhinand/malayalam-llama-7b-instruct-v0.1")
tokenizer.pad_token = tokenizer.eos_token

# Hyperparameters for data and training
ENC_MAX_LEN = 256        # max tokens for source (article)
DEC_MAX_LEN = 64         # max tokens for target (headline)
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
PATIENCE = 5             # early stopping patience

# Model hyperparameters (reduced capacity to combat overfitting)
VOCAB_SIZE = tokenizer.vocab_size
D_MODEL = 256            # reduced hidden dimension
NHEAD = 4                # fewer attention heads
NUM_ENCODER_LAYERS = 3   # fewer encoder layers
NUM_DECODER_LAYERS = 3   # fewer decoder layers
DIM_FEEDFORWARD = 512    # reduced feed-forward dimension
DROPOUT = 0.3            # increased dropout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Dataset Definition
# -------------------------------
def encode_example(para: str, headline: str):
    enc_ids = tokenizer.encode(para, add_special_tokens=False, truncation=True, max_length=ENC_MAX_LEN)
    tgt_ids = tokenizer.encode(headline, add_special_tokens=False, truncation=True, max_length=DEC_MAX_LEN-2)
    dec_input_ids = [tokenizer.bos_token_id] + tgt_ids
    dec_output_ids = tgt_ids + [tokenizer.eos_token_id]
    return enc_ids, dec_input_ids, dec_output_ids

class NewsHeadlineDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        para = self.data.loc[idx, "para"]
        headline = self.data.loc[idx, "headline"]
        enc_ids, dec_input_ids, dec_output_ids = encode_example(para, headline)
        return {"enc_ids": enc_ids, "dec_in_ids": dec_input_ids, "dec_out_ids": dec_output_ids}

def collate_fn(batch):
    enc_batch = [torch.tensor(item["enc_ids"], dtype=torch.long) for item in batch]
    dec_in_batch = [torch.tensor(item["dec_in_ids"], dtype=torch.long) for item in batch]
    dec_out_batch = [torch.tensor(item["dec_out_ids"], dtype=torch.long) for item in batch]
    enc_pad = torch.nn.utils.rnn.pad_sequence(enc_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    dec_in_pad = torch.nn.utils.rnn.pad_sequence(dec_in_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    dec_out_pad = torch.nn.utils.rnn.pad_sequence(dec_out_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    enc_pad_mask = (enc_pad == tokenizer.pad_token_id)
    dec_pad_mask = (dec_in_pad == tokenizer.pad_token_id)
    tgt_seq_len = dec_in_pad.size(1)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len)
    return enc_pad, dec_in_pad, dec_out_pad, enc_pad_mask, dec_pad_mask, tgt_mask

# -------------------------------
# Data Loading and Splitting
# -------------------------------
df = pd.read_csv(CLEANED_FILE)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(0.9 * len(df))
train_df = df.iloc[:split_idx]
val_df = df.iloc[split_idx:]

train_dataset = NewsHeadlineDataset(train_df)
val_dataset = NewsHeadlineDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# -------------------------------
# Model, Loss, Optimizer, Scheduler
# -------------------------------
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
)
model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# To save loss logs in CSV
losses_csv = "losses.csv"
losses_data = []

# -------------------------------
# Training Loop with Early Stopping
# -------------------------------
best_val_loss = float("inf")
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for enc_pad, dec_in_pad, dec_out_pad, enc_pad_mask, dec_pad_mask, tgt_mask in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        enc_pad = enc_pad.to(device)
        dec_in_pad = dec_in_pad.to(device)
        dec_out_pad = dec_out_pad.to(device)
        enc_pad_mask = enc_pad_mask.to(device)
        dec_pad_mask = dec_pad_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        
        labels = dec_out_pad.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        optimizer.zero_grad()
        logits = model(
            enc_pad, dec_in_pad,
            src_mask=None, tgt_mask=tgt_mask,
            src_key_padding_mask=enc_pad_mask,
            tgt_key_padding_mask=dec_pad_mask,
            memory_key_padding_mask=enc_pad_mask
        )
        loss = criterion(logits.view(-1, VOCAB_SIZE), labels.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    
    # Validation loop
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for enc_pad, dec_in_pad, dec_out_pad, enc_pad_mask, dec_pad_mask, tgt_mask in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            enc_pad = enc_pad.to(device)
            dec_in_pad = dec_in_pad.to(device)
            dec_out_pad = dec_out_pad.to(device)
            enc_pad_mask = enc_pad_mask.to(device)
            dec_pad_mask = dec_pad_mask.to(device)
            tgt_mask = tgt_mask.to(device)
            
            labels = dec_out_pad.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            
            logits = model(
                enc_pad, dec_in_pad,
                src_mask=None, tgt_mask=tgt_mask,
                src_key_padding_mask=enc_pad_mask,
                tgt_key_padding_mask=dec_pad_mask,
                memory_key_padding_mask=enc_pad_mask
            )
            loss = criterion(logits.view(-1, VOCAB_SIZE), labels.view(-1))
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    
    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    losses_data.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
    
    # Save checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": avg_val_loss,
    }, checkpoint_path)
    
    # Step the scheduler
    scheduler.step(avg_val_loss)
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

# Save losses log to CSV
losses_df = pd.DataFrame(losses_data)
losses_df.to_csv(losses_csv, index=False)
print(f"Losses saved to {losses_csv}")

# Plot train and validation loss curve
plt.figure()
plt.plot(losses_df['epoch'], losses_df['train_loss'], label='Train Loss')
plt.plot(losses_df['epoch'], losses_df['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve")
plt.legend()
plt.savefig("training_validation_loss.png")
plt.show()

# -------------------------------
# Attention Visualization
# -------------------------------
# Get one batch from validation set
model.eval()
sample_batch = next(iter(val_loader))
sample_enc, sample_dec_in, sample_dec_out, sample_enc_mask, sample_dec_mask, sample_tgt_mask = sample_batch
sample_enc = sample_enc.to(device)
sample_enc_mask = sample_enc_mask.to(device)
with torch.no_grad():
    # Get encoder outputs and attention weights
    memory, enc_attn_weights = model.encode(sample_enc, src_key_padding_mask=sample_enc_mask)
    # For decoding, use the first sample from the batch
    sample_dec_in = sample_dec_in.to(device)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(sample_dec_in.size(1)).to(device)
    dec_output, dec_self_attn, dec_cross_attn = model.decode(sample_dec_in, memory, tgt_mask=tgt_mask,
                                                              tgt_key_padding_mask=(sample_dec_in == tokenizer.pad_token_id),
                                                              memory_key_padding_mask=sample_enc_mask)
    # Visualize cross-attention from the first decoder layer for the first sample
    # dec_cross_attn is a list (one per decoder layer); take layer 1
    attn = dec_cross_attn[0]  # shape: (tgt_len, batch, src_len)
    attn = attn[:, 0, :].cpu().numpy()  # select first sample
    plt.figure(figsize=(8, 6))
    plt.imshow(attn, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.xlabel("Source Sequence")
    plt.ylabel("Target Sequence")
    plt.title("Decoder Cross-Attention Weights (Layer 1, Sample 1)")
    plt.savefig("attention_scores.png")
    plt.show()
