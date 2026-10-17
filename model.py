import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding with dropout.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) for broadcasting
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class CustomTransformerEncoderLayer(nn.Module):
    """
    A custom Transformer Encoder layer that returns self-attention weights.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src: (seq_len, batch, d_model)
        # Self-attention block
        attn_output, attn_weights = self.self_attn(src, src, src,
                                                   attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask,
                                                   need_weights=True)
        src2 = self.dropout1(attn_output)
        src = self.norm1(src + src2)
        # Feedforward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src, attn_weights

class CustomTransformerEncoder(nn.Module):
    """
    Stacks multiple CustomTransformerEncoderLayers.
    Returns final output and a list of attention weights from each layer.
    """
    def __init__(self, encoder_layer, num_layers):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        attn_weights_list = []
        for layer in self.layers:
            output, attn_weights = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_weights_list.append(attn_weights)
        return output, attn_weights_list

class CustomTransformerDecoderLayer(nn.Module):
    """
    A custom Transformer Decoder layer that returns both self-attention
    and cross-attention weights.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = F.relu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # tgt: (tgt_len, batch, d_model)
        # Self-attention on target
        self_attn_output, self_attn_weights = self.self_attn(tgt, tgt, tgt,
                                                              attn_mask=tgt_mask,
                                                              key_padding_mask=tgt_key_padding_mask,
                                                              need_weights=True)
        tgt = self.norm1(tgt + self.dropout1(self_attn_output))
        # Cross-attention: query=tgt, key/value=memory
        cross_attn_output, cross_attn_weights = self.cross_attn(tgt, memory, memory,
                                                                attn_mask=memory_mask,
                                                                key_padding_mask=memory_key_padding_mask,
                                                                need_weights=True)
        tgt = self.norm2(tgt + self.dropout2(cross_attn_output))
        # Feedforward network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt, self_attn_weights, cross_attn_weights

class CustomTransformerDecoder(nn.Module):
    """
    Stacks multiple CustomTransformerDecoderLayers.
    Returns final output, list of self-attention weights, and list of cross-attention weights.
    """
    def __init__(self, decoder_layer, num_layers):
        super(CustomTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        self_attn_list = []
        cross_attn_list = []
        for layer in self.layers:
            output, self_attn, cross_attn = layer(output, memory, tgt_mask=tgt_mask,
                                                  memory_mask=memory_mask,
                                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                                  memory_key_padding_mask=memory_key_padding_mask)
            self_attn_list.append(self_attn)
            cross_attn_list.append(cross_attn)
        return output, self_attn_list, cross_attn_list

class TransformerSummarizer(nn.Module):
    """
    Encoder-Decoder Transformer for summarization with attention weight output.
    Regularization via dropout and layer normalization is included.
    """
    def __init__(self, vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 bos_id=1, eos_id=2, pad_id=2):
        super(TransformerSummarizer, self).__init__()
        self.d_model = d_model
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = CustomTransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = CustomTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = CustomTransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Dictionary to store attention weights for visualization.
        self.attention_weights = {}

    def encode(self, src, src_mask=None, src_key_padding_mask=None):
        # src: (batch, src_len)
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        # Transformer expects (src_len, batch, d_model)
        src_emb = src_emb.transpose(0, 1)
        memory, enc_attn_weights = self.encoder(src_emb, mask=src_mask,
                                                src_key_padding_mask=src_key_padding_mask)
        return memory, enc_attn_weights

    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None,
               tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # tgt: (batch, tgt_len)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)
        output, dec_self_attn, dec_cross_attn = self.decoder(tgt_emb, memory,
                                                             tgt_mask=tgt_mask,
                                                             memory_mask=memory_mask,
                                                             tgt_key_padding_mask=tgt_key_padding_mask,
                                                             memory_key_padding_mask=memory_key_padding_mask)
        return output, dec_self_attn, dec_cross_attn

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        # Encode source
        memory, enc_attn_weights = self.encode(src, src_mask, src_key_padding_mask)
        # Decode target
        dec_output, dec_self_attn, dec_cross_attn = self.decode(tgt, memory, tgt_mask,
                                                                memory_mask=None,
                                                                tgt_key_padding_mask=tgt_key_padding_mask,
                                                                memory_key_padding_mask=memory_key_padding_mask)
        # (tgt_len, batch, d_model) -> (batch, tgt_len, d_model)
        dec_output = dec_output.transpose(0, 1)
        logits = self.output_proj(dec_output)
        # Store attention weights for visualization
        self.attention_weights = {
            'encoder': enc_attn_weights,
            'decoder_self': dec_self_attn,
            'decoder_cross': dec_cross_attn
        }
        return logits

    def generate(self, src, tokenizer, max_length=32):
        """
        Greedy decoding for summarization.
        src: (batch, src_len)
        Returns a list of generated summaries (strings).
        """
        self.eval()
        with torch.no_grad():
            memory, _ = self.encode(src, src_key_padding_mask=(src == self.pad_id))
            batch_size = src.size(0)
            # Start each sequence with the BOS token.
            generated = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=src.device)
            for _ in range(max_length):
                tgt_mask = self._generate_square_subsequent_mask(generated.size(1)).to(src.device)
                dec_output, _, _ = self.decode(generated, memory, tgt_mask=tgt_mask,
                                               tgt_key_padding_mask=(generated == self.pad_id),
                                               memory_key_padding_mask=(src == self.pad_id))
                dec_output = dec_output.transpose(0, 1)  # (batch, seq_len, d_model)
                logits = self.output_proj(dec_output[:, -1, :])  # (batch, vocab_size)
                next_tokens = logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_tokens], dim=1)
                if (next_tokens == self.eos_id).all():
                    break
            summaries = tokenizer.batch_decode(generated[:, 1:], skip_special_tokens=True)
            return summaries

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
