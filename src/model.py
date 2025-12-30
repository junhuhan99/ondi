"""
OnDi - Custom Transformer Model from Scratch
Designed for Coding & English
License: 100% Owned by Creator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-Head Self Attention mechanism"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(attn_output)


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with GELU activation"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single Transformer decoder block"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x


class OnDiModel(nn.Module):
    """
    OnDi - Custom GPT-style Language Model
    Optimized for Coding & English

    Architecture:
    - Decoder-only Transformer
    - Pre-norm (LayerNorm before attention/FFN)
    - GELU activation
    - Weight tying (embedding & output)
    """

    def __init__(
        self,
        vocab_size=32000,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
        max_seq_len=1024,
        dropout=0.1,
        pad_token_id=0
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)
        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id
            )

        return {'loss': loss, 'logits': logits}

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.9):
        """Generate text autoregressively"""
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]

            outputs = self(idx_cond)
            logits = outputs['logits'][:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def get_model_config(size='small'):
    """Get model configuration by size"""
    configs = {
        'tiny': {
            'vocab_size': 32000,
            'd_model': 256,
            'n_heads': 4,
            'n_layers': 4,
            'd_ff': 1024,
            'max_seq_len': 512,
            'dropout': 0.1
        },
        'small': {
            'vocab_size': 32000,
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 8,
            'd_ff': 2048,
            'max_seq_len': 1024,
            'dropout': 0.1
        },
        'medium': {
            'vocab_size': 32000,
            'd_model': 768,
            'n_heads': 12,
            'n_layers': 12,
            'd_ff': 3072,
            'max_seq_len': 1024,
            'dropout': 0.1
        }
    }
    return configs.get(size, configs['small'])


if __name__ == '__main__':
    print("Testing OnDi Model...")

    config = get_model_config('small')
    model = OnDiModel(**config)

    print(f"\nModel Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print(f"\nTotal Parameters: {model.n_params:,} ({model.n_params/1e6:.1f}M)")

    x = torch.randint(0, config['vocab_size'], (2, 128))
    output = model(x, labels=x)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output logits shape: {output['logits'].shape}")
    print(f"  Loss: {output['loss'].item():.4f}")

    print("\nModel test successful!")
