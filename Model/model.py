import math
import torch
import torch.nn as nn
from Logger import logger
from types import SimpleNamespace
from transformers import AutoTokenizer


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class DecTransformer(nn.Module):
    def __init__(
        self,
        d_model=768,
        num_heads=4,
        dropout=0.1,
        num_layers=6,
        dim_feedforward=2048,
        max_position_embeddings=512,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Model initialized on device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained("Tokenizer/Custom")
        self.vocab_size = len(self.tokenizer)

        self.bos_token = self.tokenizer.decode(self.tokenizer.bos_token_id)
        self.eos_token_id = self.tokenizer.eos_token_id
        logger.info("Tokenizer loaded successfully.")

        self.max_position_embeddings = max_position_embeddings

        self.token_embedding_layer = nn.Embedding(self.vocab_size, d_model)
        self.pos_encoding_layer = SinusoidalPositionalEncoding(
            d_model, max_position_embeddings
        )
        self.dropout = nn.Dropout(dropout)

        self.decoder = nn.ModuleDict(
            {
                "layers": nn.ModuleList(
                    [
                        DecoderBlock(d_model, num_heads, dim_feedforward, dropout)
                        for _ in range(num_layers)
                    ]
                )
            }
        )

        self.output_projection = nn.Linear(d_model, self.vocab_size, bias=False)

        causal_mask = torch.triu(
            torch.ones(max_position_embeddings, max_position_embeddings), diagonal=1
        ).bool()

        self.register_buffer("causal_mask", causal_mask)

        self.to(self.device)

    def forward(
        self, input_ids, attention_mask=None, labels=None, return_activations=False
    ):
        batch_size, seq_length = input_ids.size()

        if seq_length > self.max_position_embeddings:
            raise ValueError(
                f"Input length {seq_length} exceeds max {self.max_position_embeddings}"
            )

        x = self.token_embedding_layer(input_ids)
        x = self.pos_encoding_layer(x)
        x = self.dropout(x)

        attn_mask = self.causal_mask[:seq_length, :seq_length]
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        activations = [] if return_activations else None

        for idx, layer in enumerate(self.decoder.layers):
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            if return_activations:
                activations.append(x.detach())

        logits = self.output_projection(x)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(
                label_smoothing=0.1, ignore_index=self.tokenizer.pad_token_id
            )
            loss = loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
            return (
                SimpleNamespace(loss=loss, logits=logits, activations=activations)
                if return_activations
                else SimpleNamespace(loss=loss, logits=logits)
            )

        return logits

    def generate(self, input_text, top_k=50, max_length=50, temperature=0.7):
        self.eval()
        prompt = f"{self.bos_token} <|user|> {input_text} <|assistant|> "

        with torch.no_grad():
            tokenized = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_position_embeddings,
                padding=True,
                add_special_tokens=False,
            )
            input_ids = tokenized.input_ids.to(self.device)
            attention_mask = tokenized.attention_mask.to(self.device)
            generated_ids = input_ids.clone()

            for step in range(max_length):
                logits = self.forward(generated_ids, attention_mask)
                next_token_logits = logits[:, -1, :]

                if (
                    torch.isnan(next_token_logits).any()
                    or torch.isinf(next_token_logits).any()
                ):
                    logger.warning(f"Step {step + 1}: Invalid logits")
                    next_token_id = torch.zeros(
                        (input_ids.size(0), 1), dtype=torch.long, device=self.device
                    )
                else:
                    top_k = min(top_k, self.vocab_size)
                    top_k_logits, top_k_indices = torch.topk(
                        next_token_logits, top_k, dim=-1
                    )
                    scaled_logits = top_k_logits / max(temperature, 1e-8)
                    probs = torch.softmax(scaled_logits, dim=-1)

                    if torch.isnan(probs).any() or probs.sum(dim=-1).eq(0).any():
                        logger.warning(
                            f"Step {step + 1}: Invalid probs, fallback to argmax"
                        )
                        next_token_id = torch.argmax(
                            next_token_logits, dim=-1, keepdim=True
                        )
                    else:
                        try:
                            next_token_idx = torch.multinomial(probs, num_samples=1)
                            next_token_id = top_k_indices.gather(-1, next_token_idx)
                        except RuntimeError as e:
                            logger.error(f"Sampling error: {e}")
                            next_token_id = torch.argmax(
                                next_token_logits, dim=-1, keepdim=True
                            )

                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                new_attention = torch.ones(
                    (attention_mask.size(0), 1), device=self.device
                )
                attention_mask = torch.cat([attention_mask, new_attention], dim=-1)

                if (next_token_id == self.eos_token_id).all():
                    logger.info(f"Step {step + 1}: EOS detected")
                    break

                if generated_ids.size(1) >= self.max_position_embeddings:
                    logger.info("Reached maximum positional embeddings limit")
                    break

            output_text = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        torch.cuda.empty_cache()
        return output_text
