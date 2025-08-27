import math
import warnings
import re
from typing import Optional, Tuple, Union, Dict, Any, List, NamedTuple
from dataclasses import dataclass
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.checkpoint import checkpoint
import numpy as np


class ModelOutput(NamedTuple):
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    last_hidden_state: torch.Tensor = None
    pooler_output: torch.Tensor = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None


class SequenceClassifierOutput(NamedTuple):
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None


@dataclass
class ModelConfig:
    vocab_size: int = 30522
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: str = "gelu"
    max_position_embeddings: int = 512
    num_classes: int = 3
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    use_cache: bool = True
    gradient_checkpointing: bool = False
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    classifier_dropout: Optional[float] = None
    use_native_attention: bool = True


class SimpleTokenizer:
    def __init__(self, vocab_size: int = 30522, max_length: int = 512, 
                 lowercase: bool = True, strip_accents: bool = True):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        
        self.special_tokens = {
            '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4
        }
        self.vocab = dict(self.special_tokens)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_built = False
        
    def _preprocess_text(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()
        if self.strip_accents:
            text = re.sub(r'[àáâãäåæ]', 'a', text)
            text = re.sub(r'[èéêë]', 'e', text)
            text = re.sub(r'[ìíîï]', 'i', text)
            text = re.sub(r'[òóôõöø]', 'o', text)
            text = re.sub(r'[ùúûü]', 'u', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def _tokenize(self, text: str) -> List[str]:
        text = self._preprocess_text(text)
        tokens = text.split()
        return tokens
        
    def build_vocab(self, texts: List[str]) -> None:
        token_counts = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            token_counts.update(tokens)
            
        most_common = token_counts.most_common(self.vocab_size - len(self.special_tokens))
        
        current_id = len(self.special_tokens)
        for token, _ in most_common:
            if token not in self.vocab:
                self.vocab[token] = current_id
                self.inverse_vocab[current_id] = token
                current_id += 1
                
        self.vocab_built = True
        
    def encode(self, text: str, add_special_tokens: bool = True, 
               padding: str = 'max_length', truncation: bool = True) -> Dict[str, torch.Tensor]:
        tokens = self._tokenize(text)
        
        if add_special_tokens:
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            
        if truncation and len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            
        token_ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
        attention_mask = [1] * len(token_ids)
        
        if padding == 'max_length':
            padding_length = self.max_length - len(token_ids)
            token_ids.extend([self.vocab['[PAD]']] * padding_length)
            attention_mask.extend([0] * padding_length)
            
        return {
            'input_ids': torch.tensor([token_ids], dtype=torch.long),
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long)
        }
        
    def decode(self, token_ids: Union[List[int], torch.Tensor], 
               skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        tokens = []
        for token_id in token_ids:
            token = self.inverse_vocab.get(token_id, '[UNK]')
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
            
        return ' '.join(tokens)
        
    def batch_encode(self, texts: List[str], **kwargs) -> Dict[str, torch.Tensor]:
        encoded_batch = [self.encode(text, **kwargs) for text in texts]
        
        input_ids = torch.cat([item['input_ids'] for item in encoded_batch], dim=0)
        attention_mask = torch.cat([item['attention_mask'] for item in encoded_batch], dim=0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
    def save_vocab(self, path: str) -> None:
        torch.save({
            'vocab': self.vocab,
            'config': {
                'vocab_size': self.vocab_size,
                'max_length': self.max_length,
                'lowercase': self.lowercase,
                'strip_accents': self.strip_accents
            }
        }, path)
        
    @classmethod
    def load_vocab(cls, path: str) -> 'SimpleTokenizer':
        data = torch.load(path)
        tokenizer = cls(**data['config'])
        tokenizer.vocab = data['vocab']
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.vocab_built = True
        return tokenizer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class OptimizedMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, batch_first: bool = True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first
        )
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_weights = self.attention(
            query=query,
            key=key, 
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        return attn_output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = query.size(0), query.size(1)
        
        Q = self.w_q(query).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            if mask.dtype == torch.bool:
                attention_scores.masked_fill_(~mask, float('-inf'))
            else:
                attention_scores = attention_scores + mask
            
        if key_padding_mask is not None:
            attention_scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context = torch.matmul(attention_probs, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.w_o(context)
        return output, attention_probs


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "swish":
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            self.activation = F.gelu
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.use_native_attention:
            self.attention = OptimizedMultiHeadAttention(
                d_model=config.d_model,
                nhead=config.nhead,
                dropout=config.attention_probs_dropout_prob,
                batch_first=True
            )
        else:
            self.attention = MultiHeadAttention(
                d_model=config.d_model,
                nhead=config.nhead,
                dropout=config.attention_probs_dropout_prob
            )
            
        self.feed_forward = FeedForward(
            d_model=config.d_model,
            dim_feedforward=config.dim_feedforward,
            dropout=config.hidden_dropout_prob,
            activation=config.activation
        )
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.gradient_checkpointing = config.gradient_checkpointing
        self.use_native_attention = config.use_native_attention
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        def attention_block(hidden_states):
            if self.use_native_attention:
                attn_output, _ = self.attention(
                    query=hidden_states, key=hidden_states, value=hidden_states,
                    attn_mask=mask, key_padding_mask=key_padding_mask
                )
            else:
                attn_output, _ = self.attention(
                    query=hidden_states, key=hidden_states, value=hidden_states, 
                    mask=mask, key_padding_mask=key_padding_mask
                )
            return self.norm1(hidden_states + self.dropout(attn_output))
            
        def feedforward_block(hidden_states):
            ff_output = self.feed_forward(hidden_states)
            return self.norm2(hidden_states + self.dropout(ff_output))
            
        if self.gradient_checkpointing and self.training:
            x = checkpoint(attention_block, x, use_reentrant=False)
            x = checkpoint(feedforward_block, x, use_reentrant=False)
        else:
            x = attention_block(x)
            x = feedforward_block(x)
        
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        
    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
            
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = word_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class TransformerClassifier(nn.Module):
    def __init__(self, config: Union[ModelConfig, Dict[str, Any]]):
        super().__init__()
        
        if isinstance(config, dict):
            config = ModelConfig(**config)
            
        self.config = config
        self.embeddings = EmbeddingLayer(config)
        self.tokenizer = SimpleTokenizer(
            vocab_size=config.vocab_size,
            max_length=config.max_position_embeddings
        )
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(config) 
            for _ in range(config.num_encoder_layers)
        ])
        
        self.pooler = nn.Linear(config.d_model, config.d_model)
        self.pooler_activation = nn.Tanh()
        
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None 
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.d_model, config.num_classes)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def get_attention_mask(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (input_ids != 0).float()
        return attention_mask
        
    def get_extended_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")
            
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(extended_attention_mask.dtype).min
        return extended_attention_mask
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        
        attention_mask = self.get_attention_mask(input_ids, attention_mask)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        
        embedding_output = self.embeddings(input_ids, position_ids)
        
        hidden_states = embedding_output
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for i, layer_module in enumerate(self.encoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            def layer_forward(x):
                return layer_module(x, mask=extended_attention_mask)
                
            if self.config.gradient_checkpointing and self.training:
                hidden_states = checkpoint(layer_forward, hidden_states, use_reentrant=False)
            else:
                hidden_states = layer_module(
                    hidden_states,
                    mask=extended_attention_mask
                )
                
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        sequence_output = hidden_states
        pooled_output = self.pooler_activation(
            self.pooler(sequence_output[:, 0])
        )
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_classes), labels.view(-1))
            
        if not return_dict:
            output = (logits,) + (hidden_states,)
            return ((loss,) + output) if loss is not None else output
            
        return ModelOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )
        
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
        
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        
    def resize_token_embeddings(self, new_num_tokens: int):
        old_embeddings = self.get_input_embeddings()
        old_num_tokens = old_embeddings.weight.size(0)
        
        if old_num_tokens != new_num_tokens:
            warnings.warn(
                f"Resizing token embeddings from {old_num_tokens} to {new_num_tokens}. "
                f"Old weights will be partially copied for the first {min(old_num_tokens, new_num_tokens)} tokens.",
                UserWarning
            )
            
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)
        self.config.vocab_size = new_num_tokens
        return self.get_input_embeddings()
        
    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: int
    ) -> nn.Embedding:
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        
        if old_num_tokens == new_num_tokens:
            return old_embeddings
            
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
        
        self._init_weights(new_embeddings)
        
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
        
        return new_embeddings
        
    @classmethod
    def from_pretrained(cls, model_path: str, config: Optional[Union[ModelConfig, Dict]] = None):
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if config is None:
            config = checkpoint.get('config', {})
            
        if isinstance(config, dict):
            config = ModelConfig(**config)
            
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
        
    def save_pretrained(self, save_path: str, save_config: bool = True):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config
        }, save_path)
        
    def get_num_parameters(self, only_trainable: bool = False) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
            
    def freeze_embeddings(self):
        for param in self.embeddings.parameters():
            param.requires_grad = False
        warnings.warn("Embedding layer has been frozen. Gradients will not be computed for embedding parameters.", UserWarning)
            
    def unfreeze_embeddings(self):
        for param in self.embeddings.parameters():
            param.requires_grad = True
            
    def freeze_encoder(self, num_layers: Optional[int] = None):
        layers_to_freeze = self.encoder_layers if num_layers is None else self.encoder_layers[:num_layers]
        frozen_count = len(layers_to_freeze)
        
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
                
        warnings.warn(
            f"Frozen {frozen_count} encoder layers. Gradients will not be computed for these layers.",
            UserWarning
        )
                
    def unfreeze_encoder(self, num_layers: Optional[int] = None):
        layers_to_unfreeze = self.encoder_layers if num_layers is None else self.encoder_layers[:num_layers]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
                
    def predict(self, text: str, return_probabilities: bool = False) -> Union[str, Dict[str, float]]:
        if not self.tokenizer.vocab_built:
            warnings.warn("Tokenizer vocabulary not built. Call build_vocab() first.", UserWarning)
            return "Error: Tokenizer not ready"
            
        self.eval()
        with torch.no_grad():
            encoded = self.tokenizer.encode(text)
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            probabilities = F.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            
            if return_probabilities:
                return {
                    'predicted_class': predicted_class,
                    'probabilities': probabilities[0].tolist()
                }
            else:
                return str(predicted_class)
                
    def build_tokenizer_vocab(self, texts: List[str]) -> None:
        self.tokenizer.build_vocab(texts)
        warnings.warn(f"Tokenizer vocabulary built with {len(self.tokenizer.vocab)} tokens.", UserWarning)
                
    def gradient_checkpointing_enable(self):
        self.config.gradient_checkpointing = True
        for layer in self.encoder_layers:
            layer.gradient_checkpointing = True
        warnings.warn("Gradient checkpointing enabled. Memory usage will be reduced at the cost of increased computation time.", UserWarning)
        
    def gradient_checkpointing_disable(self):
        self.config.gradient_checkpointing = False
        for layer in self.encoder_layers:
            layer.gradient_checkpointing = False


class BertPooler(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class TransformerForSequenceClassification(TransformerClassifier):
    def __init__(self, config: Union[ModelConfig, Dict[str, Any]], num_labels: int = None):
        if num_labels is not None:
            if isinstance(config, dict):
                config['num_classes'] = num_labels
            else:
                config.num_classes = num_labels
                
        super().__init__(config)
        self.num_labels = self.config.num_classes
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[SequenceClassifierOutput, Tuple]:
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            return_dict=True
        )
        
        if not return_dict:
            return (outputs.loss, outputs.logits) + outputs[2:] if outputs.loss is not None else (outputs.logits,) + outputs[2:]
            
        return SequenceClassifierOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


def create_model(config: Union[ModelConfig, Dict[str, Any]]) -> TransformerClassifier:
    return TransformerClassifier(config)


def create_classification_model(
    vocab_size: int = 30522,
    num_classes: int = 3,
    d_model: int = 512,
    nhead: int = 8,
    num_layers: int = 6,
    **kwargs
) -> TransformerClassifier:
    config = ModelConfig(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        **kwargs
    )
    return TransformerClassifier(config)