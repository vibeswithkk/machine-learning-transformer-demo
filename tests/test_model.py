import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
from pathlib import Path
from hypothesis import given, strategies as st
import time

from src.model import (
    ModelConfig, 
    TransformerClassifier, 
    SimpleTokenizer, 
    MultiHeadAttention, 
    FeedForward,
    TransformerEncoderBlock,
    PositionalEncoding,
    OptimizedMultiHeadAttention
)


# Parametrized test for ModelConfig initialization
@pytest.mark.parametrize("vocab_size,d_model,nhead,num_encoder_layers,dim_feedforward,dropout,activation,max_position_embeddings,num_classes", [
    (30522, 512, 8, 6, 2048, 0.1, "gelu", 512, 3),
    (1000, 128, 4, 3, 512, 0.2, "relu", 256, 5),
    (5000, 256, 8, 4, 1024, 0.0, "swish", 128, 2),
])
def test_model_config_initialization(vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation, max_position_embeddings, num_classes):
    config = ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        max_position_embeddings=max_position_embeddings,
        num_classes=num_classes
    )
    assert config.vocab_size == vocab_size
    assert config.d_model == d_model
    assert config.nhead == nhead
    assert config.num_encoder_layers == num_encoder_layers
    assert config.dim_feedforward == dim_feedforward
    assert config.dropout == dropout
    assert config.activation == activation
    assert config.max_position_embeddings == max_position_embeddings
    assert config.num_classes == num_classes


def test_model_config_default_values():
    config = ModelConfig()
    assert config.vocab_size == 30522
    assert config.d_model == 512
    assert config.nhead == 8
    assert config.num_encoder_layers == 6
    assert config.dim_feedforward == 2048
    assert config.dropout == 0.1
    assert config.activation == "gelu"
    assert config.max_position_embeddings == 512
    assert config.num_classes == 3
    assert config.layer_norm_eps == 1e-12
    assert config.initializer_range == 0.02


def test_model_config_immutability():
    config = ModelConfig()
    with pytest.raises(AttributeError):
        config.vocab_size = 5000


# SimpleTokenizer tests
@pytest.fixture
def simple_tokenizer():
    return SimpleTokenizer(vocab_size=100, max_length=32)


def test_tokenizer_initialization(simple_tokenizer):
    assert simple_tokenizer.vocab_size == 100
    assert simple_tokenizer.max_length == 32
    assert len(simple_tokenizer.special_tokens) == 5
    assert '[PAD]' in simple_tokenizer.special_tokens
    assert '[UNK]' in simple_tokenizer.special_tokens
    assert '[CLS]' in simple_tokenizer.special_tokens
    assert '[SEP]' in simple_tokenizer.special_tokens
    assert '[MASK]' in simple_tokenizer.special_tokens


def test_preprocess_text(simple_tokenizer):
    text = "Hello, World! àáâãäå"
    processed = simple_tokenizer._preprocess_text(text)
    assert processed == "hello world aaaaaa"


def test_tokenize(simple_tokenizer):
    text = "This is a test sentence."
    tokens = simple_tokenizer._tokenize(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_build_vocab(simple_tokenizer):
    texts = [
        "This is the first sentence.",
        "This is the second sentence.",
        "Another different sentence."
    ]
    simple_tokenizer.build_vocab(texts)
    assert simple_tokenizer.vocab_built
    assert len(simple_tokenizer.vocab) > 5  # At least special tokens


def test_encode_decode(simple_tokenizer):
    texts = ["This is a test.", "Another test sentence."]
    simple_tokenizer.build_vocab(texts)
    
    # Test encoding
    encoded = simple_tokenizer.encode("This is a test.")
    assert 'input_ids' in encoded
    assert 'attention_mask' in encoded
    assert encoded['input_ids'].shape[1] == simple_tokenizer.max_length
    
    # Test decoding
    decoded = simple_tokenizer.decode(encoded['input_ids'][0])
    assert isinstance(decoded, str)


def test_batch_encode(simple_tokenizer):
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    simple_tokenizer.build_vocab(texts)
    
    batch_encoded = simple_tokenizer.batch_encode(texts)
    assert 'input_ids' in batch_encoded
    assert 'attention_mask' in batch_encoded
    assert batch_encoded['input_ids'].shape[0] == len(texts)
    assert batch_encoded['input_ids'].shape[1] == simple_tokenizer.max_length


def test_save_load_vocab(simple_tokenizer):
    texts = ["Test sentence one.", "Test sentence two."]
    simple_tokenizer.build_vocab(texts)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = os.path.join(tmpdir, "vocab.pt")
        simple_tokenizer.save_vocab(vocab_path)
        assert os.path.exists(vocab_path)
        
        loaded_tokenizer = SimpleTokenizer.load_vocab(vocab_path)
        assert loaded_tokenizer.vocab_size == simple_tokenizer.vocab_size
        assert len(loaded_tokenizer.vocab) == len(simple_tokenizer.vocab)


# PositionalEncoding tests
def test_positional_encoding_forward():
    d_model = 512
    max_len = 100
    pe = PositionalEncoding(d_model, max_len=max_len)
    
    batch_size = 4
    seq_len = 50
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = pe(x)
    assert output.shape == (batch_size, seq_len, d_model)


def test_positional_encoding_dropout():
    d_model = 512
    pe = PositionalEncoding(d_model, dropout=0.1)
    
    batch_size = 2
    seq_len = 30
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = pe(x)
    assert output.shape == (batch_size, seq_len, d_model)


# MultiHeadAttention tests
@pytest.fixture
def multi_head_attention():
    d_model = 512
    nhead = 8
    return MultiHeadAttention(d_model, nhead)


def test_attention_initialization(multi_head_attention):
    assert multi_head_attention.d_model == 512
    assert multi_head_attention.nhead == 8
    assert multi_head_attention.d_k == 512 // 8


def test_attention_forward(multi_head_attention):
    batch_size = 4
    seq_len = 32
    
    query = torch.randn(batch_size, seq_len, 512)
    key = torch.randn(batch_size, seq_len, 512)
    value = torch.randn(batch_size, seq_len, 512)
    
    output, weights = multi_head_attention(query, key, value)
    assert output.shape == (batch_size, seq_len, 512)
    assert weights.shape == (batch_size, 8, seq_len, seq_len)


def test_attention_with_mask(multi_head_attention):
    batch_size = 2
    seq_len = 16
    
    query = torch.randn(batch_size, seq_len, 512)
    key = torch.randn(batch_size, seq_len, 512)
    value = torch.randn(batch_size, seq_len, 512)
    
    # Test with boolean mask
    mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)
    output, weights = multi_head_attention(query, key, value, mask=mask)
    assert output.shape == (batch_size, seq_len, 512)
    
    # Test with additive mask
    mask = torch.zeros(batch_size, seq_len, seq_len)
    output, weights = multi_head_attention(query, key, value, mask=mask)
    assert output.shape == (batch_size, seq_len, 512)


# OptimizedMultiHeadAttention tests
@pytest.fixture
def optimized_multi_head_attention():
    d_model = 512
    nhead = 8
    return OptimizedMultiHeadAttention(d_model, nhead)


def test_optimized_attention_initialization(optimized_multi_head_attention):
    assert optimized_multi_head_attention.d_model == 512
    assert optimized_multi_head_attention.nhead == 8
    assert isinstance(optimized_multi_head_attention.attention, nn.MultiheadAttention)


def test_optimized_attention_forward(optimized_multi_head_attention):
    batch_size = 4
    seq_len = 32
    
    query = torch.randn(batch_size, seq_len, 512)
    key = torch.randn(batch_size, seq_len, 512)
    value = torch.randn(batch_size, seq_len, 512)
    
    output, weights = optimized_multi_head_attention(query, key, value)
    assert output.shape == (batch_size, seq_len, 512)
    assert weights is not None


# Parametrized FeedForward tests
@pytest.mark.parametrize("activation", ["gelu", "relu", "swish"])
def test_feedforward_activations(activation):
    d_model = 128
    dim_feedforward = 512
    ff = FeedForward(d_model, dim_feedforward, activation=activation)
    x = torch.randn(2, 16, d_model)
    output = ff(x)
    assert output.shape == (2, 16, d_model)


def test_feedforward_initialization():
    d_model = 512
    dim_feedforward = 2048
    ff = FeedForward(d_model, dim_feedforward)
    
    assert isinstance(ff.linear1, nn.Linear)
    assert isinstance(ff.linear2, nn.Linear)
    assert isinstance(ff.dropout, nn.Dropout)


def test_feedforward_forward():
    d_model = 512
    dim_feedforward = 2048
    ff = FeedForward(d_model, dim_feedforward)
    
    batch_size = 4
    seq_len = 32
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = ff(x)
    assert output.shape == (batch_size, seq_len, d_model)


# TransformerEncoderBlock tests
@pytest.fixture
def transformer_encoder_config():
    return ModelConfig(
        d_model=128,
        nhead=4,
        dim_feedforward=512,
        num_encoder_layers=3
    )


@pytest.fixture
def transformer_encoder_block(transformer_encoder_config):
    return TransformerEncoderBlock(transformer_encoder_config)


def test_encoder_block_initialization(transformer_encoder_block):
    assert isinstance(transformer_encoder_block.attention, (MultiHeadAttention, OptimizedMultiHeadAttention))
    assert isinstance(transformer_encoder_block.feed_forward, FeedForward)
    assert isinstance(transformer_encoder_block.norm1, nn.LayerNorm)
    assert isinstance(transformer_encoder_block.norm2, nn.LayerNorm)
    assert isinstance(transformer_encoder_block.dropout, nn.Dropout)


def test_encoder_block_forward(transformer_encoder_block):
    batch_size = 4
    seq_len = 16
    
    x = torch.randn(batch_size, seq_len, 128)
    output = transformer_encoder_block(x)
    
    assert output.shape == (batch_size, seq_len, 128)


def test_encoder_block_with_mask(transformer_encoder_block):
    batch_size = 2
    seq_len = 8
    
    x = torch.randn(batch_size, seq_len, 128)
    mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)
    
    output = transformer_encoder_block(x, mask=mask)
    assert output.shape == (batch_size, seq_len, 128)


# TransformerClassifier tests
@pytest.fixture
def transformer_classifier_config():
    return ModelConfig(
        vocab_size=1000,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        dim_feedforward=512,
        max_position_embeddings=64,
        num_classes=3
    )


@pytest.fixture
def transformer_classifier(transformer_classifier_config):
    return TransformerClassifier(transformer_classifier_config)


def test_model_initialization(transformer_classifier, transformer_classifier_config):
    assert transformer_classifier.config.vocab_size == 1000
    assert transformer_classifier.config.d_model == 128
    assert transformer_classifier.config.nhead == 4
    assert transformer_classifier.config.num_encoder_layers == 3
    assert transformer_classifier.config.num_classes == 3
    assert isinstance(transformer_classifier.embeddings, nn.Module)
    assert isinstance(transformer_classifier.encoder_layers, nn.ModuleList)
    assert len(transformer_classifier.encoder_layers) == 3


def test_model_forward(transformer_classifier):
    batch_size = 4
    seq_len = 32
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    outputs = transformer_classifier(input_ids, attention_mask=attention_mask)
    assert 'logits' in outputs
    assert 'last_hidden_state' in outputs
    assert 'pooler_output' in outputs
    assert outputs['logits'].shape == (batch_size, 3)
    assert outputs['last_hidden_state'].shape == (batch_size, seq_len, 128)
    assert outputs['pooler_output'].shape == (batch_size, 128)


def test_model_forward_with_labels(transformer_classifier):
    batch_size = 4
    seq_len = 32
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 3, (batch_size,))
    
    outputs = transformer_classifier(input_ids, attention_mask=attention_mask, labels=labels)
    assert 'loss' in outputs
    assert 'logits' in outputs
    assert outputs['loss'] is not None
    assert outputs['logits'].shape == (batch_size, 3)


def test_model_resize_token_embeddings(transformer_classifier):
    original_vocab_size = transformer_classifier.config.vocab_size
    new_vocab_size = 1500
    
    old_embeddings = transformer_classifier.get_input_embeddings()
    assert old_embeddings.weight.shape[0] == original_vocab_size
    
    transformer_classifier.resize_token_embeddings(new_vocab_size)
    new_embeddings = transformer_classifier.get_input_embeddings()
    assert new_embeddings.weight.shape[0] == new_vocab_size


def test_model_get_num_parameters(transformer_classifier):
    num_params = transformer_classifier.get_num_parameters()
    assert isinstance(num_params, int)
    assert num_params > 0
    
    num_trainable_params = transformer_classifier.get_num_parameters(only_trainable=True)
    assert isinstance(num_trainable_params, int)
    assert num_trainable_params > 0
    assert num_trainable_params <= num_params


def test_model_freeze_unfreeze(transformer_classifier):
    # Test freezing embeddings
    transformer_classifier.freeze_embeddings()
    for param in transformer_classifier.embeddings.parameters():
        assert not param.requires_grad
        
    # Test unfreezing embeddings
    transformer_classifier.unfreeze_embeddings()
    for param in transformer_classifier.embeddings.parameters():
        assert param.requires_grad
        
    # Test freezing encoder layers
    transformer_classifier.freeze_encoder()
    for layer in transformer_classifier.encoder_layers:
        for param in layer.parameters():
            assert not param.requires_grad
            
    # Test unfreezing encoder layers
    transformer_classifier.unfreeze_encoder()
    for layer in transformer_classifier.encoder_layers:
        for param in layer.parameters():
            assert param.requires_grad


def test_model_save_load_pretrained(transformer_classifier):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.pt")
        
        # Save model
        transformer_classifier.save_pretrained(model_path)
        assert os.path.exists(model_path)
        
        # Load model
        loaded_model = TransformerClassifier.from_pretrained(model_path)
        assert isinstance(loaded_model, TransformerClassifier)
        assert loaded_model.config.vocab_size == transformer_classifier.config.vocab_size
        assert loaded_model.config.d_model == transformer_classifier.config.d_model


# Integration tests
@pytest.fixture
def integration_test_config():
    return ModelConfig(
        vocab_size=500,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=128,
        max_position_embeddings=32,
        num_classes=3
    )


@pytest.fixture
def integration_test_model(integration_test_config):
    return TransformerClassifier(integration_test_config)


def test_end_to_end_pipeline(integration_test_model):
    # Build vocabulary
    texts = [
        "This is a positive example.",
        "This is a negative example.",
        "This is a neutral example."
    ]
    integration_test_model.build_tokenizer_vocab(texts)
    
    # Encode text
    encoded = integration_test_model.tokenizer.encode("This is a test example.")
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    
    # Forward pass
    outputs = integration_test_model(input_ids, attention_mask=attention_mask)
    assert 'logits' in outputs
    assert outputs['logits'].shape == (1, 3)
    
    # Prediction
    probabilities = torch.softmax(outputs['logits'], dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1)
    assert predicted_class.shape == (1,)


def test_batch_processing(integration_test_model):
    # Build vocabulary
    texts = [
        "First example sentence.",
        "Second example sentence.",
        "Third example sentence."
    ]
    integration_test_model.build_tokenizer_vocab(texts)
    
    # Batch encode
    batch_texts = [
        "First example sentence.",
        "Second example sentence."
    ]
    batch_encoded = integration_test_model.tokenizer.batch_encode(batch_texts)
    input_ids = batch_encoded['input_ids']
    attention_mask = batch_encoded['attention_mask']
    
    # Batch forward pass
    outputs = integration_test_model(input_ids, attention_mask=attention_mask)
    assert 'logits' in outputs
    assert outputs['logits'].shape == (2, 3)


# Performance tests
@pytest.fixture
def performance_test_config():
    return ModelConfig(
        vocab_size=1000,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        dim_feedforward=512,
        max_position_embeddings=64,
        num_classes=3
    )


@pytest.fixture
def performance_test_model(performance_test_config):
    return TransformerClassifier(performance_test_config)


def test_forward_pass_performance(performance_test_model):
    batch_size = 8
    seq_len = 32
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Warm up
    for _ in range(5):
        _ = performance_test_model(input_ids, attention_mask=attention_mask)
        
    # Measure performance
    start_time = time.time()
    num_iterations = 10
    
    for _ in range(num_iterations):
        _ = performance_test_model(input_ids, attention_mask=attention_mask)
        
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    
    # Assert reasonable performance (less than 100ms per forward pass on CPU)
    assert avg_time < 0.1


def test_memory_usage(performance_test_model):
    batch_size = 4
    seq_len = 32
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Check that model can handle the input without memory issues
    outputs = performance_test_model(input_ids, attention_mask=attention_mask)
    assert 'logits' in outputs
    assert outputs['logits'].shape == (batch_size, 3)


# Edge case tests
def test_empty_input():
    config = ModelConfig(
        vocab_size=100,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=64,
        max_position_embeddings=16,
        num_classes=2
    )
    model = TransformerClassifier(config)
    
    # Test with minimal input
    input_ids = torch.tensor([[0]])  # Single token
    attention_mask = torch.tensor([[1]])
    
    outputs = model(input_ids, attention_mask=attention_mask)
    assert 'logits' in outputs
    assert outputs['logits'].shape == (1, 2)


def test_max_sequence_length():
    max_len = 128
    config = ModelConfig(
        vocab_size=100,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=64,
        max_position_embeddings=max_len,
        num_classes=2
    )
    model = TransformerClassifier(config)
    
    # Test with maximum sequence length
    input_ids = torch.randint(0, config.vocab_size, (2, max_len))
    attention_mask = torch.ones(2, max_len)
    
    outputs = model(input_ids, attention_mask=attention_mask)
    assert 'logits' in outputs
    assert outputs['logits'].shape == (2, 2)


def test_single_head_attention():
    config = ModelConfig(
        d_model=32,
        nhead=1,  # Single head
        num_encoder_layers=1,
        dim_feedforward=64,
        num_classes=2
    )
    model = TransformerClassifier(config)
    
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    attention_mask = torch.ones(2, 16)
    
    outputs = model(input_ids, attention_mask=attention_mask)
    assert 'logits' in outputs
    assert outputs['logits'].shape == (2, 2)


# Numerical stability tests
def test_gradient_flow():
    config = ModelConfig(
        vocab_size=100,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=128,
        max_position_embeddings=32,
        num_classes=3
    )
    model = TransformerClassifier(config)
    
    # Create input
    input_ids = torch.randint(0, config.vocab_size, (4, 16))
    attention_mask = torch.ones(4, 16)
    labels = torch.randint(0, config.num_classes, (4,))
    
    # Forward pass
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs['loss']
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist and are not NaN
    for param in model.parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()


def test_softmax_stability():
    # Test that softmax operations are numerically stable
    large_logits = torch.tensor([[1000.0, 2000.0, 3000.0]])
    softmax_output = torch.softmax(large_logits, dim=-1)
    
    # Should not produce NaN or inf
    assert not torch.isnan(softmax_output).any()
    assert not torch.isinf(softmax_output).any()
    
    # Should sum to 1
    assert abs(softmax_output.sum().item() - 1.0) < 1e-5


# Reproducibility tests
def test_weight_initialization():
    config = ModelConfig(
        vocab_size=100,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=64,
        max_position_embeddings=16,
        num_classes=2,
        seed=42
    )
    
    # Create two models with same config
    model1 = TransformerClassifier(config)
    model2 = TransformerClassifier(config)
    
    # Check that initial weights are the same
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2
        assert torch.allclose(param1, param2, atol=1e-7), f"Parameter {name1} differs between models"


def test_deterministic_forward():
    config = ModelConfig(
        vocab_size=100,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=64,
        max_position_embeddings=16,
        num_classes=2
    )
    
    # Create two models
    model1 = TransformerClassifier(config)
    model2 = TransformerClassifier(config)
    
    # Same input
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    attention_mask = torch.ones(2, 8)
    
    # Forward passes
    with torch.no_grad():
        outputs1 = model1(input_ids, attention_mask=attention_mask)
        outputs2 = model2(input_ids, attention_mask=attention_mask)
        
    # Should be identical
    assert torch.allclose(outputs1['logits'], outputs2['logits'], atol=1e-6)


# API compatibility tests
def test_dict_vs_namedtuple_output():
    config = ModelConfig(
        vocab_size=100,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=64,
        max_position_embeddings=16,
        num_classes=2
    )
    model = TransformerClassifier(config)
    
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    attention_mask = torch.ones(2, 8)
    
    # Test dict output
    dict_outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
    assert isinstance(dict_outputs, dict)
    assert 'logits' in dict_outputs


# Error handling tests
def test_invalid_input_shapes():
    config = ModelConfig(
        vocab_size=100,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=64,
        max_position_embeddings=16,
        num_classes=2
    )
    model = TransformerClassifier(config)
    
    # Test with wrong input shape - should raise an exception
    with pytest.raises(Exception):
        wrong_input = torch.randint(0, config.vocab_size, (2, 8, 5))  # Wrong dimensions
        _ = model(wrong_input)


def test_invalid_vocab_indices():
    config = ModelConfig(
        vocab_size=100,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=64,
        max_position_embeddings=16,
        num_classes=2
    )
    model = TransformerClassifier(config)
    
    # Test with out-of-bounds indices
    input_ids = torch.tensor([[-1, 1000]])  # Invalid indices
    attention_mask = torch.ones(1, 2)
    
    # Should handle gracefully (embedding layers typically handle this)
    try:
        outputs = model(input_ids, attention_mask=attention_mask)
        assert 'logits' in outputs
    except Exception as e:
        # If it raises an exception, it should be informative
        assert isinstance(e, (IndexError, RuntimeError))


# GPU tests
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_gpu():
    config = ModelConfig(
        vocab_size=100,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=64,
        max_position_embeddings=16,
        num_classes=2
    )
    model = TransformerClassifier(config)
    model.cuda()
    
    input_ids = torch.randint(0, config.vocab_size, (2, 16)).cuda()
    attention_mask = torch.ones(2, 16).cuda()
    
    outputs = model(input_ids, attention_mask=attention_mask)
    assert 'logits' in outputs
    assert outputs['logits'].device.type == 'cuda'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_batch_processing_gpu():
    config = ModelConfig(
        vocab_size=100,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=64,
        max_position_embeddings=16,
        num_classes=2
    )
    model = TransformerClassifier(config)
    model.cuda()
    
    input_ids = torch.randint(0, config.vocab_size, (4, 16)).cuda()
    attention_mask = torch.ones(4, 16).cuda()
    
    outputs = model(input_ids, attention_mask=attention_mask)
    assert 'logits' in outputs
    assert outputs['logits'].device.type == 'cuda'
    assert outputs['logits'].shape == (4, 2)


# Serialization stress test
def test_serialization_stress_test():
    config = ModelConfig(
        vocab_size=100,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=64,
        max_position_embeddings=16,
        num_classes=2
    )
    model = TransformerClassifier(config)
    
    # Build vocabulary
    texts = [
        "This is a positive example.",
        "This is a negative example.",
        "This is a neutral example."
    ]
    model.build_tokenizer_vocab(texts)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "stress_test_model.pt")
        
        # Save model, tokenizer, and optimizer state
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.__dict__,
            'tokenizer_vocab': model.tokenizer.vocab,
            'tokenizer_config': {
                'vocab_size': model.tokenizer.vocab_size,
                'max_length': model.tokenizer.max_length
            }
        }, model_path)
        
        assert os.path.exists(model_path)
        
        # Load everything back
        checkpoint = torch.load(model_path)
        loaded_config = ModelConfig(**checkpoint['config'])
        loaded_model = TransformerClassifier(loaded_config)
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore tokenizer
        loaded_model.tokenizer.vocab = checkpoint['tokenizer_vocab']
        loaded_model.tokenizer.vocab_built = True
        loaded_model.tokenizer.inverse_vocab = {v: k for k, v in loaded_model.tokenizer.vocab.items()}
        
        # Restore optimizer
        loaded_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=5e-5)
        loaded_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Verify predictions are identical
        test_text = "This is a test example."
        original_result = model(test_text)
        loaded_result = loaded_model(test_text)
        
        assert torch.allclose(original_result['logits'], loaded_result['logits'], atol=1e-6)


# Benchmark extensions
def test_cpu_benchmark():
    config = ModelConfig(
        vocab_size=1000,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        max_position_embeddings=32,
        num_classes=3
    )
    model = TransformerClassifier(config)
    
    batch_size = 16
    seq_len = 32
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Warm up
    for _ in range(3):
        _ = model(input_ids, attention_mask=attention_mask)
    
    # Benchmark
    start_time = time.time()
    num_iterations = 20
    
    for _ in range(num_iterations):
        _ = model(input_ids, attention_mask=attention_mask)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    throughput = batch_size / avg_time
    
    # Log benchmark results (in a real scenario, you might use a proper logger)
    print(f"CPU Benchmark: {avg_time*1000:.2f}ms per forward pass, {throughput:.2f} samples/sec")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_benchmark():
    config = ModelConfig(
        vocab_size=1000,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        max_position_embeddings=32,
        num_classes=3
    )
    model = TransformerClassifier(config)
    model.cuda()
    
    batch_size = 32
    seq_len = 32
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
    attention_mask = torch.ones(batch_size, seq_len).cuda()
    
    # Warm up
    for _ in range(3):
        _ = model(input_ids, attention_mask=attention_mask)
    
    # Benchmark
    start_time = time.time()
    num_iterations = 50
    
    for _ in range(num_iterations):
        _ = model(input_ids, attention_mask=attention_mask)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    throughput = batch_size / avg_time
    
    # Log benchmark results
    print(f"GPU Benchmark: {avg_time*1000:.2f}ms per forward pass, {throughput:.2f} samples/sec")


# Property-based testing with Hypothesis
@given(st.integers(min_value=1, max_value=1000))
def test_model_handles_variable_batch_sizes(batch_size):
    config = ModelConfig(
        vocab_size=100,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=64,
        max_position_embeddings=16,
        num_classes=2
    )
    model = TransformerClassifier(config)
    
    # Ensure batch_size is reasonable for testing
    batch_size = min(batch_size, 10)
    seq_len = 8
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    outputs = model(input_ids, attention_mask=attention_mask)
    assert 'logits' in outputs
    assert outputs['logits'].shape == (batch_size, 2)


@given(st.integers(min_value=1, max_value=128))
def test_model_handles_variable_sequence_lengths(seq_len):
    config = ModelConfig(
        vocab_size=100,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=64,
        max_position_embeddings=128,  # Ensure max_position_embeddings is sufficient
        num_classes=2
    )
    model = TransformerClassifier(config)
    
    # Ensure seq_len is reasonable for testing
    seq_len = min(seq_len, 16)
    batch_size = 4
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    outputs = model(input_ids, attention_mask=attention_mask)
    assert 'logits' in outputs
    assert outputs['logits'].shape == (batch_size, 2)


@given(st.floats(min_value=0.0, max_value=1.0))
def test_dropout_variations(dropout_rate):
    config = ModelConfig(
        vocab_size=100,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=64,
        dropout=dropout_rate,
        max_position_embeddings=16,
        num_classes=2
    )
    model = TransformerClassifier(config)
    
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    attention_mask = torch.ones(2, 8)
    
    # Should not crash with any dropout rate
    outputs = model(input_ids, attention_mask=attention_mask)
    assert 'logits' in outputs
    assert outputs['logits'].shape == (2, 2)