# Testing Framework

This project includes a comprehensive testing framework with advanced features for ensuring code quality and reliability.

## Test Structure

The tests are organized in the `tests/` directory and use `pytest` as the testing framework. All tests follow modern pytest conventions without unittest inheritance.

## Test Features

### 1. Parametrized Tests
Tests use pytest's `@pytest.mark.parametrize` decorator for testing multiple configurations efficiently:

```python
@pytest.mark.parametrize("activation", ["gelu", "relu", "swish"])
def test_feedforward_activations(activation):
    # Test implementation
```

### 2. GPU Testing
GPU tests are automatically skipped if CUDA is not available:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_gpu():
    # GPU-specific test implementation
```

### 3. Property-Based Testing
Using Hypothesis for robustness testing with random inputs:

```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=1, max_value=1000))
def test_model_handles_variable_batch_sizes(batch_size):
    # Property-based test implementation
```

### 4. Fixtures for Test Data
Pytest fixtures provide reusable test data and setup:

```python
@pytest.fixture
def transformer_classifier_config():
    return ModelConfig(vocab_size=1000, d_model=128, ...)
```

### 5. Serialization Stress Testing
Comprehensive tests for model saving/loading with state preservation:

```python
def test_serialization_stress_test():
    # Save model, tokenizer, and optimizer state
    # Load everything back
    # Verify predictions are identical
```

### 6. Performance Benchmarking
CPU and GPU benchmark tests for performance validation:

```python
def test_cpu_benchmark():
    # Benchmark CPU performance
    
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_benchmark():
    # Benchmark GPU performance
```

## Coverage Reporting

The testing framework includes built-in coverage reporting:

```bash
# Generate terminal coverage report
pytest --cov=src tests/

# Generate HTML coverage report
pytest --cov=src --cov-report=html tests/

# Generate XML coverage report for CI/CD
pytest --cov=src --cov-report=xml tests/
```

Configuration is defined in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["src"]
omit = ["*/venv/*", "*/tests/*"]
```

## CI/CD Pipeline

GitHub Actions workflow automatically runs tests on every push:

```yaml
name: Run Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
      - name: Install dependencies
      - name: Run tests with coverage
      - name: Upload coverage to Codecov
```

## Test Categories

### Unit Tests
- Model component tests (MultiHeadAttention, FeedForward, etc.)
- Configuration tests
- Tokenizer tests

### Integration Tests
- End-to-end pipeline tests
- Batch processing tests
- Model saving/loading tests

### Performance Tests
- Forward pass performance
- Memory usage tests
- Benchmark comparisons

### Edge Case Tests
- Empty input handling
- Maximum sequence length tests
- Single head attention tests

### Numerical Stability Tests
- Gradient flow validation
- Softmax stability tests

### Reproducibility Tests
- Weight initialization consistency
- Deterministic forward pass tests

### API Compatibility Tests
- Dictionary vs NamedTuple output tests

### Error Handling Tests
- Invalid input shape tests
- Vocabulary boundary tests

## Coverage Goals

- Minimum: 80% coverage
- Target: 90% coverage
- Ideal: 95%+ coverage

The coverage configuration in `pyproject.toml` enforces a minimum of 80% coverage:

```toml
[tool.pytest.ini_options]
addopts = ["--cov-fail-under=80"]
```

## Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run specific test function
pytest tests/test_model.py::test_model_config_default_values
```

### With Coverage Reporting
```bash
# Terminal coverage report
pytest --cov=src tests/

# HTML coverage report
pytest --cov=src --cov-report=html tests/

# Multiple report formats
pytest --cov=src --cov-report=html --cov-report=xml --cov-report=term-missing tests/
```

### Verbose Output
```bash
# Verbose output
pytest -v

# Verbose with coverage
pytest -v --cov=src tests/
```

## Development Workflow

1. Write tests first for new features
2. Run tests locally before committing
3. Check coverage to ensure adequate test coverage
4. Push to GitHub to trigger CI/CD pipeline
5. Review coverage reports from Codecov

## Test Best Practices

1. Use descriptive test names that clearly indicate what is being tested
2. Follow AAA pattern (Arrange, Act, Assert)
3. Use fixtures for reusable test setup
4. Parametrize tests for multiple similar test cases
5. Mock external dependencies when appropriate
6. Test edge cases and error conditions
7. Maintain test independence - tests should not depend on each other
8. Keep tests fast - avoid unnecessary computations in tests
9. Use appropriate assertions - prefer specific assertions over generic ones
10. Document complex test logic with comments

## Quality Metrics

- Test Coverage: >= 80%
- Code Quality: No pylint errors
- Type Checking: mypy compliance
- Documentation: 100% docstring coverage
- Performance: < 100ms per forward pass on CPU