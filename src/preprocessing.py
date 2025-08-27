#!/usr/bin/env python3
"""
Advanced NLP preprocessing module for text classification
"""

import re
import string
from typing import List, Dict, Any, Optional
from collections import Counter
import unicodedata

import torch
import numpy as np
from transformers import AutoTokenizer


class AdvancedTextPreprocessor:
    """Advanced text preprocessor with multiple normalization techniques"""
    
    def __init__(self, language: str = "en", lowercase: bool = True, 
                 remove_punctuation: bool = True, remove_accents: bool = True):
        self.language = language
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_accents = remove_accents
        self.special_tokens = {
            '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4
        }
        
    def normalize_text(self, text: str) -> str:
        """Apply comprehensive text normalization"""
        if not isinstance(text, str):
            raise ValueError("Input text must be a string")
            
        # Convert to lowercase if requested
        if self.lowercase:
            text = text.lower()
            
        # Remove accents if requested
        if self.remove_accents:
            text = self._remove_accents(text)
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove punctuation if requested
        if self.remove_punctuation:
            text = self._remove_punctuation(text)
            
        return text
    
    def _remove_accents(self, text: str) -> str:
        """Remove accents from text"""
        try:
            text = unicodedata.normalize('NFD', text)
            text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
            return text
        except Exception:
            # Fallback if unicode normalization fails
            return re.sub(r'[^\x00-\x7F]+', '', text)
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text"""
        # Keep alphanumeric characters and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_text(self, text: str, max_length: int = 512) -> List[str]:
        """Tokenize text into words"""
        normalized_text = self.normalize_text(text)
        tokens = normalized_text.split()
        
        # Truncate if necessary
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            
        return tokens
    
    def build_vocabulary(self, texts: List[str], vocab_size: int = 10000) -> Dict[str, int]:
        """Build vocabulary from a list of texts"""
        # Collect all tokens
        all_tokens = []
        for text in texts:
            tokens = self.tokenize_text(text)
            all_tokens.extend(tokens)
            
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Build vocabulary with most common tokens
        vocab = dict(self.special_tokens)  # Start with special tokens
        most_common = token_counts.most_common(vocab_size - len(self.special_tokens))
        
        # Add tokens to vocabulary
        for i, (token, _) in enumerate(most_common):
            vocab[token] = i + len(self.special_tokens)
            
        return vocab
    
    def encode_text(self, text: str, vocab: Dict[str, int], 
                   max_length: int = 512) -> Dict[str, Any]:
        """Encode text using vocabulary"""
        tokens = self.tokenize_text(text, max_length)
        
        # Convert tokens to IDs
        token_ids = []
        for token in tokens:
            if token in vocab:
                token_ids.append(vocab[token])
            else:
                token_ids.append(vocab['[UNK]'])
                
        # Add special tokens
        if '[CLS]' in vocab:
            token_ids = [vocab['[CLS]']] + token_ids
        if '[SEP]' in vocab:
            token_ids = token_ids + [vocab['[SEP]']]
            
        # Pad or truncate to max_length
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        elif len(token_ids) < max_length:
            padding_length = max_length - len(token_ids)
            if '[PAD]' in vocab:
                token_ids.extend([vocab['[PAD]']] * padding_length)
            else:
                token_ids.extend([0] * padding_length)
                
        # Create attention mask
        attention_mask = [1 if id != 0 else 0 for id in token_ids]
        
        return {
            'input_ids': torch.tensor([token_ids], dtype=torch.long),
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long)
        }


class MultiLanguagePreprocessor:
    """Multi-language text preprocessor with language detection"""
    
    def __init__(self):
        self.language_patterns = {
            'en': r'[a-zA-Z]',
            'id': r'[a-zA-Zàáâãäåæèéêëìíîïòóôõöøùúûüýÿ]',
            'es': r'[a-zA-Záéíóúüñ]',
            'fr': r'[a-zA-Zàâäéèêëïîôöùûüÿç]',
        }
        
    def detect_language(self, text: str) -> str:
        """Detect the language of the text"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 'en'  # Default to English
            
        # Count characters matching each language pattern
        scores = {}
        for lang, pattern in self.language_patterns.items():
            matches = len(re.findall(pattern, text.lower()))
            scores[lang] = matches / len(text) if len(text) > 0 else 0
            
        # Return language with highest score
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'en'
    
    def preprocess_for_language(self, text: str, language: Optional[str] = None) -> str:
        """Preprocess text for a specific language"""
        if language is None:
            language = self.detect_language(text)
            
        preprocessor = AdvancedTextPreprocessor(language=language)
        return preprocessor.normalize_text(text)


class TextAugmentation:
    """Text augmentation techniques for training data"""
    
    def __init__(self):
        pass
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Replace n words with synonyms (simplified implementation)"""
        # This is a simplified version - in practice, you would use a thesaurus
        words = text.split()
        if len(words) <= n:
            return text
            
        # Randomly select words to replace
        indices = np.random.choice(len(words), min(n, len(words)), replace=False)
        
        # Simple synonym replacements (in practice, use a real thesaurus)
        synonyms = {
            'good': 'great',
            'bad': 'terrible',
            'nice': 'wonderful',
            'happy': 'joyful',
            'sad': 'unhappy'
        }
        
        for idx in indices:
            word = words[idx].lower()
            if word in synonyms:
                words[idx] = synonyms[word]
                
        return ' '.join(words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """Insert n random words (simplified implementation)"""
        words = text.split()
        if len(words) == 0:
            return text
            
        # Simple random words to insert
        random_words = ['very', 'really', 'quite', 'extremely', 'highly']
        
        for _ in range(n):
            # Select a random word to insert
            random_word = np.random.choice(random_words)
            # Select a random position to insert
            pos = np.random.randint(0, len(words) + 1)
            words.insert(pos, random_word)
            
        return ' '.join(words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Delete words with probability p"""
        words = text.split()
        if len(words) == 1:
            return text
            
        new_words = []
        for word in words:
            if np.random.random() > p:
                new_words.append(word)
                
        # Ensure at least one word remains
        if len(new_words) == 0:
            return np.random.choice(words)
            
        return ' '.join(new_words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """Swap n pairs of words"""
        words = text.split()
        if len(words) < 2:
            return text
            
        for _ in range(n):
            # Select two random indices
            idx1, idx2 = np.random.choice(len(words), 2, replace=False)
            # Swap words
            words[idx1], words[idx2] = words[idx2], words[idx1]
            
        return ' '.join(words)


def create_preprocessor_pipeline() -> Dict[str, Any]:
    """Create a complete preprocessing pipeline"""
    return {
        'normalizer': AdvancedTextPreprocessor(),
        'language_detector': MultiLanguagePreprocessor(),
        'augmenter': TextAugmentation()
    }


# Example usage
if __name__ == "__main__":
    # Create preprocessor
    preprocessor = AdvancedTextPreprocessor()
    
    # Example text
    text = "This is a sample text with àccénts and punctuation!"
    
    # Normalize text
    normalized = preprocessor.normalize_text(text)
    print(f"Original: {text}")
    print(f"Normalized: {normalized}")
    
    # Tokenize text
    tokens = preprocessor.tokenize_text(normalized)
    print(f"Tokens: {tokens}")
    
    # Build vocabulary
    texts = [
        "This is a positive example",
        "This is a negative example",
        "This is a neutral example"
    ]
    vocab = preprocessor.build_vocabulary(texts, vocab_size=100)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Encode text
    encoded = preprocessor.encode_text(normalized, vocab, max_length=32)
    print(f"Encoded input_ids shape: {encoded['input_ids'].shape}")
    print(f"Encoded attention_mask shape: {encoded['attention_mask'].shape}")