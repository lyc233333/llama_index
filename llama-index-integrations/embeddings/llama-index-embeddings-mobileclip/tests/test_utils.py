"""Tests for MobileCLIP utility functions."""

import pytest
from unittest.mock import patch, MagicMock

from llama_index.embeddings.mobileclip.utils import preprocess_image


def test_preprocess_image():
    """Test the preprocess_image function."""
    
    assert callable(preprocess_image)
    
    assert preprocess_image.__code__.co_argcount == 2
    
    assert preprocess_image.__doc__ is not None
