# MobileClip Embeddings for LlamaIndex

This package provides a LlamaIndex integration for MobileClip, a lightweight CLIP model for generating embeddings.

## Features

- Uses the original MobileClip implementation directly (not using openclip)
- Downloads weights on demand rather than including them in the repository
- Implements both text and image embedding functionality
- Follows the same pattern as the existing CLIP embeddings implementation

## Installation

The package is designed to work with an existing LlamaIndex installation. Make sure you have the following dependencies installed:

- torch
- PIL
- mobileclip

## How to Use

```python
from llama_index.embeddings.mobileclip import MobileClipEmbedding

# Initialize the embedding model
embedding_model = MobileClipEmbedding(
    model_name="mobileclip_s0",
    weights_path="/path/to/weights/mobileclip_s0.pt"  # Optional, will download if not provided
)

# Generate image embeddings
image_embedding = embedding_model.get_image_embedding("/path/to/image.jpg")

# Generate text embeddings
text_embedding = embedding_model.get_text_embedding("This is a test")
```

## Testing

To run the tests:

```bash
# Run all tests
python run_tests.py

# Run only the basic MobileClip inference test
python test_mobileclip.py

# Run only the LlamaIndex integration test
python tests/test_mobileclip.py
```

## Model Weights

The model weights are downloaded automatically when needed. The default model is `mobileclip_s0.pt`, which is a lightweight CLIP model optimized for mobile devices. The weights are stored in the `checkpoints` directory.

To manually download the weights:

```bash
bash get_mobileclip_s0.sh
```
