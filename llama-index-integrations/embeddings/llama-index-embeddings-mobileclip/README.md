# LlamaIndex Embeddings Integration: MobileCLIP

This package provides a LlamaIndex integration for MobileCLIP embeddings, allowing you to generate image embeddings using MobileCLIP models within the LlamaIndex framework.

## Installation

```bash
pip install llama-index-embeddings-mobileclip
```

## Usage

```python
from llama_index.embeddings.mobileclip import MobileclipEmbedding
from PIL import Image

# Initialize the MobileCLIP embedding model
embed_model = MobileclipEmbedding(
    model_name="MobileCLIP-S2"  # Options: "MobileCLIP-B", "MobileCLIP-S1", "MobileCLIP-S2"
)

# Get embedding for an image
image_path = "path/to/your/image.jpg"
image_embedding = embed_model.get_image_embedding(image_path)

# Get embedding for text
text_embedding = embed_model.get_text_embedding("This is a sample text")
```

## Requirements

- open-clip-torch
- torch
- PIL
