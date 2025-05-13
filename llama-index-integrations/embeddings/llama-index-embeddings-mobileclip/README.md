# LlamaIndex Embeddings Integration: MobileCLIP

This package provides a LlamaIndex integration for MobileCLIP embeddings, allowing you to generate image embeddings using MobileCLIP models within the LlamaIndex framework.

## Installation

```bash
pip install llama-index-embeddings-mobileclip
```

## Setup

This integration uses local MobileCLIP weights. You need to download the weights before using the integration:

```bash
# Download the MobileCLIP S0 model weight
./get_mobileclip_s0.sh
```

This will download the MobileCLIP S0 model weight to the `checkpoints` directory.

## Usage

```python
from llama_index.embeddings.mobileclip import MobileclipEmbedding
from PIL import Image

# Initialize the MobileCLIP embedding model
embed_model = MobileclipEmbedding(
    model_name="mobileclip_s0"  # Using the locally downloaded S0 model
)

# Get embedding for an image
image_path = "path/to/your/image.jpg"
image_embedding = embed_model.get_image_embedding(image_path)

# Get embedding for text
text_embedding = embed_model.get_text_embedding("This is a sample text")
```

## Requirements

- torch
- torchvision
- PIL
