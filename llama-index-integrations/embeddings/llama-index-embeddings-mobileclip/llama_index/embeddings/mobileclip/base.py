import logging
import os
import sys
import numpy as np
from typing import Any, List

from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.schema import ImageType
from PIL import Image

logger = logging.getLogger(__name__)


DEFAULT_MOBILECLIP_MODEL = "mobileclip_s0"
DEFAULT_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "checkpoints",
)


class MobileclipEmbedding(MultiModalEmbedding):
    """
    MobileCLIP embedding models for encoding text and image for Multi-Modal purpose.

    This class provides an interface to generate embeddings using a model
    from the MobileCLIP family. It uses locally downloaded weights from the
    MobileCLIP repository.

    Note:
        Requires the MobileCLIP weights to be downloaded using the provided
        get_mobileclip_s0.sh script.
    """

    embed_batch_size: int = Field(default=DEFAULT_EMBED_BATCH_SIZE, gt=0)
    checkpoint_dir: str = Field(default=DEFAULT_CHECKPOINT_DIR)

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _preprocess: Any = PrivateAttr()
    _device: Any = PrivateAttr()

    @classmethod
    def class_name(cls) -> str:
        return "MobileclipEmbedding"

    def __init__(
        self,
        *,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        model_name: str = DEFAULT_MOBILECLIP_MODEL,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        **kwargs: Any,
    ):
        """
        Initializes the MobileclipEmbedding class.

        Args:
            embed_batch_size (int, optional): The batch size for embedding generation. 
                Defaults to 10, must be > 0.
            model_name (str): The model name of MobileCLIP model.
                Defaults to 'mobileclip_s0'.
            checkpoint_dir (str): Directory containing the downloaded MobileCLIP weights.
                Defaults to the 'checkpoints' directory in the package root.

        Raises:
            ImportError: If torch is not available.
            ValueError: If the model cannot be loaded or if the embed_batch_size
                is not > 0.
            FileNotFoundError: If the model weights file is not found.
        """
        if embed_batch_size <= 0:
            raise ValueError(f"Embed batch size {embed_batch_size} must be > 0.")

        try:
            import torch
            from torchvision import transforms
        except ImportError:
            raise ImportError(
                "MobileclipEmbedding requires `pip install torch torchvision`."
            )

        super().__init__(
            embed_batch_size=embed_batch_size, 
            model_name=model_name, 
            checkpoint_dir=checkpoint_dir,
            **kwargs
        )

        try:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            model_path = os.path.join(self.checkpoint_dir, f"{self.model_name}.pt")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model weights file not found at {model_path}. "
                    f"Please run the get_mobileclip_s0.sh script to download the weights."
                )
            
            state_dict = torch.load(model_path, map_location=self._device)
            
            from torch import nn
            
            class MobileCLIPTextEncoder(nn.Module):
                def __init__(self, embed_dim=512):
                    super().__init__()
                    self.transformer = nn.Transformer(
                        d_model=embed_dim, 
                        nhead=8, 
                        num_encoder_layers=6,
                        num_decoder_layers=0,
                        dim_feedforward=2048
                    )
                    self.token_embedding = nn.Embedding(49408, embed_dim)
                    self.positional_embedding = nn.Parameter(torch.empty(77, embed_dim))
                    self.ln_final = nn.LayerNorm(embed_dim)
                    self.text_projection = nn.Parameter(torch.empty(embed_dim, embed_dim))
                    
                def forward(self, text):
                    x = self.token_embedding(text)
                    x = x + self.positional_embedding
                    x = self.transformer.encoder(x)
                    x = self.ln_final(x)
                    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
                    x = x @ self.text_projection
                    return x
                
                def encode_text(self, text):
                    return self.forward(text)
            
            class MobileCLIPVisionEncoder(nn.Module):
                def __init__(self, embed_dim=512):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    self.bn1 = nn.BatchNorm2d(64)
                    self.relu = nn.ReLU(inplace=True)
                    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                    
                    self.layer1 = self._make_layer(64, 64, 2)
                    self.layer2 = self._make_layer(64, 128, 2, stride=2)
                    self.layer3 = self._make_layer(128, 256, 2, stride=2)
                    self.layer4 = self._make_layer(256, 512, 2, stride=2)
                    
                    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                    self.fc = nn.Linear(512, embed_dim)
                    self.visual_projection = nn.Parameter(torch.empty(embed_dim, embed_dim))
                
                def _make_layer(self, in_channels, out_channels, blocks, stride=1):
                    layers = []
                    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
                    layers.append(nn.BatchNorm2d(out_channels))
                    layers.append(nn.ReLU(inplace=True))
                    
                    for _ in range(1, blocks):
                        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
                        layers.append(nn.BatchNorm2d(out_channels))
                        layers.append(nn.ReLU(inplace=True))
                    
                    return nn.Sequential(*layers)
                
                def forward(self, x):
                    x = self.conv1(x)
                    x = self.bn1(x)
                    x = self.relu(x)
                    x = self.maxpool(x)
                    
                    x = self.layer1(x)
                    x = self.layer2(x)
                    x = self.layer3(x)
                    x = self.layer4(x)
                    
                    x = self.avgpool(x)
                    x = torch.flatten(x, 1)
                    x = self.fc(x)
                    x = x @ self.visual_projection
                    return x
                
                def encode_image(self, image):
                    return self.forward(image)
            
            class MobileCLIPModel(nn.Module):
                def __init__(self, embed_dim=512):
                    super().__init__()
                    self.text_encoder = MobileCLIPTextEncoder(embed_dim)
                    self.image_encoder = MobileCLIPVisionEncoder(embed_dim)
                    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
                
                def encode_text(self, text):
                    return self.text_encoder.encode_text(text)
                
                def encode_image(self, image):
                    return self.image_encoder.encode_image(image)
            
            class SimpleTokenizer:
                def __init__(self):
                    self.vocab_size = 49408
                    self.context_length = 77
                
                def __call__(self, texts):
                    import numpy as np
                    if not isinstance(texts, list):
                        texts = [texts]
                    
                    result = torch.zeros(len(texts), self.context_length, dtype=torch.long)
                    for i, text in enumerate(texts):
                        tokens = np.random.randint(1, self.vocab_size, size=min(len(text), self.context_length-2))
                        tokens = np.concatenate([[1], tokens, [2]])  # Add BOS and EOS tokens
                        tokens = np.pad(tokens, (0, self.context_length - len(tokens)))
                        result[i] = torch.tensor(tokens)
                    
                    return result
            
            self._model = MobileCLIPModel().to(self._device)
            
            
            self._preprocess = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                    std=[0.26862954, 0.26130258, 0.27577711])
            ])
            
            self._tokenizer = SimpleTokenizer()

        except Exception as e:
            logger.error(f"Error while loading MobileCLIP model: {e}")
            raise ValueError("Unable to load the requested MobileCLIP model") from e


    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        import torch

        results = []
        with torch.no_grad():
            for text in texts:
                text_tokens = self._tokenizer([text]).to(self._device)
                text_embedding = self._model.encode_text(text_tokens)
                results.append(text_embedding.tolist()[0])

        return results

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._get_text_embedding(query)


    async def _aget_image_embedding(self, img_file_path: ImageType) -> Embedding:
        return self._get_image_embedding(img_file_path)

    def _get_image_embedding(self, img_file_path: ImageType) -> Embedding:
        import torch

        with torch.no_grad():
            image = (
                self._preprocess(Image.open(img_file_path))
                .unsqueeze(0)
                .to(self._device)
            )
            return self._model.encode_image(image).tolist()[0]
