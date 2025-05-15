import logging
import os
import sys
from typing import Any, List

from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.schema import ImageType
from PIL import Image

logger = logging.getLogger(__name__)

MOBILECLIP_PATH = '/home/ubuntu/repos/llama_index/reference/ml-mobileclip-main'
sys.path.append(MOBILECLIP_PATH)

DEFAULT_MOBILECLIP_MODEL = "mobileclip_s0"


class MobileClipEmbedding(MultiModalEmbedding):
    """
    MobileClip embedding model for encoding text and image for Multi-Modal purpose.

    This class provides an interface to generate embeddings using MobileClip.
    At initialization, it requires a model name and path to weights.

    Note:
        Requires the MobileClip package to be available.
    """

    embed_batch_size: int = Field(default=DEFAULT_EMBED_BATCH_SIZE, gt=0)
    weights_path: str = Field(default="")

    _model: Any = PrivateAttr()
    _preprocess: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _device: Any = PrivateAttr()

    @classmethod
    def class_name(cls) -> str:
        return "MobileClipEmbedding"

    def __init__(
        self,
        *,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        model_name: str = DEFAULT_MOBILECLIP_MODEL,
        weights_path: str = "",
        **kwargs: Any,
    ):
        """
        Initializes the MobileClipEmbedding class.

        Args:
            embed_batch_size (int, optional): The batch size for embedding generation.
                Defaults to 10, must be > 0.
            model_name (str): The model name of MobileClip model.
            weights_path (str): Path to the MobileClip weights.

        Raises:
            ImportError: If the MobileClip package is not available.
            ValueError: If the model cannot be loaded.
        """
        if embed_batch_size <= 0:
            raise ValueError(f"Embed batch size {embed_batch_size} must be > 0.")

        try:
            import torch
            import mobileclip
        except ImportError:
            raise ImportError(
                "MobileClipEmbedding requires the MobileClip package."
            )

        super().__init__(
            embed_batch_size=embed_batch_size, 
            model_name=model_name, 
            weights_path=weights_path,
            **kwargs
        )

        try:
            self._device = "cpu"
            
            if not self.weights_path:
                default_weights_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                    "checkpoints",
                    f"{model_name}.pt"
                )
                if not os.path.exists(default_weights_path):
                    os.makedirs(os.path.dirname(default_weights_path), exist_ok=True)
                    import urllib.request
                    urllib.request.urlretrieve(
                        f"https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/{model_name}.pt",
                        default_weights_path
                    )
                self.weights_path = default_weights_path
            
            self._model, _, self._preprocess = mobileclip.create_model_and_transforms(
                self.model_name, 
                pretrained=self.weights_path,
                device=self._device
            )
            self._tokenizer = mobileclip.get_tokenizer(self.model_name)

        except Exception as e:
            logger.error(f"Error while loading MobileClip model: {e}")
            raise ValueError("Unable to load the MobileClip model") from e


    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        import torch
        
        results = []
        text_tokens = self._tokenizer(texts)
        
        with torch.no_grad():
            text_embeddings = self._model.encode_text(text_tokens)
            
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
            for embedding in text_embeddings:
                results.append(embedding.tolist())
        
        return results

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._get_text_embedding(query)


    async def _aget_image_embedding(self, img_file_path: ImageType) -> Embedding:
        return self._get_image_embedding(img_file_path)

    def _get_image_embedding(self, img_file_path: ImageType) -> Embedding:
        import torch

        with torch.no_grad():
            image = (
                self._preprocess(Image.open(img_file_path).convert('RGB'))
                .unsqueeze(0)
                .to(self._device)
            )
            image_embedding = self._model.encode_image(image)
            
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            
            return image_embedding.tolist()[0]
