import logging
from typing import Any, List

from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.schema import ImageType
from PIL import Image

logger = logging.getLogger(__name__)


DEFAULT_MOBILECLIP_MODEL = "MobileCLIP-S2"


class MobileclipEmbedding(MultiModalEmbedding):
    """
    MobileCLIP embedding models for encoding text and image for Multi-Modal purpose.

    This class provides an interface to generate embeddings using a model
    from the MobileCLIP family. At the initialization it requires a model name
    of MobileCLIP.

    Note:
        Requires `open_clip_torch` package to be available in the PYTHONPATH.
    """

    embed_batch_size: int = Field(default=DEFAULT_EMBED_BATCH_SIZE, gt=0)

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
        **kwargs: Any,
    ):
        """
        Initializes the MobileclipEmbedding class.

        During the initialization the `open_clip_torch` package is imported.

        Args:
            embed_batch_size (int, optional): The batch size for embedding generation. Defaults to 10,
                must be > 0.
            model_name (str): The model name of MobileCLIP model.

        Raises:
            ImportError: If the `open_clip_torch` package is not available in the PYTHONPATH.
            ValueError: If the model cannot be loaded or if the embed_batch_size
                is not > 0.
        """
        if embed_batch_size <= 0:
            raise ValueError(f"Embed batch size {embed_batch_size} must be > 0.")

        try:
            import open_clip
            import torch
        except ImportError:
            raise ImportError(
                "MobileclipEmbedding requires `pip install open-clip-torch` and torch."
            )

        super().__init__(
            embed_batch_size=embed_batch_size, model_name=model_name, **kwargs
        )

        try:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name=self.model_name, device=self._device
            )
            self._tokenizer = open_clip.get_tokenizer(model_name=self.model_name)

        except Exception as e:
            logger.error("Error while loading MobileCLIP model.")
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
