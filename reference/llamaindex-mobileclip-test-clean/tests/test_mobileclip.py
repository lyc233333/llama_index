import os
import sys
import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llama_index.embeddings.mobileclip import MobileClipEmbedding

MOBILECLIP_PATH = '/home/ubuntu/repos/llama_index/reference/ml-mobileclip-main'
sys.path.append(MOBILECLIP_PATH)


def test_mobileclip_embedding_class():
    names_of_base_classes = [b.__name__ for b in MobileClipEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


def test_mobileclip_embedding_functionality():
    weights_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoints",
        "mobileclip_s0.pt"
    )
    
    if not os.path.exists(weights_path):
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "get_mobileclip_s0.sh"
        )
        os.system(f"bash {script_path}")
    
    test_image_path = os.path.join(
        MOBILECLIP_PATH,
        "docs/fig_accuracy_latency.png"
    )
    
    embedding_model = MobileClipEmbedding(
        model_name="mobileclip_s0",
        weights_path=weights_path
    )
    
    image_embedding = embedding_model.get_image_embedding(test_image_path)
    
    assert image_embedding is not None
    assert len(image_embedding) > 0
    
    text_embedding = embedding_model.get_text_embedding("This is a test")
    
    assert text_embedding is not None
    assert len(text_embedding) > 0
    
    print("MobileClip embedding test successful!")
