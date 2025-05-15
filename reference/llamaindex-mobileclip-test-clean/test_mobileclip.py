import torch
from PIL import Image
import sys
import os

sys.path.append('/home/ubuntu/repos/llama_index/reference/ml-mobileclip-main')
import mobileclip

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints/mobileclip_s0.pt')
TEST_IMAGE_PATH = '/home/ubuntu/repos/llama_index/reference/ml-mobileclip-main/docs/fig_accuracy_latency.png'

def test_mobileclip_inference():
    model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained=WEIGHTS_PATH)
    tokenizer = mobileclip.get_tokenizer('mobileclip_s0')
    
    image = preprocess(Image.open(TEST_IMAGE_PATH).convert('RGB')).unsqueeze(0)
    text = tokenizer(["a diagram", "a dog", "a cat"])
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    print("Label probs:", text_probs)
    print("Image embedding shape:", image_features.shape)
    print("Text embedding shape:", text_features.shape)
    
    image_embeddings_list = image_features.tolist()[0]
    print("\nFirst 5 values of image embeddings:")
    for i, val in enumerate(image_embeddings_list[:5]):
        print(f"  {i}: {val}")
    
    text_embeddings_list = text_features.tolist()
    print("\nFirst 5 values of text embeddings for 'a diagram':")
    for i, val in enumerate(text_embeddings_list[0][:5]):
        print(f"  {i}: {val}")
    
    return image_features.tolist()[0]

if __name__ == "__main__":
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints/mobileclip_s0.pt')):
        print("Downloading weights...")
        os.system(f"bash {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'get_mobileclip_s0.sh')}")
    
    embeddings = test_mobileclip_inference()
    print("Test successful!")
