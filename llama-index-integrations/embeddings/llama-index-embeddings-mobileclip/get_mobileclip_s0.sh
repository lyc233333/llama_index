#
#
mkdir -p checkpoints
echo "Downloading MobileCLIP S0 model weight..."
wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt -P checkpoints
echo "Download complete. Model weight saved to checkpoints/mobileclip_s0.pt"
