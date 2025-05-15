import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Running basic MobileClip inference test...")
os.system("python test_mobileclip.py")

print("\nRunning MobileClip embeddings integration test...")
os.system("python tests/test_mobileclip.py")
