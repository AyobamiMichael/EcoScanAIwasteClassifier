import torch
from pathlib import Path

# Check model file
model_path = "model/ecoscan_model.pth"
print(f"Model exists: {Path(model_path).exists()}")

# Load and inspect
if Path(model_path).exists():
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"\nModel info:")
    print(f"Type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"Keys: {checkpoint.keys()}")
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Check shapes
    print(f"\nLayer shapes:")
    for key, value in list(state_dict.items())[:5]:
        print(f"  {key}: {value.shape}")
    
    # Check classifier
    if 'classifier.1.weight' in state_dict:
        weight = state_dict['classifier.1.weight']
        print(f"\nClassifier output: {weight.shape[0]} classes")
        print(f"Classifier input: {weight.shape[1]} features")