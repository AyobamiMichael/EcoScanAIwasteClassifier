"""
Quick inference test script to verify model works before deployment
Run this before deploying to catch any issues early
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import sys
from pathlib import Path

def test_model_loading():
    """Test if model loads correctly"""
    print("=" * 60)
    print("🧪 Testing Model Loading...")
    print("=" * 60)
    
    try:
        # Check if model file exists
        model_path = "model/ecoscan_model.pth"
        if not Path(model_path).exists():
            print(f"❌ Model file not found: {model_path}")
            print("   Please place your trained model in the model/ folder")
            return False
        
        print(f"✅ Found model file: {model_path}")
        
        # Check class names
        class_names_path = "model/class_names.json"
        if not Path(class_names_path).exists():
            print(f"❌ Class names file not found: {class_names_path}")
            return False
        
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        
        print(f"✅ Found {len(class_names)} classes: {class_names}")
        
        # Load model architecture
        print("\n🏗️  Building model architecture...")
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, len(class_names))
        )
        
        # Load weights
        print("📦 Loading weights...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully on {device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference():
    """Test inference on a dummy image"""
    print("\n" + "=" * 60)
    print("🔍 Testing Inference...")
    print("=" * 60)
    
    try:
        # Load model
        model_path = "model/ecoscan_model.pth"
        class_names_path = "model/class_names.json"
        
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, len(class_names))
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Create dummy image
        print("📸 Creating test image (300x300 RGB)...")
        dummy_image = Image.new('RGB', (300, 300), color='blue')
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(dummy_image).unsqueeze(0).to(device)
        
        # Run inference
        print("🚀 Running inference...")
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
        
        print(f"✅ Inference successful!")
        print(f"   Predicted: {predicted_class}")
        print(f"   Confidence: {confidence_score*100:.2f}%")
        
        # Show top-3 predictions
        print("\n📊 Top-3 Predictions:")
        top3_probs, top3_indices = torch.topk(probabilities[0], min(3, len(class_names)))
        for prob, idx in zip(top3_probs, top3_indices):
            print(f"   {class_names[idx.item()]}: {prob.item()*100:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dependencies():
    """Test if all required packages are installed"""
    print("\n" + "=" * 60)
    print("📦 Testing Dependencies...")
    print("=" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'gradio': 'Gradio',
        'cv2': 'OpenCV (cv2)',
        'numpy': 'NumPy'
    }
    
    all_installed = True
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def test_file_structure():
    """Test if project structure is correct"""
    print("\n" + "=" * 60)
    print("📂 Testing File Structure...")
    print("=" * 60)
    
    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
        "model/ecoscan_model.pth",
        "model/class_names.json"
    ]
    
    optional_files = [
        "examples/plastic_bottle.jpg",
        "examples/cardboard_box.jpg",
        "examples/glass_jar.jpg"
    ]
    
    all_present = True
    
    print("\n🔍 Required files:")
    for file_path in required_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
            print(f"✅ {file_path} ({size:.2f} MB)")
        else:
            print(f"❌ {file_path} - MISSING")
            all_present = False
    
    print("\n🎨 Optional files:")
    for file_path in optional_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"⚠️  {file_path} - not found (optional)")
    
    return all_present

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  🌱 EcoScan - Pre-Deployment Testing Suite  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    tests = [
        ("File Structure", test_file_structure),
        ("Dependencies", test_dependencies),
        ("Model Loading", test_model_loading),
        ("Inference", test_inference)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your app is ready for deployment!")
        print("\nNext steps:")
        print("  1. Test locally: python app.py")
        print("  2. Deploy to Hugging Face Spaces")
        print("  3. Share with the world! 🌍")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Please fix the issues above before deploying.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Download model from Kaggle to model/ folder")
        print("  - Verify file paths match your structure")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())