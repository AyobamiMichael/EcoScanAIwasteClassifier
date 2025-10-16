"""
EcoScan - AI-Powered Waste Sorting Classifier
Using Gradio Interface for Deployment

"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr
import numpy as np
import cv2
import json
from pathlib import Path
from huggingface_hub import hf_hub_download


#
# CONFIGURATION
# 

class Config:
    MODEL_PATH = "model/ecoscan_model.pth"
    CLASS_NAMES_PATH = "model/class_names.json"
    MODEL_NAME = "efficientnet_b3"
    NUM_CLASSES = 6,
    IMAGE_SIZE = 300,
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# RECYCLING INFORMATION DATABASE
RECYCLING_INFO = {
    "cardboard":{
         "icon": "📦",
         "tip": "Flatten boxes to save space. Remove any plastic tape or labels. Keep dry - wet cardboard contaminates recycling.",
         "eco_score": 9,
         "decompose_time": "2-3 months",
         "facts": "Recyling 1 ton of cardboard saves 17 trees and 7,000 gallons of water!"
    },
    "glass":{
         "icon": "🍾",
         "tip": "Rinse glass containers to remove food residue. Remove lids and caps, as they are often made of different materials.",
         "eco_score": 8,
         "decompose_time": "1 million years",
         "facts": "Recycling glass saves 30% of the energy required to make new glass from raw materials."
    },
     "metal":{
          "icon": "🔩",
          "tip": "Rinse aluminum cans and steel containers, Crush cans to save space. Metal recyling saves 95% of  enerdy!",
          "eco_score": 9,
          "decompose_time": "50-500 years",
          "facts": "Recycling aluminum saves 95% of the energy needed to make new aluminium from raw materials. "


    },
    "paper":{
       "icon": "📄",
        "tip": "Keep paper dry and clean. Remove staples and paper clips. Shred sensitive documents before recylcing.",
        "eco_score": 8,
        "decompose_time": "2-6 weeks",
        "facts": "Recycling 1 ton of paper saves 17 trees, 380 gallons of oil, and 7,000 gallons of water."      

    },
    "plastic":{
        "icon": "🧴",
        "tip": "Rinse plastic containers to remove food residue. Check the recycling symbol and number to ensure it's accepted in your local program.",
        "eco_score": 4,
        "decompose_time": "450-1000 years",
        "facts": "Only about 9% of all plastic waste ever produced has been recycled. Recycling plastic saves 88% of the energy compared to producing new plastic from raw materials."
    },
    "trash":{
        "icon": "🗑️",
        "tip": "This item is general waste or e-waste. Check for specialized recylcing programs. Consider composting organic materials",
        "eco_score": 3,
        "decompose_time": "Variable (decades to never)",
        "facts": "E-waste contains valuable materials like gold and copper, but also toxic substances. Always use proper disposal."
    }
    
}


# MODEL LOADING

def load_model():
    """Load the trained model"""
    print(f"Loading model on {config.DEVICE}...")


      # Download from Hub if not local
    if not Path(config.MODEL_PATH).exists():
        print("Downloading model from Hugging Face Hub...")
        try:
            hf_hub_download(
                repo_id="AyobamiMichael/ecoscan-model",
                filename="ecoscan_model.pth",
                local_dir="model",
                repo_type="model"
            )
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise

    # Check if model file exists
    if not Path(config.MODEL_PATH).exists():
        raise FileNotFoundError(f"Model file not found:{config.MODEL_PATH}")
    
    print(f"Loading complete model from: {config.MODEL_PATH}")
    # Create mode architecture
    if config.MODEL_NAME == "efficientnet_b3":
        from torchvision.models import efficientnet_b3

        # Load pretrianed model to get correct architecture
        print("Building EfficientNet-B3 architecture...")
        model = efficientnet_b3(weights=None)

        # Get the input features from the last layer
        in_features = 1536
        num_classes = 6 
        print(f"EfficinetNet-B3 classifier input features: {in_features}")

        # Replace classifier
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    elif config.MODEL_NAME == "resnet50":
        from torchvision.models import resnet50

        print("Building ResNet50 architecture...")
        model = resnet50(weights=None)

        # Get the input features
        in_features = 2048
        num_classes = 6 
        print(f"ResNet50 fc input features: {in_features}")

        # Replace final layer
        model.fc = nn.Linear(in_features,num_classes)

    else:
        raise ValueError(f"Unknown model: {config.MODEL_NAME}")
    
    # Load trained weights
    print(f"Loading weights from: {config.MODEL_PATH}")
    state_dict = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    try:
        #state_dict = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
        model.load_state_dict(state_dict, strict=True)
        print("✅ All weights loaded successfully!")
    except Exception as e:
        print(f"⚠️  Warning: {e}")
        print("Some weights may not match. Loading with strict=False...")
        model.load_state_dict(state_dict, strict=False)
        print("✅ Weights loaded (partial)")
    
    model.to(config.DEVICE)
    model.eval()

    # Verify the model
    print(f"✅ Model ready on {config.DEVICE}")
    print(f"   Input features: {in_features}")
    print(f"   Output classes: {config.NUM_CLASSES}")
    
    return model


def load_class_names():
    """"Load class names from JSON file"""
    with open(config.CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    return class_names


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def get_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

# ============================================================================
# GRAD-CAM VISUALIZATION
# ============================================================================

class GradCAM:
    """"Gradient-weighted Class Activation Mapping"""


    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, class_idx):
        """Generate CAM for a specific class"""
        
        try:
            # Forward pass
            output = self.model(input_image)
            
            # Backward pass
            self.model.zero_grad()
            class_loss = output[0, class_idx]
            class_loss.backward()
            
            # Generate CAM
            if self.gradients is None or self.activations is None:
                print("Warning: gradients or activations not captured")
                return np.ones((input_image.shape[2], input_image.shape[3]))
            
            gradients = self.gradients[0]  # [C, H, W]
            activations = self.activations[0]  # [C, H, W]
            
            # Global average pooling on gradients
            weights = torch.mean(gradients, dim=(1, 2))  # [C]
            
            # Weighted combination
            cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
            for i, w in enumerate(weights):
                cam += w * activations[i]
            
            # ReLU
            cam = torch.relu(cam)
            
            # Normalize
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max - cam_min > 0:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)
            
            return cam.cpu().numpy()
        
        except Exception as e:
            print(f"Grad-CAM generation error: {e}")
            return np.ones((input_image.shape[2], input_image.shape[3]))

def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay heatmap on original image"""
    
    # Ensure image is numpy array
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert BGR to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
    
    return overlay

# Global MODELAND CLASS NAMES (will be loaded at startup)

model = None
class_names = None

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def classify_image(image):
    """Main classification function """

    global model, class_names

    if image is None:
        return None, None, "Please upload an image first!"
    

    # Convert to PIL Image
    
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image

    # Preprocess
    transform  = get_transforms()
    input_tensor = transform(pil_image).unsqueeze(0).to(config.DEVICE)

    # Get predictions
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()

    # Generate Grad-CAM

    try:
        # Get traget layer
        if config.MODEL_NAME == "efficientnet_b3":
            target_layer = model.features[-1]
        elif config.MODEL_NAME == "resnet50":
            target_layer = model.layer4[-1]
         
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate_cam(input_tensor, predicted.item())

        # Create overlay
        original_img = np.array(pil_image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))
        heatmap_img = gradcam.overlay_heatmap(original_img, cam)
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        heatmap_img = np.array(pil_image)

    
    # Get recycling info
    info = RECYCLING_INFO.get(predicted_class, RECYCLING_INFO["trash"])

    # Format predictions for top-3
    top3_probs, top3_indices = torch.topk(probabilities[0], 3)
    predictions_dict = {}
    for prob, idx in zip(top3_probs, top3_indices):
        class_name = class_names[idx.item()]
        confidence = float(prob.item())
        predictions_dict[class_name] = confidence

    
    # Create detailed output
     # Create detailed output
    output_text = f"""
## {info['icon']} Classification Result

**Detected Material:** {predicted_class.upper()}  
**Confidence:** {confidence_score*100:.1f}%

---

### ♻️ Recycling Instructions
{info['tip']}

---

### 📊 Environmental Impact
- **EcoScore:** {info['eco_score']}/10
- **Decomposition Time:** {info['decompose_time']}

### 💡 Did You Know?
{info['facts']}
    """
    
    return predictions_dict, heatmap_img, output_text   



# ============================================================================
# INITIALIZE MODEL & CLASS NAMES AT STARTUP
# ============================================================================

print("🚀 Initializing EcoScan...")
model = load_model()
class_names = load_class_names()
print(f"✅ Loaded {len(class_names)} classes: {class_names}")
print("🌱 EcoScan ready!")




# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Custom CSS
custom_css = """
#title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
#output-box {
    border: 2px solid #667eea;
    border-radius: 10px;
    padding: 15px;
}
.eco-high { color: #10b981; font-weight: bold; }
.eco-medium { color: #f59e0b; font-weight: bold; }
.eco-low { color: #ef4444; font-weight: bold; }
"""

# Example images
examples = [
    ["examples/plastic_bottle.jpg"] if Path("examples/plastic_bottle.jpg").exists() else None,
    ["examples/cardboard_box.jpg"] if Path("examples/cardboard_box.jpg").exists() else None,
    ["examples/glass_jar.jpg"] if Path("examples/glass_jar.jpg").exists() else None,
]
examples = [ex for ex in examples if ex is not None]

# Create interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        <div id="title">
            <h1>🌱 EcoScan - AI Waste Classifier</h1>
            <p>Upload an image of waste material to get instant classification and recycling guidance</p>
        </div>
        """,
        elem_id="title"
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📸 Upload Waste Image",
                type="pil",
                height=400
            )
            
            classify_btn = gr.Button(
                "🔍 Classify Waste",
                variant="primary",
                size="lg"
            )
            
            gr.Markdown(
                """
                ### 📋 Instructions
                1. Upload a clear image of waste material
                2. Click "Classify Waste"
                3. View classification and recycling tips
                
                ### 🎯 Supported Categories
                Cardboard • Glass • Metal • Paper • Plastic • General Waste
                """
            )
        
        with gr.Column(scale=1):
            with gr.Tab("📊 Results"):
                predictions = gr.Label(
                    label="Classification Confidence",
                    num_top_classes=3
                )
                recycling_info = gr.Markdown(
                    label="Recycling Information",
                    elem_id="output-box"
                )
            
            with gr.Tab("🔥 AI Visualization"):
                heatmap = gr.Image(
                    label="Attention Map (What the AI sees)",
                    height=400
                )
                gr.Markdown(
                    """
                    **Grad-CAM Visualization**: Warmer colors (red/yellow) show regions 
                    the AI focused on for classification. Cooler colors (blue) indicate 
                    less important regions.
                    """
                )
    
    # Examples section
    if examples:
        gr.Examples(
            examples=examples,
            inputs=input_image,
            label="📷 Try These Examples"
        )
    
    # Footer
    gr.Markdown(
        """
        ---
        <div style="text-align: center; color: #666;">
            <p>Built with ❤️ for a sustainable future | Powered by EfficientNet-B3 & PyTorch</p>
            <p>💡 <strong>Tip:</strong> This AI model was trained on 2,500+ waste images with 90%+ accuracy</p>
        </div>
        """
    )
    
    # Connect button
    classify_btn.click(
        fn=classify_image,
        inputs=input_image,
        outputs=[predictions, heatmap, recycling_info]
    )

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        debug=True
    
    )