
title: EcoScan - AI Waste Classifier
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit


# 🌱 EcoScan - AI-Powered Waste Sorting Classifier

![EcoScan Banner](https://img.shields.io/badge/AI-Waste_Classification-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange?style=for-the-badge)

An intelligent waste classification system that helps promote smart recycling and sustainability through AI-powered image recognition.

## 🎯 Features

- **🔍 Real-time Classification**: Upload waste images and get instant predictions
- **📊 Confidence Scores**: See top-3 predictions with confidence percentages
- **🔥 Explainable AI**: Grad-CAM visualization shows what the model focuses on
- **♻️ Recycling Guidance**: Get specific tips for each material type
- **🌍 Environmental Impact**: Learn decomposition times and eco-scores
- **📱 Easy to Use**: Clean, intuitive Gradio interface

## 🗂️ Supported Categories

| Category | Icon | EcoScore | Decomposition Time |
|----------|------|----------|-------------------|
| Cardboard | 📦 | 9/10 | 2-3 months |
| Glass | 🥃 | 8/10 | 1 million years |
| Metal | 🔩 | 9/10 | 50-500 years |
| Paper | 📄 | 8/10 | 2-6 weeks |
| Plastic | 🧴 | 4/10 | 450-1000 years |
| General Waste | 🗑️ | 3/10 | Variable |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ecoscan.git
cd ecoscan

# Install dependencies
pip install -r requirements.txt
```

### Project Structure

```
ecoscan/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── model/
│   ├── ecoscan_model.pth      # Trained model weights
│   └── class_names.json       # Class label mappings
└── examples/             # Sample images (optional)
    ├── plastic_bottle.jpg
    ├── cardboard_box.jpg
    └── glass_jar.jpg
```

### Running Locally

```bash
python app.py
```

Then open your browser to `http://localhost:7860`

## 🧠 Model Details

- **Architecture**: EfficientNet-B3
- **Input Size**: 300x300 pixels
- **Training Dataset**: TrashNet + Custom curated data (~2,500 images)
- **Accuracy**: 90%+ on test set
- **Framework**: PyTorch
- **Inference Time**: <2 seconds per image

## 🔧 Technical Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch 2.0+ |
| Model Architecture | EfficientNet-B3 |
| Web Framework | Gradio 4.0+ |
| Computer Vision | OpenCV, Torchvision |
| Explainability | Grad-CAM |

## 📊 Performance Metrics

- **Overall Accuracy**: 90.2%
- **Precision**: 89.5%
- **Recall**: 90.1%
- **F1-Score**: 89.8%

## 🌐 Deployment

### Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Select Gradio as the SDK
3. Upload all files from this repository
4. Your app will automatically deploy!

### Docker (Optional)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
```

## 💡 Usage Tips

1. **Best Results**: Use well-lit, clear images with minimal background clutter
2. **Multiple Items**: For best accuracy, photograph one item at a time
3. **Angle**: Capture the item from a recognizable angle
4. **Distance**: Fill at least 50% of the frame with the waste item

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: TrashNet by Gary Thung and Mindy Yang
- Model Architecture: EfficientNet by Google Research
- Framework: PyTorch by Meta AI
- Interface: Gradio by Hugging Face

## 📧 Contact

For questions or feedback, please open an issue or reach out via email.

---

<div align="center">
  <p>Built with ❤️ for a sustainable future</p>
  <p>⭐ Star this repo if you find it useful!</p>
</div>