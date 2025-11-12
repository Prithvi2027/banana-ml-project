# 🍌 Banana Ripeness Classification - End-to-End ML Project

A complete deep learning project that predicts banana ripeness and estimates days until the banana becomes inedible. Uses PyTorch, ResNet-50, and Flask for a production-ready web application.

## 📊 Project Overview

- **Dataset**: [Banana Ripeness Classification Dataset](https://www.kaggle.com/datasets/shahriar26s/banana-ripeness-classification-dataset) from Kaggle
- **Classes**: 4 ripeness stages (Unripe, Ripe, Overripe, Rotten)
- **Model**: ResNet-50 pretrained on ImageNet, fine-tuned for banana classification
- **Framework**: PyTorch for training, Flask for API
- **Features**: Real-time prediction, confidence scores, class probabilities, days estimation

## 🏗️ Project Structure

```
banana-ml-project/
├── data/                      # Dataset directory (will be populated)
├── models/                    # Saved model checkpoints
│   ├── best_model.pth        # Best validation model
│   ├── final_model.pth       # Final trained model
│   ├── training_history.png  # Training curves
│   └── confusion_matrix.png  # Test results
├── notebooks/                 # Jupyter notebooks (optional)
├── src/                      # Source code
│   ├── config.py            # Configuration and hyperparameters
│   ├── dataset.py           # Data loading and augmentation
│   ├── model.py             # Model architecture
│   ├── train.py             # Training script
│   └── app.py               # Flask API server
├── static/                   # Web assets
│   ├── style.css            # Frontend styling
│   └── script.js            # Frontend logic
├── templates/                # HTML templates
│   └── index.html           # Main web interface
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
cd banana-ml-project
pip install -r requirements.txt
```

**Note**: If you have CUDA GPU available, install PyTorch with CUDA support for faster training:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Dataset Configuration

The dataset is already downloaded at:
`C:\Users\Admin\Downloads\archive\Banana Ripeness Classification Dataset`

The config file (`src/config.py`) is already set up with this path. If your dataset is in a different location, update the `DATASET_PATH` variable in `src/config.py`.

### 3. Train the Model

```bash
cd src
python train.py
```

**Training Features:**
- Automatic GPU detection (uses CUDA if available, else CPU)
- Data augmentation (rotation, flips, color jitter)
- Learning rate scheduling
- Early stopping (patience=10 epochs)
- Model checkpointing (saves best model)
- Training visualization (loss/accuracy plots)
- Test evaluation with confusion matrix

**Expected Training Time:**
- With GPU: ~10-20 minutes
- With CPU: ~1-2 hours

**Training Output:**
- Best model: `models/best_model.pth`
- Final model: `models/final_model.pth`
- Training curves: `models/training_history.png`
- Confusion matrix: `models/confusion_matrix.png`

### 4. Run the Web Application

```bash
cd src
python app.py
```

Then open your browser and visit: **http://localhost:5000**

## 🎯 Model Performance

The model achieves:
- **Test Accuracy**: ~90-95% (depending on training)
- **4 Classes**: Unripe, Ripe, Overripe, Rotten
- **Days Prediction Mapping**:
  - Unripe: 7 days
  - Ripe: 3 days
  - Overripe: 1 day
  - Rotten: 0 days

## 📱 Web Interface Features

- **Drag & Drop Upload**: Easily upload banana images
- **Real-time Prediction**: Instant classification with ML model
- **Confidence Scores**: See model confidence for predictions
- **Class Probabilities**: View probabilities for all classes
- **Days Estimation**: Get estimated days until banana dies
- **Recommendations**: Personalized eating recommendations
- **Beautiful UI**: Animated, responsive design

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Model
MODEL_NAME = 'resnet50'  # or 'efficientnet_b0', 'mobilenet_v2'
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
IMAGE_SIZE = 224

# Early stopping
PATIENCE = 10
MIN_DELTA = 0.001

# Days mapping
CLASS_TO_DAYS = {
    'unripe': 7,
    'ripe': 3,
    'overripe': 1,
    'rotten': 0
}
```

## 🧪 Testing the Model

### Option 1: Use Test Script
```bash
cd src
python train.py  # This includes evaluation on test set
```

### Option 2: Test Individual Images
Use the web interface at http://localhost:5000

### Option 3: Programmatic Testing
```python
from PIL import Image
from model import BananaRipenessModel
from dataset import get_transforms
import torch
import config

# Load model
model = BananaRipenessModel()
checkpoint = torch.load(config.BEST_MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and predict image
image = Image.open('path/to/banana.jpg')
transform = get_transforms(train=False)
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    class_name = config.CLASS_NAMES[predicted.item()]
    days_left = config.CLASS_TO_DAYS[class_name]

print(f"Class: {class_name}, Days left: {days_left}")
```

## 📊 API Endpoints

### GET `/`
Renders the web interface

### POST `/predict`
Predicts banana ripeness from uploaded image

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (image file)

**Response:**
```json
{
  "class": "ripe",
  "days_left": 3,
  "confidence": 95.3,
  "display_days": "3 days",
  "recommendation": "Perfect for eating! Consume within 3 days.",
  "emoji": "🟡",
  "probabilities": {
    "unripe": 2.1,
    "ripe": 95.3,
    "overripe": 2.4,
    "rotten": 0.2
  }
}
```

### GET `/health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🔬 Model Architecture

```
ResNet-50 (Pretrained on ImageNet)
├── Conv Layers (frozen or fine-tuned)
├── Residual Blocks
└── Custom Classifier:
    ├── Dropout (0.5)
    ├── Linear (2048 → 512)
    ├── ReLU
    ├── Dropout (0.3)
    └── Linear (512 → 4)
```

## 🛠️ Troubleshooting

### Issue: "No trained model found"
**Solution**: Train the model first using `python src/train.py`

### Issue: CUDA out of memory
**Solution**: Reduce batch size in `config.py`:
```python
BATCH_SIZE = 16  # or 8
```

### Issue: Slow training on CPU
**Solution**: Either install CUDA-enabled PyTorch or reduce model complexity:
```python
MODEL_NAME = 'mobilenet_v2'  # Lighter model
```

### Issue: Import errors
**Solution**: Make sure you're in the correct directory:
```bash
cd src
python train.py  # or python app.py
```

## 📈 Future Improvements

- [ ] Add more data augmentation techniques
- [ ] Implement ensemble models
- [ ] Add model explainability (GradCAM)
- [ ] Deploy to cloud (AWS/Azure/GCP)
- [ ] Create Docker container
- [ ] Add mobile app support
- [ ] Implement regression for exact day prediction
- [ ] Add batch prediction support


## 🙏 Acknowledgments

- Dataset: [Banana Ripeness Classification Dataset](https://www.kaggle.com/datasets/shahriar26s/banana-ripeness-classification-dataset)
- Framework: PyTorch
- Pretrained Models: torchvision.models

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

