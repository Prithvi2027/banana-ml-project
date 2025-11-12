# 🚀 Quick Start Guide

Get your banana ripeness classifier running in 3 steps!

## Step 1: Install Dependencies (2 minutes)

Open PowerShell in the project folder and run:

```powershell
pip install -r requirements.txt
```

## Step 2: Train the Model (10-60 minutes)

Double-click `train_model.bat` or run:

```powershell
cd src
python train.py
```

**What happens:**
- Loads 5000+ banana images from your dataset
- Trains ResNet-50 CNN with transfer learning
- Saves best model to `models/best_model.pth`
- Creates visualization plots

**Time required:**
- With GPU: 10-20 minutes
- With CPU: 1-2 hours

## Step 3: Run the Web App

Double-click `run_app.bat` or run:

```powershell
cd src
python app.py
```

Then open: **http://localhost:5000**

## 📸 Using the App

1. **Upload** a banana image (drag & drop or click)
2. **Wait** for AI analysis (~1 second)
3. **Get results**:
   - Days until overripe
   - Ripeness classification
   - Model confidence
   - Eating recommendation

## 🎯 What You Get

- **4 Classes**: Unripe, Ripe, Overripe, Rotten
- **Days Prediction**: 
  - Unripe → 7 days
  - Ripe → 3 days
  - Overripe → 1 day
  - Rotten → 0 days
- **High Accuracy**: ~90-95% on test set
- **Real-time**: Predictions in < 1 second

## 📁 Project Files

```
banana-ml-project/
├── train_model.bat      ← Double-click to train
├── run_app.bat          ← Double-click to run app
├── requirements.txt     ← Dependencies
├── README.md            ← Full documentation
├── src/
│   ├── train.py        ← Training script
│   ├── app.py          ← Web server
│   ├── model.py        ← Neural network
│   ├── dataset.py      ← Data pipeline
│   └── config.py       ← Settings
├── models/             ← Saved models appear here
├── static/             ← CSS/JS for web
└── templates/          ← HTML files
```

## 🛠️ Troubleshooting

**Problem**: "No module named 'torch'"
→ Run: `pip install -r requirements.txt`

**Problem**: "No trained model found"
→ Train first: Double-click `train_model.bat`

**Problem**: Training is slow
→ Normal on CPU! Use GPU for 10x speedup

**Problem**: Port 5000 already in use
→ Change port in `src/app.py` line 154: `app.run(port=5001)`

## 💡 Tips

- Use well-lit banana photos for best results
- The model works on single bananas or bunches
- Try different ripeness stages to see accuracy
- Check `models/confusion_matrix.png` after training

## 📊 Training Outputs

After training, check the `models/` folder:

- `best_model.pth` - Your trained model
- `training_history.png` - Loss/accuracy curves
- `confusion_matrix.png` - Test results visualization

## 🎓 Next Steps

- Read full `README.md` for advanced options
- Customize hyperparameters in `src/config.py`
- Try different models: ResNet, EfficientNet, MobileNet
- Deploy to cloud for public access

---

**Need help?** Check the full README.md or open an issue!
