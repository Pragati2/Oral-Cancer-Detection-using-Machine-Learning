# Oral Cancer Detection using Deep Learning

A deep learning-based image classification model that detects oral cancer using X-ray images. Built using VGG-16 architecture and trained on a dataset of over 6,500 labeled images. Achieved 91% test accuracy, outperforming traditional ML baselines by 35%.

## 🔍 Problem Statement
Oral cancer is a significant health concern, especially in developing regions. Early detection can drastically improve survival rates. This project leverages convolutional neural networks to automate the detection of cancerous tissues from medical images.

## 📊 Dataset
- **Size:** ~6,500 X-ray images
- **Classes:** Cancerous vs. Non-Cancerous
- **Source:** [Add source or "Not publicly available due to privacy"]

> Note: The dataset was cleaned and preprocessed (resizing, normalization, augmentation) to improve model generalization.

## Model Architecture
- **Base Model:** VGG-16 (Transfer Learning)
- **Layers:** Fine-tuned top layers for binary classification
- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy

## Results
- **Accuracy:** 91%
- **Precision/Recall:** 0.89 / 0.93
- **Inference Time:** Reduced by 40% after optimization
- **Model Size:** ~82MB

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- Matplotlib / Seaborn

## Visualizations

| Original Image | Augmented | Heatmap |
|----------------|-----------|---------|
| ![orig](images/original.jpg) | ![aug](images/augmented.jpg) | ![gradcam](images/gradcam.jpg) |

> Add Grad-CAM or saliency maps to explain the model’s decision-making.

## ⚙️ How to Run

```bash
git clone https://github.com/Pragati2/Oral-Cancer-Detection-using-Machine-Learning.git
cd oral-cancer-detection
pip install -r requirements.txt
python train.py
