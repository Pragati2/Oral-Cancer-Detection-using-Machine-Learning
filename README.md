# Oral Cancer Detection - Implementation Guide

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Project Structure

```
oral-cancer-detection/
├── data/
│   ├── train/
│   │   ├── cancerous/
│   │   └── non_cancerous/
│   ├── validation/
│   │   ├── cancerous/
│   │   └── non_cancerous/
│   └── test/
│       ├── cancerous/
│       └── non_cancerous/
├── models/
│   ├── best_model.h5
│   └── unet_model.h5
├── uploads/
├── results/
├── logs/
├── templates/
│   └── index.html
├── improved_oral_cancer_detection.py
├── improved_flask_app.py
├── requirements.txt
└── README.md
```

---

## 📊 Training Models

### 1. Prepare Your Dataset

Organize your dataset in the following structure:
```
data/
├── train/
│   ├── cancerous/       # Images of oral cancer
│   └── non_cancerous/   # Images of healthy mouths
├── validation/
│   ├── cancerous/
│   └── non_cancerous/
└── test/
    ├── cancerous/
    └── non_cancerous/
```

**Recommended Dataset Size:**
- Minimum: 500 images per class
- Good: 1000+ images per class
- Excellent: 2000+ images per class

### 2. Train VGG16 Model

```python
from improved_oral_cancer_detection import *

# Create data generators
train_gen, val_gen = create_enhanced_data_generators(
    train_dir='data/train',
    val_dir='data/validation',
    batch_size=32
)

# Build VGG16 model
model = build_enhanced_vgg16(num_classes=2)

# Train
history = compile_and_train(
    model=model,
    train_gen=train_gen,
    val_gen=val_gen,
    epochs=100,
    model_name='vgg16_oral_cancer'
)

# Plot training history
plot_training_history(history, 'vgg16_oral_cancer')

# Save model
model.save('models/best_model.h5')
```

### 3. Train Alternative Models

```python
# EfficientNet (Recommended for production)
efficient_model = build_efficientnet_model(num_classes=2)
history_eff = compile_and_train(efficient_model, train_gen, val_gen, epochs=100, model_name='efficientnet')

# ResNet50 (Good for complex patterns)
resnet_model = build_resnet_model(num_classes=2)
history_res = compile_and_train(resnet_model, train_gen, val_gen, epochs=100, model_name='resnet50')

# MobileNet (Lightweight for deployment)
mobile_model = build_mobilenet_model(num_classes=2)
history_mob = compile_and_train(mobile_model, train_gen, val_gen, epochs=100, model_name='mobilenet')
```

### 4. Train U-Net for Segmentation

```python
# Build U-Net
unet_model = build_improved_unet(input_shape=(256, 256, 3))

# Compile for segmentation
unet_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)]
)

# Train (you need segmentation masks)
# unet_history = unet_model.fit(...)

# Save
unet_model.save('models/unet_model.h5')
```

---

## 🧪 Evaluation

### Comprehensive Evaluation

```python
from improved_oral_cancer_detection import *

# Load model
model = tf.keras.models.load_model('models/best_model.h5')

# Create test generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    'data/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate
results = comprehensive_evaluation(
    model=model,
    test_gen=test_gen,
    class_names=['Non-Cancerous', 'Cancerous']
)

print(f"\nSensitivity: {results['sensitivity']:.4f}")
print(f"Specificity: {results['specificity']:.4f}")
print(f"ROC AUC: {results['roc_auc']:.4f}")
```

### Performance Targets

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Accuracy | 85% | 90% | 95% |
| Sensitivity | 90% | 95% | 98% |
| Specificity | 85% | 90% | 95% |
| ROC AUC | 0.90 | 0.95 | 0.98 |

---

## 🌐 Running the Web Application

### 1. Start the Flask Server

```bash
python improved_flask_app.py
```

The server will start at `http://localhost:5000`

### 2. API Endpoints

#### Health Check
```bash
curl http://localhost:5000/health
```

#### Make Prediction
```bash
curl -X POST -F "file=@path/to/image.jpg" http://localhost:5000/predict
```

#### Get Statistics
```bash
curl http://localhost:5000/stats
```

### 3. Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

Upload an image and get instant predictions with:
- Classification result (Cancerous/Non-Cancerous)
- Confidence score
- Medical recommendation
- Grad-CAM visualization (what the AI is looking at)
- Segmentation mask (if cancerous)

---

## 🔒 Security Considerations

### File Upload Security
- Maximum file size: 16MB
- Allowed formats: PNG, JPG, JPEG
- Secure filename handling
- File integrity checks (SHA256)

### Data Privacy
- No data is stored permanently without consent
- All predictions are logged with anonymized IDs
- HIPAA compliance considerations built-in

### Production Deployment Checklist
- [ ] Change `SECRET_KEY` in config
- [ ] Set `debug=False` in app.run()
- [ ] Use HTTPS/SSL certificates
- [ ] Implement rate limiting
- [ ] Add user authentication
- [ ] Set up monitoring and alerting
- [ ] Configure backup systems
- [ ] Enable CORS if needed

---

## 📈 Model Improvement Strategies

### 1. Data Quality
- **Increase dataset size**: Aim for 2000+ images per class
- **Balance classes**: Equal distribution of cancerous/non-cancerous
- **Diverse sources**: Multiple hospitals, demographics, lighting
- **Expert validation**: Medical professionals verify labels

### 2. Advanced Augmentation
```python
# Heavy augmentation for small datasets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    channel_shift_range=20,
    fill_mode='reflect'
)
```

### 3. Ensemble Methods
```python
# Combine multiple models
def ensemble_predict(image_path, models):
    predictions = []
    for model in models:
        img = preprocess_image(image_path)
        pred = model.predict(img)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred
```

### 4. Hyperparameter Tuning
```python
import keras_tuner as kt

def build_model(hp):
    model = Sequential([
        Conv2D(
            filters=hp.Int('conv1_filters', min_value=32, max_value=128, step=32),
            kernel_size=(3, 3),
            activation='relu',
            input_shape=(224, 224, 3)
        ),
        # ... more layers with tunable parameters
    ])
    
    model.compile(
        optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10
)

tuner.search(train_gen, validation_data=val_gen, epochs=50)
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue: Model accuracy stuck at 50%**
- **Cause**: Model is guessing randomly
- **Solution**: 
  - Check data labels are correct
  - Increase dataset size
  - Simplify model architecture
  - Increase training epochs

**Issue: High training accuracy, low validation accuracy**
- **Cause**: Overfitting
- **Solution**:
  - Add more dropout layers
  - Increase data augmentation
  - Reduce model complexity
  - Get more training data

**Issue: Out of memory errors**
- **Cause**: Batch size too large or images too big
- **Solution**:
  - Reduce batch size (try 16 or 8)
  - Use mixed precision training
  - Reduce image size

**Issue: Flask app won't start**
- **Cause**: Model files missing or port in use
- **Solution**:
  - Check models exist in `models/` directory
  - Change port in `app.run(port=5001)`
  - Check firewall settings

---

## 📚 Additional Resources

### Medical AI Ethics
- Always include medical disclaimers
- Never replace professional diagnosis
- Be transparent about AI limitations
- Ensure diverse training data

### Performance Optimization
- Use TensorFlow Lite for mobile deployment
- Implement model quantization for faster inference
- Use GPU acceleration for training
- Batch predictions for efficiency

### Clinical Validation
1. Partner with medical institutions
2. Conduct clinical trials
3. Compare with gold standard (biopsy)
4. Measure real-world impact

---

## 🎯 Next Steps

### Phase 1: Improve Model (Weeks 1-4)
- [ ] Expand dataset to 1000+ images per class
- [ ] Implement cross-validation
- [ ] Achieve 95%+ sensitivity
- [ ] Add explainability (Grad-CAM)

### Phase 2: Enhanced Features (Weeks 5-8)
- [ ] Multi-class classification (cancer types)
- [ ] Improved segmentation
- [ ] Real-time inference optimization
- [ ] Mobile app development

### Phase 3: Clinical Deployment (Weeks 9-12)
- [ ] Clinical validation study
- [ ] Regulatory compliance (FDA/CE)
- [ ] Integration with hospital systems
- [ ] User training and documentation

### Phase 4: Scale (Months 4-6)
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Load balancing and auto-scaling
- [ ] Continuous learning pipeline
- [ ] Global rollout

---

## 📞 Support

For questions or issues:
1. Check this README
2. Review code comments
3. Check logs in `logs/app.log`
4. Create an issue on GitHub

---

## ⚖️ Legal & Compliance

**IMPORTANT DISCLAIMERS:**

1. **Not Medical Advice**: This system provides AI predictions only and is NOT a substitute for professional medical diagnosis.

2. **Experimental**: This is a research/educational project. Not FDA approved or clinically validated.

3. **Liability**: Users assume all risks. Consult healthcare professionals for actual diagnosis.

4. **Data Privacy**: Comply with HIPAA, GDPR, and local regulations when handling medical data.

5. **Informed Consent**: Users must understand AI limitations before using the system.

---

## 📄 License

This project is for educational and research purposes only.

---

## 🙏 Acknowledgments

- Based on research in medical image analysis
- Uses pre-trained ImageNet weights
- Inspired by clinical need for early cancer detection

---

**Remember**: The goal is to ASSIST healthcare professionals, not replace them. Always prioritize patient safety and ethical AI use.
