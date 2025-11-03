# Wound First Aid ML System

An AI-powered wound classification system that analyzes wound images and provides appropriate first aid recommendations.

## 🎯 Features

- **4 Wound Types**: Classifies abrasions, lacerations, burns, and puncture wounds
- **First Aid Guidance**: Provides step-by-step first aid instructions
- **High Accuracy**: Uses transfer learning with EfficientNetB0
- **REST API**: Flask-based API for easy integration
- **Multiple Deployment Options**: Docker, Kubernetes, mobile (TFLite), web (TF.js)
- **Comprehensive Evaluation**: Detailed metrics and visualizations

## 📁 Project Structure

```
wound-classifier-project/
├── README.md
├── requirements.txt
├── .gitignore
├── Dockerfile
├── docker-compose.yml
│
├── models/
│   └── wound_classifier_final.h5
│
├── wound_dataset/
│   ├── train/
│   │   ├── abrasion/
│   │   ├── laceration/
│   │   ├── burn/
│   │   └── puncture/
│   └── test/
│       ├── abrasion/
│       ├── laceration/
│       ├── burn/
│       └── puncture/
│
├── train_wound_model.py
├── fetch_dataset.py
├── predict.py
├── flask_api.py
├── evaluate_model.py
└── export_model.py
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd wound-classifier-project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
# Create directory structure and get dataset info
python fetch_dataset.py

# After downloading images, organize them:
# - Place training images in wound_dataset/train/<category>/
# - Place test images in wound_dataset/test/<category>/
# Recommended: 500-1000 images per category
```

### 3. Train Model

```bash
python train_wound_model.py
```

This will:

- Train the model using transfer learning
- Apply data augmentation
- Save the best model as `wound_classifier_final.h5`
- Training time: 2-4 hours with GPU, 10-15 hours with CPU

### 4. Evaluate Model

```bash
python evaluate_model.py
```

Generates:

- Confusion matrix
- Per-class metrics
- Confidence distribution plots
- `evaluation_results.json`

### 5. Run API Server

```bash
python flask_api.py
```

The API will be available at `http://localhost:5000`

### 6. Make Predictions

Using Python:

```python
from predict import WoundClassifier

classifier = WoundClassifier('wound_classifier_final.h5')
result = classifier.predict(
    'wound_image.jpg',
    description='minor cut from kitchen knife'
)
print(result)
```

Using curl:

```bash
curl -X POST -F "image=@wound.jpg" -F "description=minor cut" \
     http://localhost:5000/api/predict
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build Docker image
docker build -t wound-classifier .

# Run container
docker run -p 5000:5000 -v $(pwd)/models:/app/models wound-classifier

# Or use docker-compose
docker-compose up -d
```

## 📊 API Endpoints

### GET /

API documentation and information

### GET /api/health

Health check endpoint

### GET /api/classes

Get list of wound classes

### POST /api/predict

Predict wound type and get first aid

**Parameters:**

- `image`: Image file (required)
- `description`: Text description (optional)

**Response:**

```json
{
  "wound_type": "laceration",
  "wound_name": "Laceration (Cut)",
  "confidence": 92.5,
  "description": "A deep cut or tear in the skin",
  "first_aid_steps": ["Apply direct pressure to stop bleeding", "..."],
  "severity": "Moderate to Severe",
  "seek_medical_help": "If bleeding persists..."
}
```

## 📱 Mobile & Web Deployment

### Convert to TensorFlow Lite (Mobile)

```bash
python export_model.py
```

This generates:

- `wound_classifier.tflite` - Mobile model
- `tflite_inference_example.py` - Usage example

### Convert to TensorFlow.js (Web)

```bash
python export_model.py
```

This generates:

- `tfjs_model/` - Web model
- `tfjs_inference_example.html` - Web demo

## 📈 Model Performance

Expected performance metrics:

- Overall Accuracy: 85-95%
- Precision: 0.85-0.95 per class
- Recall: 0.85-0.95 per class
- F1-Score: 0.85-0.95 per class

## 🔧 Configuration

### Model Parameters

Edit `train_wound_model.py`:

```python
IMG_SIZE = 224        # Input image size
BATCH_SIZE = 32       # Batch size for training
EPOCHS = 50           # Number of training epochs
NUM_CLASSES = 4       # Number of wound types
```

### API Configuration

Edit `flask_api.py`:

```python
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
```

## 📚 Dataset Sources

### Recommended Sources:

1. **Kaggle Datasets**

   - Search for: "wound images", "burn classification"
   - Requires Kaggle account

2. **DermNet NZ**

   - URL: https://dermnetnz.org
   - Educational use with attribution

3. **NIH Medical Images**

   - URL: https://openi.nlm.nih.gov
   - Public domain medical images

4. **MedPix**
   - URL: https://medpix.nlm.nih.gov
   - Free for educational purposes

### Dataset Requirements:

- Minimum: 500 images per category
- Recommended: 1000+ images per category
- Format: JPG, PNG
- Size: At least 224x224 pixels

## ⚠️ Important Notes

### Medical Disclaimer

- This is an AI-based suggestion tool
- NOT a replacement for professional medical advice
- Always seek professional help for serious injuries
- Use for educational and reference purposes only

### Legal & Ethical Considerations

- Comply with medical device regulations (FDA, CE marking)
- Implement data privacy measures (HIPAA if in US)
- Include proper disclaimers
- Ensure proper licensing for medical images
- Regular model updates with new data
- Human oversight recommended for all predictions

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

## 🔐 Security

- Input validation for all uploads
- File size limits enforced
- Allowed file types restricted
- Secure filename handling
- CORS configured for API

## 📝 License

This project is for educational purposes. Ensure you have proper licenses for:

- Medical image datasets
- Commercial deployment
- Healthcare applications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For issues or questions:

- Open an issue on GitHub
- Check documentation
- Review API examples

## 🔄 Updates

### Version 1.0.0

- Initial release
- 4 wound types supported
- Transfer learning with EfficientNetB0
- REST API with Flask
- Docker deployment support
- Mobile (TFLite) and Web (TF.js) export

## 📖 Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Docker Documentation](https://docs.docker.com/)
- [First Aid Guidelines](https://www.redcross.org/take-a-class/first-aid)

---

**Built with ❤️ for improving first aid response through AI**
