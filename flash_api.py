from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from predict import WoundClassifier
import os
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
try:
    classifier = WoundClassifier('wound_classifier_final.h5')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    classifier = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """Home page with API documentation"""
    return jsonify({
        'message': 'Wound First Aid ML API',
        'version': '1.0.0',
        'endpoints': {
            '/api/predict': 'POST - Predict wound type and get first aid',
            '/api/health': 'GET - Check API health status',
            '/api/classes': 'GET - Get list of wound classes',
            '/api/batch-predict': 'POST - Batch prediction for multiple images'
        },
        'usage': {
            'predict': {
                'method': 'POST',
                'content_type': 'multipart/form-data',
                'parameters': {
                    'image': 'Image file (required)',
                    'description': 'Text description (optional)'
                }
            }
        }
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if classifier is None:
        return jsonify({
            'status': 'unhealthy',
            'error': 'Model not loaded'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': True,
        'upload_folder': UPLOAD_FOLDER
    })

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get list of wound classes the model can predict"""
    if classifier is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    classes_info = []
    for wound_class in classifier.classes:
        info = classifier.first_aid_guide[wound_class]
        classes_info.append({
            'class': wound_class,
            'name': info['name'],
            'description': info['description'],
            'severity': info['severity']
        })
    
    return jsonify({
        'classes': classes_info,
        'total': len(classes_info)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict wound type and provide first aid instructions
    
    Request:
        - image: Image file (multipart/form-data)
        - description: Optional text description
        
    Response:
        - wound_type: Predicted wound class
        - confidence: Prediction confidence (0-100)
        - first_aid_steps: List of first aid instructions
        - severity: Wound severity level
        - seek_medical_help: When to seek professional help
    """
    if classifier is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    # Check if image is in request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    description = request.form.get('description', '')
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Processing image: {filename}")
        result = classifier.predict(filepath, description)
        result['metadata'] = {
            'filename': filename,
            'timestamp': timestamp,
            'description_provided': bool(description)
        }
        try:
            os.remove(filepath)
        except:
            pass
        logger.info(f"Prediction completed: {result['wound_type']} ({result['confidence']:.1f}%)")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple images
    
    Request:
        - images: Multiple image files
        
    Response:
        - results: List of prediction results
    """
    if classifier is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    files = request.files.getlist('images')
    
    if not files or len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400
    
    if len(files) > 10:
        return jsonify({'error': 'Maximum 10 images allowed per request'}), 400
    
    results = []
    
    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            results.append({
                'filename': file.filename,
                'error': 'Invalid file'
            })
            continue
        
        try:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = classifier.predict(filepath)
            result['filename'] = file.filename
            results.append(result)
            os.remove(filepath)
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    return jsonify({
        'total': len(results),
        'results': results
    })
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large',
        'max_size': '16MB'
    }), 413

@app.errorhandler(404)
def not_found(error):
    """Handle not found error"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'Please check the API documentation at /'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server error"""
    logger.error(f"Internal error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'Please try again later'
    }), 500

if __name__ == '__main__':
    print("=" * 70)
    print("WOUND FIRST AID API SERVER")
    print("=" * 70)
    print(f"Starting server...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Max file size: {MAX_CONTENT_LENGTH / (1024*1024)}MB")
    print(f"Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}")
    print("=" * 70)
    print("\nEndpoints:")
    print("  - GET  /           : API documentation")
    print("  - GET  /api/health : Health check")
    print("  - GET  /api/classes: Get wound classes")
    print("  - POST /api/predict: Predict wound type")
    print("  - POST /api/batch-predict: Batch prediction")
    print("=" * 70)
    print("\nExample usage:")
    print("  curl -X POST -F 'image=@wound.jpg' -F 'description=minor cut' \\")
    print("       http://localhost:5000/api/predict")
    print("=" * 70)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
