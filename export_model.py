# export_model.py
import tensorflow as tf
from tensorflow import keras
import os

def convert_to_tflite(model_path, output_path='wound_classifier.tflite'):
    """
    Convert Keras model to TensorFlow Lite for mobile deployment
    
    Args:
        model_path: Path to trained Keras model
        output_path: Path to save TFLite model
    """
    print("="*70)
    print("CONVERTING MODEL TO TENSORFLOW LITE")
    print("="*70)
    
    # Load the model
    print(f"\nLoading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully")
    
    # Convert to TFLite
    print("\nConverting to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Enable quantization for smaller size
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get file sizes
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    tflite_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\n✓ TFLite model saved to: {output_path}")
    print(f"\nModel Size Comparison:")
    print(f"  Original Keras model: {original_size:.2f} MB")
    print(f"  TFLite model: {tflite_size:.2f} MB")
    print(f"  Size reduction: {(1 - tflite_size/original_size)*100:.1f}%")
    
    print("\n" + "="*70)
    print("TFLITE CONVERSION COMPLETE")
    print("="*70)
    
    return output_path

def convert_to_tfjs(model_path, output_dir='tfjs_model'):
    """
    Convert Keras model to TensorFlow.js for web deployment
    
    Args:
        model_path: Path to trained Keras model
        output_dir: Directory to save TF.js model
    """
    print("\n" + "="*70)
    print("CONVERTING MODEL TO TENSORFLOW.JS")
    print("="*70)
    
    try:
        import tensorflowjs as tfjs
    except ImportError:
        print("\n❌ tensorflowjs not installed")
        print("Install it with: pip install tensorflowjs")
        return None
    
    # Load the model
    print(f"\nLoading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully")
    
    # Convert to TF.js
    print(f"\nConverting to TensorFlow.js format...")
    os.makedirs(output_dir, exist_ok=True)
    
    tfjs.converters.save_keras_model(model, output_dir)
    
    print(f"\n✓ TF.js model saved to: {output_dir}")
    print(f"\nFiles created:")
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        size = os.path.getsize(file_path) / 1024
        print(f"  - {file} ({size:.2f} KB)")
    
    print("\n" + "="*70)
    print("TFJS CONVERSION COMPLETE")
    print("="*70)
    
    return output_dir

def create_tflite_inference_example(tflite_path='wound_classifier.tflite'):
    """
    Create example code for using TFLite model
    
    Args:
        tflite_path: Path to TFLite model
    """
    example_code = f'''
# Example: Using TFLite Model for Inference

import numpy as np
from PIL import Image
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="{tflite_path}")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Make prediction
image_path = 'wound_image.jpg'
input_data = preprocess_image(image_path)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data[0])

classes = ['abrasion', 'laceration', 'burn', 'puncture']
confidence = output_data[0][predicted_class]

print(f"Predicted: {{classes[predicted_class]}} ({{confidence*100:.1f}}% confidence)")
'''
    
    with open('tflite_inference_example.py', 'w') as f:
        f.write(example_code)
    
    print("\n✓ Created: tflite_inference_example.py")

def create_tfjs_inference_example():
    """Create example HTML/JS code for using TF.js model"""
    
    html_code = '''<!DOCTYPE html>
<html>
<head>
    <title>Wound Classifier - TensorFlow.js</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        #imagePreview { max-width: 400px; margin: 20px 0; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
        #result { margin-top: 20px; padding: 20px; background: #f0f0f0; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Wound Classifier</h1>
    <input type="file" id="imageInput" accept="image/*">
    <br><br>
    <img id="imagePreview" style="display:none;">
    <br>
    <button onclick="predict()" id="predictBtn" disabled>Analyze Wound</button>
    <div id="result"></div>

    <script>
        let model;
        const classes = ['abrasion', 'laceration', 'burn', 'puncture'];

        // Load model
        async function loadModel() {
            model = await tf.loadLayersModel('tfjs_model/model.json');
            console.log('Model loaded');
        }

        // Handle image upload
        document.getElementById('imageInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    const img = document.getElementById('imagePreview');
                    img.src = event.target.result;
                    img.style.display = 'block';
                    document.getElementById('predictBtn').disabled = false;
                }
                reader.readAsDataURL(file);
            }
        });

        // Preprocess image
        function preprocessImage(img) {
            return tf.tidy(() => {
                const tensor = tf.browser.fromPixels(img)
                    .resizeNearestNeighbor([224, 224])
                    .toFloat()
                    .div(255.0)
                    .expandDims();
                return tensor;
            });
        }

        // Make prediction
        async function predict() {
            const img = document.getElementById('imagePreview');
            const tensor = preprocessImage(img);
            
            const predictions = await model.predict(tensor).data();
            const topPrediction = Array.from(predictions)
                .map((p, i) => ({ class: classes[i], probability: p }))
                .sort((a, b) => b.probability - a.probability);

            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = `
                <h2>Prediction Results</h2>
                <p><strong>${topPrediction[0].class}</strong></p>
                <p>Confidence: ${(topPrediction[0].probability * 100).toFixed(1)}%</p>
                <h3>All Predictions:</h3>
                <ul>
                    ${topPrediction.map(p => 
                        `<li>${p.class}: ${(p.probability * 100).toFixed(1)}%</li>`
                    ).join('')}
                </ul>
            `;

            tensor.dispose();
        }

        // Load model on page load
        loadModel();
    </script>
</body>
</html>'''
    
    with open('tfjs_inference_example.html', 'w') as f:
        f.write(html_code)
    
    print("✓ Created: tfjs_inference_example.html")

def export_saved_model_format(model_path, output_dir='saved_model'):
    """
    Export model in SavedModel format for TensorFlow Serving
    
    Args:
        model_path: Path to trained Keras model
        output_dir: Directory to save SavedModel
    """
    print("\n" + "="*70)
    print("EXPORTING TO SAVEDMODEL FORMAT")
    print("="*70)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully")
    
    # Save in SavedModel format
    print(f"\nExporting to SavedModel format...")
    tf.saved_model.save(model, output_dir)
    
    print(f"\n✓ SavedModel exported to: {output_dir}")
    print("\nThis format can be used with:")
    print("  - TensorFlow Serving for production deployment")
    print("  - TensorFlow Lite conversion")
    print("  - TensorFlow.js conversion")
    
    print("\n" + "="*70)
    print("SAVEDMODEL EXPORT COMPLETE")
    print("="*70)

if __name__ == "__main__":
    import sys
    
    model_path = 'wound_classifier_final.h5'
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please train the model first or provide correct path")
        sys.exit(1)
    
    print("="*70)
    print("MODEL EXPORT UTILITY")
    print("="*70)
    print(f"\nModel: {model_path}\n")
    
    # 1. Convert to TFLite
    tflite_path = convert_to_tflite(model_path)
    create_tflite_inference_example(tflite_path)
    
    # 2. Convert to TensorFlow.js
    tfjs_dir = convert_to_tfjs(model_path)
    if tfjs_dir:
        create_tfjs_inference_example()
    
    # 3. Export SavedModel format
    export_saved_model_format(model_path)
    
    print("\n" + "="*70)
    print("ALL EXPORTS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - wound_classifier.tflite (for mobile apps)")
    print("  - tfjs_model/ (for web deployment)")
    print("  - saved_model/ (for TensorFlow Serving)")
    print("  - tflite_inference_example.py")
    print("  - tfjs_inference_example.html")
    print("="*70)
