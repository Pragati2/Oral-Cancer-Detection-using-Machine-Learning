"""
Enhanced Flask Application for Oral Cancer Detection
Includes security, error handling, logging, and explainability
"""

from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import cv2
import os
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import hashlib
import secrets

# ============================================================================
# APPLICATION SETUP
# ============================================================================

app = Flask(__name__)

# Configuration
class Config:
    """Application configuration"""
    UPLOAD_FOLDER = 'uploads/'
    RESULTS_FOLDER = 'results/'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    SECRET_KEY = secrets.token_hex(32)
    MODEL_PATH = 'models/best_model.h5'
    SEGMENTATION_MODEL_PATH = 'models/unet_model.h5'
    CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for predictions

app.config.from_object(Config)

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure application logging"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10240000,  # 10MB
        backupCount=10
    )
    
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Oral Cancer Detection Application Startup')

setup_logging()

# ============================================================================
# MODEL LOADING
# ============================================================================

class ModelManager:
    """Manages model loading and predictions"""
    
    def __init__(self):
        self.classification_model = None
        self.segmentation_model = None
        self.load_models()
    
    def load_models(self):
        """Load TensorFlow models"""
        try:
            if os.path.exists(app.config['MODEL_PATH']):
                self.classification_model = tf.keras.models.load_model(
                    app.config['MODEL_PATH']
                )
                app.logger.info("Classification model loaded successfully")
            else:
                app.logger.warning(f"Classification model not found at {app.config['MODEL_PATH']}")
            
            if os.path.exists(app.config['SEGMENTATION_MODEL_PATH']):
                self.segmentation_model = tf.keras.models.load_model(
                    app.config['SEGMENTATION_MODEL_PATH']
                )
                app.logger.info("Segmentation model loaded successfully")
            else:
                app.logger.warning(f"Segmentation model not found at {app.config['SEGMENTATION_MODEL_PATH']}")
        
        except Exception as e:
            app.logger.error(f"Error loading models: {str(e)}")
    
    def is_ready(self):
        """Check if classification model is loaded"""
        return self.classification_model is not None

model_manager = ModelManager()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def get_file_hash(file_path):
    """Calculate SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Advanced image preprocessing with CLAHE enhancement
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # CLAHE enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Resize and normalize
        img = cv2.resize(img, target_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img
    
    except Exception as e:
        app.logger.error(f"Preprocessing error: {str(e)}")
        return None


def preprocess_for_segmentation(image_path, target_size=(256, 256)):
    """Preprocess image for segmentation"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img
    
    except Exception as e:
        app.logger.error(f"Segmentation preprocessing error: {str(e)}")
        return None


def generate_gradcam(img_array, model, last_conv_layer_name='block5_conv3'):
    """Generate Grad-CAM visualization"""
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    except Exception as e:
        app.logger.error(f"Grad-CAM generation error: {str(e)}")
        return None


def overlay_heatmap(original_path, heatmap, output_path):
    """Overlay Grad-CAM heatmap on original image"""
    try:
        img = cv2.imread(original_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        superimposed = heatmap_colored * 0.4 + img * 0.6
        superimposed = np.uint8(superimposed)
        
        cv2.imwrite(output_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
        return True
    
    except Exception as e:
        app.logger.error(f"Heatmap overlay error: {str(e)}")
        return False


def get_recommendation(prediction, confidence):
    """Generate medical recommendation based on prediction"""
    if prediction == 'Cancerous':
        if confidence > 0.9:
            return {
                'level': 'HIGH',
                'message': 'High confidence detection of potential oral cancer. Please consult an oncologist immediately for professional diagnosis.',
                'urgency': 'URGENT'
            }
        elif confidence > 0.7:
            return {
                'level': 'MODERATE',
                'message': 'Moderate confidence detection. Please consult a healthcare professional as soon as possible for further evaluation.',
                'urgency': 'IMPORTANT'
            }
        else:
            return {
                'level': 'LOW',
                'message': 'Low confidence detection. Consider getting a professional examination for peace of mind.',
                'urgency': 'RECOMMENDED'
            }
    else:
        if confidence > 0.9:
            return {
                'level': 'NORMAL',
                'message': 'No signs of oral cancer detected with high confidence. Continue regular dental check-ups.',
                'urgency': 'ROUTINE'
            }
        else:
            return {
                'level': 'UNCLEAR',
                'message': 'No clear signs detected, but confidence is moderate. Consider professional examination to be certain.',
                'urgency': 'RECOMMENDED'
            }


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    prediction_id = secrets.token_hex(8)
    
    try:
        # Validate request
        if 'file' not in request.files:
            app.logger.warning(f"[{prediction_id}] No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            app.logger.warning(f"[{prediction_id}] Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            app.logger.warning(f"[{prediction_id}] Invalid file type: {file.filename}")
            return jsonify({
                'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'
            }), 400
        
        # Check if model is ready
        if not model_manager.is_ready():
            app.logger.error(f"[{prediction_id}] Model not available")
            return jsonify({'error': 'Model not available. Please try again later.'}), 503
        
        # Save file securely
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{prediction_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        # Calculate file hash for integrity
        file_hash = get_file_hash(filepath)
        app.logger.info(f"[{prediction_id}] File saved: {unique_filename} (hash: {file_hash[:16]})")
        
        # Preprocess and predict
        processed_img = preprocess_image(filepath)
        if processed_img is None:
            app.logger.error(f"[{prediction_id}] Image preprocessing failed")
            return jsonify({'error': 'Failed to process image. Please ensure it is a valid image file.'}), 400
        
        # Classification
        prediction = model_manager.classification_model.predict(processed_img, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        
        class_names = ['Non-Cancerous', 'Cancerous']
        result = class_names[class_idx]
        
        app.logger.info(f"[{prediction_id}] Prediction: {result}, Confidence: {confidence:.4f}")
        
        # Check confidence threshold
        if confidence < app.config['CONFIDENCE_THRESHOLD']:
            app.logger.warning(f"[{prediction_id}] Low confidence prediction: {confidence:.4f}")
        
        # Generate Grad-CAM visualization
        gradcam_path = None
        if confidence > app.config['CONFIDENCE_THRESHOLD']:
            try:
                heatmap = generate_gradcam(processed_img, model_manager.classification_model)
                if heatmap is not None:
                    gradcam_filename = f"gradcam_{unique_filename}"
                    gradcam_path = os.path.join(app.config['RESULTS_FOLDER'], gradcam_filename)
                    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
                    
                    if overlay_heatmap(filepath, heatmap, gradcam_path):
                        app.logger.info(f"[{prediction_id}] Grad-CAM generated successfully")
            except Exception as e:
                app.logger.error(f"[{prediction_id}] Grad-CAM generation failed: {str(e)}")
        
        # Segmentation (if cancerous and model available)
        segmentation_path = None
        if class_idx == 1 and model_manager.segmentation_model is not None:
            try:
                seg_img = preprocess_for_segmentation(filepath)
                if seg_img is not None:
                    seg_mask = model_manager.segmentation_model.predict(seg_img, verbose=0)
                    seg_mask = (seg_mask[0] > 0.5).astype(np.uint8) * 255
                    
                    seg_filename = f"seg_{unique_filename}"
                    segmentation_path = os.path.join(app.config['RESULTS_FOLDER'], seg_filename)
                    cv2.imwrite(segmentation_path, seg_mask)
                    app.logger.info(f"[{prediction_id}] Segmentation completed")
            except Exception as e:
                app.logger.error(f"[{prediction_id}] Segmentation failed: {str(e)}")
        
        # Get recommendation
        recommendation = get_recommendation(result, confidence)
        
        # Prepare response
        response = {
            'prediction_id': prediction_id,
            'prediction': result,
            'confidence': round(confidence, 4),
            'probabilities': {
                'Non-Cancerous': round(float(prediction[0][0]), 4),
                'Cancerous': round(float(prediction[0][1]), 4)
            },
            'recommendation': recommendation,
            'timestamp': timestamp,
            'disclaimer': 'This is an AI prediction and should NOT replace professional medical diagnosis. Please consult a healthcare professional for accurate diagnosis.'
        }
        
        if gradcam_path:
            response['gradcam_available'] = True
        
        if segmentation_path:
            response['segmentation_available'] = True
        
        # Log successful prediction
        app.logger.info(
            f"[{prediction_id}] Prediction completed - "
            f"Result: {result}, Confidence: {confidence:.4f}, "
            f"Recommendation Level: {recommendation['level']}"
        )
        
        return jsonify(response), 200
    
    except Exception as e:
        app.logger.error(f"[{prediction_id}] Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'Prediction failed. Please try again or contact support.',
            'prediction_id': prediction_id
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if model_manager.is_ready() else 'degraded',
        'classification_model': model_manager.classification_model is not None,
        'segmentation_model': model_manager.segmentation_model is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0'
    }
    
    return jsonify(status), 200 if model_manager.is_ready() else 503


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get application statistics"""
    try:
        upload_count = len([f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                           if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))])
        
        return jsonify({
            'total_predictions': upload_count,
            'uptime': 'Available',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        app.logger.error(f"Stats error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve statistics'}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(400)
def bad_request(error):
    """Handle bad request"""
    return jsonify({'error': 'Bad request'}), 400


@app.errorhandler(404)
def not_found(error):
    """Handle not found"""
    return jsonify({'error': 'Resource not found'}), 404


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large"""
    app.logger.warning("File upload too large")
    return jsonify({
        'error': f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'
    }), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server error"""
    app.logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error. Please try again later.'}), 500


@app.errorhandler(503)
def service_unavailable(error):
    """Handle service unavailable"""
    app.logger.error("Service unavailable")
    return jsonify({'error': 'Service temporarily unavailable'}), 503


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run application
    app.logger.info("Starting Flask application...")
    app.run(
        debug=False,  # Set to False in production
        host='0.0.0.0',
        port=5000,
        threaded=True
    )
