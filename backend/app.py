import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing import image
# Replace this line:
# from tensorflow.keras.preprocessing import image
# With these:
from PIL import Image
import io
import numpy as np
import io

app = Flask(__name__)
CORS(app)

# Model loading with error handling
try:
    model = load_model('ml_models/oral_cancer_classification_modelV3.h5')
    print("✅ Model loaded successfully!")
    MODEL_READY = True
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    MODEL_READY = False

from PIL import Image  # Add this at the top with other imports

def preprocess_image(img_bytes):
    """Preprocess image according to model requirements"""
    try:
        # Load image from bytes
        img = Image.open(io.BytesIO(img_bytes))
        
        # Convert to RGB if has alpha channel (4 channels)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
            
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Ensure 3 channels
        if img_array.shape[-1] > 3:
            img_array = img_array[..., :3]
            
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        return img_array
    except Exception as e:
        print(f"Image preprocessing failed: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_READY:
        return jsonify({'error': 'Model not loaded'}), 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    try:
        file = request.files['file'].read()
        
        # Preprocess and predict
        processed_img = preprocess_image(file)
        prediction = model.predict(processed_img)
        
        # Interpret prediction (adjust based on your model)
        result = {
            'prediction': float(prediction[0][0]),  # Raw prediction value
            'diagnosis': 'High Risk' if prediction[0][0] > 0.5 else 'Low Risk',
            'confidence': round(float(prediction[0][0]) * 100, 2)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)