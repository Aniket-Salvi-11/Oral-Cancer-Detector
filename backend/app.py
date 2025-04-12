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
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        file_path = os.path.join('static', 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        prediction, confidence = model_predict(file_path)

        # Determine diagnosis
        if prediction > 0.5:  # Assuming threshold of 0.5 for cancer detection
            diagnosis = "Oral Cancer Detected"
            risk = "High Risk"
        else:
            diagnosis = "No Oral Cancer Detected"
            risk = "Low Risk"

        # Convert confidence to percentage
        confidence_percent = confidence * 100

        return render_template('result.html',
                           diagnosis=diagnosis,
                           risk=risk,
                           confidence=f"{confidence_percent:.2f}%",
                           prediction=prediction,
                           user_image=file_path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)