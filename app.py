import requests
from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image
from keras.applications.vgg16 import VGG16
import numpy as np
import joblib
from keras.preprocessing import image

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image_url = request.json.get('url')
        if image_url:
            try:
                image_data = requests.get(image_url).content
                image = Image.open(BytesIO(image_data))
                result = predict_class(image)
                return jsonify({'prediction': result})
            except Exception as e:
                return jsonify({'error': str(e)})
        else:
            return jsonify({'error': 'No image URL provided'})
    return jsonify({'error': 'Invalid request'})

def predict_class(rgb_image):
    model = joblib.load('xray.pkl')
    vggmodel = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    for layer in vggmodel.layers:
        layer.trainable = False

    label = {0: 'PCOS', 1: 'NORMAL'}

    # Resize the RGB image to (256, 256)
    test_image = image.array_to_img(rgb_image, data_format='channels_last')
    test_image = test_image.resize((256, 256))
    
    # Convert the image to an array
    test_image = image.img_to_array(test_image)
    
    # Normalize the image
    test_image /= 255.0
    
    # Expand the dimensions to match the model input shape
    test_image = np.expand_dims(test_image, axis=0)
    
    # Use VGG16 for feature extraction
    feature_extractor = vggmodel.predict(test_image)
    
    # Reshape the features
    features = feature_extractor.reshape(feature_extractor.shape[0], -1)
    
    # Make predictions
    prediction = model.predict(features)[0]
    final = label[prediction]
    
    return final

if __name__ == '__main__':
    app.run(debug=True)
