from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import base64
from collections import defaultdict

print("Initializing Flask app...")

# Initialize the Flask application
app = Flask(__name__)

print("Loading the YOLO model...")

# Load the YOLO model
model = YOLO('model\pizza_yolov8.pt')

print("YOLO model loaded successfully.")

@app.route('/predict', methods=['POST'])
def predict():
    print("Received a request.")
    data = request.get_json()
    if 'image' not in data:
        print("No image provided in the request.")
        return jsonify({'error': 'No image provided'}), 400

    image_base64 = data['image']
    image_data = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(image_data))

    # Perform prediction
    print("Performing prediction...")
    results = model(img)

    # Count occurrences of each class item
    class_count = defaultdict(int)

    # Prepare the response
    predictions = []
    for result in results:
        for box in result.boxes:
            class_name = model.names[int(box.cls)]
            class_count[class_name] += 1
            predictions.append({
                'class_name': class_name,
                'confidence': box.conf.item(),
            })

    print("Returning predictions.")
    return jsonify({'predictions': predictions, 'class_count': dict(class_count)})

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)
