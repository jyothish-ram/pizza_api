from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import base64
import numpy as np
import cv2
from collections import defaultdict

print("Initializing Flask app...")

# Initialize the Flask application
app = Flask(__name__)

print("Loading the YOLO model...")

# Load the YOLO model
model = YOLO('model/pizza_yolov8_v1.pt')  # Fixed the path separator for consistency

print("YOLO model loaded successfully.")

@app.route('/predict', methods=['POST'])
def predict():
    print("Received a request.")
    data = request.get_json()
    if not data or 'image' not in data:
        print("No image provided in the request.")
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_base64 = data['image']
        image_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
    except Exception as e:
        print(f"Error decoding image: {e}")
        return jsonify({'error': 'Invalid image data'}), 400

    # Convert PIL image to OpenCV format
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Perform prediction and draw bounding boxes
    print("Performing prediction...")
    results = model(img_cv, conf=0.25)

    # Count occurrences of each class item
    class_count = defaultdict(int)
    predictions = []

    # Draw bounding boxes on the image
    for result in results:
        for box in result.boxes:
            class_name = model.names[int(box.cls)]
            class_count[class_name] += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf.item()
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_cv, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            predictions.append({
                'class_name': class_name,
                'confidence': confidence,
            })

    # Additional image processing for circularity and output image
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 40, 150)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(img_cv, [largest_contour], -1, (0, 255, 0), 1)
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)
    diameter = 2 * radius
    circle_area = np.pi * (radius ** 2)
    mask = np.zeros_like(gray)
    cv2.circle(mask, center, radius, 255, -1)
    pizza_area = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=cv2.drawContours(np.zeros_like(gray), [largest_contour], -1, 255, -1)))
    circularity = pizza_area / circle_area * 100
    cv2.circle(img_cv, center, radius, (255, 0, 0), 2)

    # Convert the processed output image back to base64
    _, buffer = cv2.imencode('.jpg', img_cv)
    output_image_base64 = base64.b64encode(buffer).decode('utf-8')

    print("Returning predictions and processed image.")
    return jsonify({        
        'circularity': circularity,
        'class_count': dict(class_count),
        'predictions': predictions,
        'processed_image': output_image_base64
    })

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(host='0.0.0.0')
