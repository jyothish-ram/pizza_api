import requests
import base64

# Load an image and convert it to base64
with open("test/img3.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

# Prepare the request payload
payload = {
    "image": base64_image
}

# Send the request to the Flask app
response = requests.post("http://127.0.0.1:5000/predict", json=payload)

# Print the response
print(response.text)
