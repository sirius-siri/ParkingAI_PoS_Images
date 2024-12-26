
from flask import Flask, render_template, request, jsonify
from paddleocr import PaddleOCR
from ultralytics import YOLO
import cv2
import numpy as np
import os
import shutil
import time

app = Flask(__name__)

# Initialize PaddleOCR with English as the language
ocr = PaddleOCR(use_angle_cls=True, lang='en')

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
print("Base directory:", BASE_DIR)

UPLOADS_DIR = os.path.join(BASE_DIR, 'app',"uploads")
print("Uploads directory:", UPLOADS_DIR)

# Check if uploads directory exists
if not os.path.exists(UPLOADS_DIR):
    # If not, create it
    os.makedirs(UPLOADS_DIR)
    print("Uploads directory created successfully.")
else:
    print("Uploads directory already exists.")

classification_model = YOLO("./classification.pt")
detection_model = YOLO("./detection.pt")
print("model loaded")

def get_number_plate_color(image_path):
    image = cv2.imread(image_path)
    
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the region of interest (ROI) covering the entire image
    roi = hsv
    
    # Calculate the average color of the ROI
    avg_color = np.mean(roi, axis=(0, 1))
    
    # Check if the average color tends more towards green or white
    green_threshold = 100  # Tune this threshold based on your images
    if avg_color[1] > green_threshold:
        return "Green"
    else:
        return "White"

def delete_existing_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

@app.route('/')
def index():
    print("upload page")
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    
    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    # Delete existing file if any
    existing_file_path = os.path.join(UPLOADS_DIR, file.filename)
    delete_existing_file(existing_file_path)

    # Save the uploaded file
    file_path = os.path.join(UPLOADS_DIR, file.filename)
    file.save(file_path)
    print("image saved: ",file_path)

    # Perform object classification using YOLO
    classifcation_results = classification_model(file_path)
    prob1=classifcation_results[0].probs
    class_mapping = {0: '1', 1: '3'}
    vehicle = prob1.top5[0]
    vehicle_label = class_mapping[vehicle]

    print("classification completed")

    # Perform object detection using YOLO
    results = detection_model(source=file_path, project='uploads', save_crop=True, exist_ok=True, name='pred')
    print("file path: ",os.path.basename(file_path))
    cropped_file_path = os.path.join('uploads', 'pred', 'crops', 'Number Plate', os.path.basename(file_path))
    print("cropped image dir: ",cropped_file_path)

    # get the colour of number plate 
    color = get_number_plate_color(cropped_file_path)
    if color == "Green":
        if vehicle_label == "1":
            vehicle_label = "6"
        elif vehicle_label == "3":
            vehicle_label = "7"
    img = cv2.imread(cropped_file_path)

    result = ocr.ocr(img, cls=True)
    # Extract text from image
    text_output = ""
    for line in result:
        for word in line:
            text_output += word[1][0]

    text_output = text_output.upper().replace("-", "").replace(" ", "").replace(".", "").replace(":","").replace("$","S")
    if text_output and text_output[0].isdigit():
        text_output = text_output[1:]
    text_output = text_output.replace("IND", "").replace("INO", "")
    text_output = text_output[:2] + text_output[2:4].replace("O", "0") + text_output[4:]
    text_output = text_output[:1] + text_output[1:2].replace("H","N") + text_output[2:]
    text_output = text_output[:4] + text_output[4:6].replace("8", "B") + text_output[6:]
    text_output = text_output[:4] + text_output[4:6].replace("0", "D") + text_output[6:]

    while text_output and text_output[-1].isalpha():
        text_output = text_output[:-1]
   
    text_output = text_output[:10]

    # Delete the cropped image
    delete_existing_file(cropped_file_path)
    
    # return render_template('index.html', classification_output=vehicle_label, text_output=text_output)
    return jsonify({"VehicleType" : int(vehicle_label), "VehicleNo": text_output})

def zip_uploads():
    current_time = time.strftime("%Y%m%d%H%M%S")
    zip_filename = os.path.join(BASE_DIR, f'uploads_{current_time}.zip')
    shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', UPLOADS_DIR)

if __name__ == '__main__':
    app.run(debug=False)
    while True:
        zip_uploads()
        one_week_in_seconds = 7 * 24 * 60 * 60
        time.sleep(one_week_in_seconds)




