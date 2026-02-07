Chicken Weight Estimation Using Computer Vision & Machine Learning
1. Project Overview

This project aims to predict the weight of a chicken using images by combining object detection, deep feature extraction, and machine learning regression.
Instead of using physical weighing scales, the system estimates chicken weight automatically from images.

The project follows a modular pipeline, making it easy to improve or extend in the future.

2. Problem Statement

Manual weighing of chickens:

Is time-consuming

Requires physical handling (stress to birds)

Is not scalable for large poultry farms

This project proposes an image-based weight prediction system using:

YOLO for detecting chickens

CNN for extracting visual features

MLP regression for predicting weight in grams

3. Complete Workflow of the Project
Step 1: Dataset Preparation

Images of chickens were collected across multiple growth days.

Each image has a corresponding actual weight (in grams) stored in weights.csv.

The dataset is split into:

train

valid

test

📁 Key files

dataset/
 ├── train/images
 ├── valid/images
 ├── test/images
 └── data.yaml

Step 2: Chicken Detection (YOLO)

A YOLOv8 model (yolov8n.pt) is used to detect chickens in images.

For each image:

Bounding boxes are generated automatically.

Each detected chicken is cropped.

📄 Script used:

detect_and_crop.py


📁 Output:

dataset/crops/
 ├── crop_00001.jpg
 ├── crop_00002.jpg
 └── crops_meta.csv


crops_meta.csv stores:

crop filename

original image name

Step 3: Feature Extraction

Each cropped chicken image is passed through a pretrained CNN (ResNet18).

The final fully-connected layer is removed.

The output feature vector represents the chicken’s visual appearance.

📄 Script used:

extract_features.py


📁 Output:

dataset/features/features.npy

Step 4: Weight Mapping

The crop metadata is linked with actual weights using:

weights.csv


This mapping ensures:

Each feature vector corresponds to the correct ground-truth weight.

Step 5: Model Training (MLP Regression)

A Multi-Layer Perceptron (MLP) is trained using:

Input: feature vectors

Output: chicken weight (grams)

Feature normalization (mean & std) is applied.

Best model is saved based on validation loss.

📄 Script:

train_mlp.py


📁 Output:

models/
 ├── mlp.pth
 └── scaler.json

Step 6: Offline Inference & Evaluation

The trained MLP model predicts weight for all crops.

Predictions are saved for analysis.

📄 Script:

inference.py


📁 Output:

models/predictions.csv


This stage produced accurate predictions and serves as the reference logic for deployment.

Step 7: Comparison & Error Analysis

Actual vs predicted weights are compared.

Errors (absolute, percentage) can be calculated.

📄 Script:

compare_predictions.py

Step 8: Server Deployment (API)

A FastAPI server exposes a /predict endpoint.

User uploads a chicken image.

The server:

Detects the chicken

Crops the detected region

Extracts features

Predicts weight using the trained MLP

Returns weight in grams

📄 Script:

app.py


📌 Endpoint:

POST http://127.0.0.1:8000/predict

4. Folder Structure Explained
chicken_weight/
 ├── dataset/
 │   ├── train/
 │   ├── valid/
 │   ├── test/
 │   ├── crops/
 │   └── features/
 ├── models/
 │   ├── mlp.pth
 │   └── scaler.json
 ├── runs/
 ├── weights.csv
 ├── detect_and_crop.py
 ├── extract_features.py
 ├── train_mlp.py
 ├── inference.py
 ├── compare_predictions.py
 └── app.py

5. Instructions for a New Person to Continue This Project
Step 1: Environment Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Step 2: Dataset Preparation

Place images inside dataset/train/images

Update weights.csv with correct filenames & weights

Update data.yaml paths if required

Step 3: Run Detection & Cropping
python detect_and_crop.py

Step 4: Extract Features
python extract_features.py

Step 5: Train the MLP Model
python train_mlp.py \
 --features dataset/features/features.npy \
 --meta dataset/crops/crops_meta.csv \
 --weights weights.csv \
 --out models

Step 6: Generate Predictions
python inference.py --predict_all

Step 7: Start the Server
uvicorn app:app --host 0.0.0.0 --port 8000

Step 8: Test API
curl -X POST http://127.0.0.1:8000/predict \
 -F "file=@image.jpg"

6. Important Notes for Developers

Always follow the same preprocessing as inference.py

Do NOT skip normalization

Input image should:

Contain one chicken clearly visible

Be similar to training images

Weight output is always in grams

7. Future Improvements

Support multiple chickens per image

Add confidence score

Improve YOLO fine-tuning

Mobile app integration

Real-time camera support

Average / min / max weight analytics

Cloud deployment

8. Conclusion

This project demonstrates a complete real-world ML pipeline:

Detection → Feature extraction → Regression → Deployment

Modular, scalable, and production-ready

Can be extended easily for research or industry use
