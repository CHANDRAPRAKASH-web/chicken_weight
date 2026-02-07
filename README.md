Chicken Weight Estimation Using Computer Vision
📌 Project Overview

This project estimates the weight of a chicken (in grams) from an image using Computer Vision and Deep Learning.

The pipeline combines:

YOLOv8 for detecting chickens in images

CNN feature extraction using ResNet

MLP regression model to predict weight from extracted features

FastAPI server for real-time prediction using images

🧠 Project Workflow (End-to-End)
1️⃣ Dataset Preparation

Images of chickens are collected across different days.

Each image has a ground-truth weight stored in weights.csv.

YOLO annotations are used to detect chickens.

2️⃣ Chicken Detection (YOLOv8)

YOLOv8 detects chickens and generates bounding boxes.

Detected chickens are cropped from images.

Cropped images are stored and linked to the original image.

📄 Script:

detect_and_crop.py

3️⃣ Feature Extraction

Each cropped chicken image is passed through ResNet18

A 512-dimensional feature vector is extracted per crop

Saved as:

dataset/features/features.npy


📄 Script:

extract_features.py

4️⃣ Weight Mapping

Each crop is mapped back to the original image

Weight is taken from weights.csv

Final dataset:
Features → Weight (grams)

5️⃣ MLP Model Training

A simple MLP regression model is trained

Input: 512 features

Output: weight in grams

📄 Script:

python train_mlp.py \
  --features dataset/features/features.npy \
  --meta dataset/crops/crops_meta.csv \
  --weights weights.csv


📁 Output:

models/mlp.pth

6️⃣ Offline Inference (Correct & Accurate)

This is the reference inference logic (most reliable).

📄 Script:

inference.py


Output:

predictions.csv

7️⃣ API Server (FastAPI)

Accepts an image

Detects chicken using YOLO

Extracts features

Predicts weight in grams

📄 Server File:

app.py


Start server:

uvicorn app:app --host 0.0.0.0 --port 8000


API Endpoint:

POST http://127.0.0.1:8000/predict


Input:

Multipart image (.jpg, .png)

Image should contain one visible chicken

Output:

{
  "pred_g": 121.3
}

📂 Project Structure
chicken_weight/
│
├── dataset/
│   ├── train/images/
│   ├── crops/
│   ├── features/
│
├── models/
│   ├── mlp.pth
│
├── weights.csv
├── train_mlp.py
├── extract_features.py
├── detect_and_crop.py
├── inference.py
├── app.py
├── requirements.txt
└── README.md

⚠️ Important Notes

Predictions are in grams

Input image should clearly show one chicken

Use inference.py logic as the ground truth reference

API uses the same logic (no averaging / no scaling tricks)

🚀 How a New Person Can Continue This Project
Step-by-Step Guide

Clone the repository

Install dependencies:

pip install -r requirements.txt


Prepare dataset + weights

Run detection & crop

Extract features

Train MLP

Test using inference.py

Deploy using FastAPI

🔮 Future Improvements

Multiple chicken detection

Average / total weight estimation

Mobile app integration

Camera live feed

Dataset augmentation

Model calibration using scale reference

🧑‍💻 Author

CP
Computer Vision | Deep Learning | AI Systems
