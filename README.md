# 🧠 About the Project: Bottle Cap Anomaly Detection

This project focuses on detecting anomalies in bottle caps using image classification techniques powered by AI. The objective is to automate the quality control process in manufacturing lines by identifying defective bottle caps (e.g., misaligned, broken, or missing caps) in real time. By leveraging computer vision and machine learning, the system aims to improve accuracy and reduce human error in defect detection.

# 🎯 Key Objectives

Detect and classify anomalies in bottle caps with high accuracy.

Enable real-time inspection for quality assurance.

Build an easily deployable solution using modern AI frameworks.

# 🛠️ Technologies Used

Python

TensorFlow + Teachable Machine – For model training and export.

Streamlit – For building an interactive and user-friendly web app.

OpenCV – For image preprocessing and camera integration (if enabled).

PIL (Python Imaging Library) – For image handling.

NumPy & Pandas – For data management and operations.


# 📊 Dataset Preparation

Dataset contains three folders (train, test, valid) with subfolders (images, labels). After training the model on Teachable Machine this dataset is pre-prepared as TeachableDataset which contain subfolders for each class (Normal, Anomaly) with respective images. 
A prepare_dataset.py script ensures the dataset is properly formatted for use with Teachable Machine or other model training pipelines.

# 🧪 Model and Methods

Model Training : The model was trained using Google Teachable Machine for rapid prototyping.

Algorithm : Transfer learning with a pre-trained Convolutional Neural Network (CNN) backbone (e.g., MobileNet) provided by Teachable Machine.

Classification : Binary classification between normal and anomaly images.

Real-Time Deployment : The trained model is integrated with a streamlit_app.py interface, allowing users to upload images or capture live frames for classification.

# 🚀 Features

Real-time anomaly detection through uploaded images.

Simple and intuitive UI for non-technical users.

Easily extendable to other manufacturing products with minor modifications.

Also the webcam feature to detect through camera feed will be added in further implementation.














