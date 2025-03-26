# smart_guard
NSL-KDD Classification Experiment：
This repository contains a Python script that performs a classification experiment on the NSL-KDD dataset. The script covers data loading, preprocessing, model training, evaluation, and visualization of performance metrics using both Decision Tree and Random Forest classifiers

Overview：
The NSL-KDD dataset is widely used for network intrusion detection. This project demonstrates how to:

Load and preprocess data: Convert ARFF files into pandas DataFrames, decode byte-encoded categorical values, and clean class labels.

Map attack types: Consolidate various attack types into four main categories: Normal, DDoS, MITM, and Other.

Feature engineering: One-hot encode categorical features and standardize numerical features.

Train models: Use Decision Tree and Random Forest classifiers.

Evaluate performance: Compute accuracy, precision, recall, F1-score, and generate classification reports.

Visualize results: Plot a bar chart comparing model performance metrics.

Repository Structure:
ddos_detection.py: Main Python script containing the code.
KDDTrain+.arff: Training dataset in ARFF format.
KDDTest+.arff: Test dataset in ARFF format.
README.md: This README file.

Requirements
Ensure you have Python 3 installed along with the following packages:
pandas
scipy
scikit-learn
matplotlib
numpy

How to Run
Place the dataset files (KDDTrain+.arff and KDDTest+.arff) in the same directory as ddos_detection.py.
Run the script using Python: python ddos_detection.py
The script will:
1.Load and preprocess the NSL-KDD dataset.
2.Train Decision Tree and Random Forest classifiers.
3.Print out performance metrics and classification reports.
4.Display a bar chart comparing the performance of the two models.