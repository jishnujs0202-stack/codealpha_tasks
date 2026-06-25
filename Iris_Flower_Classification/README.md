# 🌸 Iris Flower Classification

## Project Overview

This project is a Machine Learning web application that predicts the species of an Iris flower based on its sepal and petal measurements. The model is trained using the famous Iris dataset and deployed using Streamlit.

## Objective

The objective of this project is to build a machine learning model capable of classifying Iris flowers into one of the following species:

* Iris Setosa
* Iris Versicolor
* Iris Virginica

## Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* Streamlit
* Joblib

## Dataset

The project uses the Iris dataset, which contains 150 flower samples with the following features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

Target Variable:

* Species of Iris Flower

## Project Workflow

### 1. Data Collection

Loaded the Iris dataset from Scikit-learn.

### 2. Data Preprocessing

* Checked dataset structure
* Separated features and target labels
* Prepared data for model training

### 3. Model Training

A Machine Learning classification model was trained using the Iris dataset.

### 4. Model Evaluation

The trained model was evaluated using test data to ensure accurate predictions.

### 5. Model Saving

The trained model was saved using Joblib for deployment.

### 6. Web Application Development

Built an interactive web application using Streamlit where users can enter flower measurements and obtain predictions instantly.

### 7. Deployment

The application was deployed online using Streamlit Cloud.

## Features

* User-friendly interface
* Real-time predictions
* Fast and lightweight deployment
* Interactive input fields

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```text
Iris_Flower_Classification/
│
├── app.py
├── iris_model.pkl
├── requirements.txt
├── README.md
└── iris.csv
```

## Results

The model successfully classifies Iris flowers into their respective species based on user-provided measurements.

## Live Demo

https://iris-flower-classification-hncp7wzxgka9gr7yiot9uk.streamlit.app/

## Author

J S Jishnu
B.Tech CSE (AI & Data Engineering)
