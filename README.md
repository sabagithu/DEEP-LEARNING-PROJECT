# DEEP-LEARNING-PROJECT

- **Company Name**: CodTech IT Solutions  
- **Intern Name**: *AYESHA SABA *  
- **Intern ID**: CT4MWP218  
- **Domain**: Data Science  
- **Duration**:  16 Weeks  
- **Mentor**: Neela Sanrosh 

# 🎬 IMDB Movie Review Sentiment Analysis

## 📖 Project Overview
This project is part of my internship deliverables at **CodTech IT Solutions**, focused on building a Natural Language Processing (NLP) model using deep learning techniques. The goal is to classify movie reviews from the IMDB dataset as either positive or negative based on their textual content.

After experimenting with multiple architectures—including a baseline LSTM and an improved LSTM—the final model combines **Convolutional Neural Networks (CNN)** with **LSTM** layers for enhanced performance.

---

## 🎯 Objectives
- Preprocess and prepare textual data for deep learning.
- Build and train various sentiment classification models.
- Evaluate and compare performance across architectures.
- Finalize the best-performing model (CNN + LSTM).

---

## 🛠 Tools & Technologies
- **Language**: Python 3.x  
- **Libraries**: TensorFlow, Keras, NumPy, Pandas  
- **Visualization**: Matplotlib, Seaborn  
- **Environment**: Jupyter Notebook / Google Colab / Conda (`tf_env`)

---

## 📂 Dataset
- **Source**: IMDB Movie Reviews Dataset (via TensorFlow Datasets)  
- **Size**: 50,000 labeled reviews (25,000 training / 25,000 testing)  
- **Balance**: Equal distribution of positive and negative sentiments

---

## 📋 Project Workflow

### 🔹 Data Loading
- Imported IMDB dataset using TensorFlow Datasets.

### 🔹 Exploratory Data Analysis (EDA)
- Analyzed class distribution, review lengths, and word frequency.

### 🔹 Text Preprocessing
- Tokenized reviews  
- Applied padding/truncation for uniform input length  
- Limited vocabulary size for efficiency

### 🔹 Model Building
- Implemented multiple models:
  - Baseline LSTM
  - Improved LSTM with dropout and tuning
  - Final CNN + LSTM hybrid model

### 🔹 Model Training & Evaluation
- Compared performance across models:
  - **Baseline LSTM Accuracy**: 73.39%  
  - **Improved LSTM Accuracy**: 81.98%  
  - **Final CNN + LSTM Accuracy**: **86.25%**

- Visualized training and validation metrics

---

## 📊 Final Results
- **Final Test Accuracy**: `0.8625`  
- **Final Test Loss**: `0.4854`  
- The CNN + LSTM model shows strong generalization and minimal overfitting.

---

## 📁 Repository Contents
- `Task2_deep_learning.ipynb` – Jupyter Notebook containing the full workflow: data loading, preprocessing, model building, training, and evaluation.  
- `best_sentiment_model.h5` – Saved LSTM model with optimized weights for sentiment classification.  
- `README.md` – Project documentation and overview.

---

## 🚀 Potential Applications
- Sentiment analysis for movie recommendation systems  
- Customer feedback classification in e-commerce platforms  
- Social media sentiment tracking for brand reputation

---

## 🧪 Environment Setup

This project was developed using a Conda environment for dependency management and reproducibility.

### 🔧 Environment Details
- **Environment Manager**: Conda  
- **Environment Name**: `tf_env`    
- **Python Version**: 3.x

### 📦 Installation
To recreate the environment:

```bash
conda create --name tf_env python=3.x
conda activate tf_env
pip install -r requirements.txt




