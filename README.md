# airLSTM

# LSTM for CO Concentration Forecasting  

## 📌 Project Overview  
This project implements an **LSTM (Long Short-Term Memory) model** to forecast **carbon monoxide (CO) concentration** using the **UCI Air Quality dataset**. The model is trained to predict future CO levels based on past concentration values, helping in air pollution analysis and prediction.  

## 📊 Dataset: UCI Air Quality  
The dataset used is the **UCI Air Quality Dataset**, which contains sensor readings of pollutants, temperature, and humidity collected hourly in an Italian city.  

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/360/air+quality)  
- **Features Used:** CO concentration over time  
- **Target Variable:** Future CO concentration  

## 🚀 Model Architecture  
The model is a **deep learning-based time series forecaster** built using **PyTorch**. It consists of:  
✅ **LSTM Layer** – Captures temporal dependencies  
✅ **Fully Connected (Linear) Layer** – Outputs the predicted CO value  
✅ **MSE Loss Function** – Measures prediction accuracy  
✅ **Adam Optimizer** – Optimizes model weights  

## 📊 Plots

