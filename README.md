# 🚗 Car Price Predictor (Backend)

This repository contains the machine learning model and Flask API for predicting the price of a used car based on user inputs.

### 📦 Features
- Trained on a cleaned dataset with preprocessing and feature selection
- Exposes a REST API endpoint for predicting price
- Additional `/options` endpoint provides dynamic dropdown values for the frontend

### 🌐 Live Frontend App
👉 [Visit the Web App](https://car-price-predictor-frontend-two.vercel.app/)

### 📁 Related Repositories
- Frontend Repo: [car-price-predictor-frontend](https://github.com/akshatbuliwal/car-price-predictor-frontend)

---

### ⚙️ API Endpoints

#### `POST /predict`
**Input** (JSON):
```json
{
  "company": "Hyundai",
  "name": "i20",
  "year": 2019,
  "fuel_type": "Petrol",
  "kms_driven": 15000
}
