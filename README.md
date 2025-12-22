# 📈 Pakistan Inflation Forecasting System (End-to-End MLOps)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Docker](https://img.shields.io/badge/Deployment-Docker-blue)
![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)

### 🔴 Live Demo: [Click Here to View App](https://inflation-backend-6.onrender.com)
*(Note: Hosted on Render Free Tier. Please allow 50-60 seconds for the container to wake up!)*

---

## 📖 Project Overview
Inflation in Pakistan is highly volatile and "sticky." Traditional linear models often fail to capture sudden economic shocks. This project is a **production-grade Machine Learning application** designed to forecast the **CPI (Consumer Price Index) Inflation Rate** for the upcoming month.

Unlike static notebooks, this is a **Full-Stack MLOps** solution that integrates historical economic data with **Real-Time News Sentiment Analysis** to provide a holistic view of the economy.

---

## 🚀 Key Features
* **🤖 Advanced Forecasting:** Uses **XGBoost Regressor** trained on historical indicators (CPI, SPI, WPI, Forex, Oil Prices).
* **📰 Live Sentiment Analysis:** Fetches real-time economic headlines via API and calculates an **"Inflation Pressure Index"** based on news sentiment.
* **🎛️ Interactive Simulation:** A **Streamlit** dashboard allows users to tweak inputs (e.g., "What if Oil prices go up?") and see immediate predictions.
* **🐳 Dockerized Architecture:** Fully containerized Frontend and Backend for consistent deployment.
* **🔄 CI/CD Pipelines:** Automated testing and deployment workflows using **GitHub Actions**.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **ML Model** | `XGBoost` | Gradient boosting for non-linear time-series regression. |
| **Backend** | `FastAPI` | High-performance async API for model inference. |
| **Frontend** | `Streamlit` | Interactive UI for visualization and user input. |
| **Data Processing** | `Pandas` / `NumPy` | Feature engineering (Lags, MoM calculations). |
| **DevOps** | `Docker` & `Docker Compose` | Containerization of services. |
| **Automation** | `GitHub Actions` | CI/CD pipeline for automated testing. |

---

## 📸 Project Screenshots

### 1. Interactive Forecasting Dashboard
*Users can input Lag values and Month-on-Month (MoM) changes to simulate scenarios.*
![Dashboard Screenshot](assets/dashboard.png)
*(Place your screenshot of the inputs here)*

### 2. Real-Time News Analysis
*The system scrapes live headlines to detect inflationary pressure.*
![News Screenshot](assets/sentiment.png)
*(Place your screenshot of the news section here)*

---

## ⚙️ Installation & Running Locally

### Option 1: Using Docker (Recommended)
If you have Docker installed, you can run the entire system with one command:

```bash
# Clone the repository
git clone [https://github.com/Musmannazir/Inflation-Forecasting-System.git](https://github.com/Musmannazir/Inflation-Forecasting-System.git)
cd Inflation-Forecasting-System

# Build and Run containers
docker-compose up --build

🧠 Model Architecture
The model treats inflation forecasting as a Supervised Regression problem.

Input: Historical Data (Lags 1-3 months) + External Regressors (Oil, USD/PKR).

Transformation: Data is converted to Stationary format using Month-on-Month (MoM) percentage changes.

Model: XGBoost Regressor (Optimized via GridSearch for max_depth and learning_rate).

Evaluation: Validated using RMSE (Root Mean Squared Error) to penalize large outliers.

🔮 Future Work
Data Drift Monitoring: Integration of Evidently AI to detect shifts in economic distributions over time.

Automated Retraining: Implementing Prefect to orchestrate monthly retraining pipelines when new bureau data is released.

Deep Learning: Experimenting with LSTM (Long Short-Term Memory) networks for capturing longer-term temporal dependencies.

👨‍🏫 Acknowledgments
This project was developed at Ghulam Ishaq Khan Institute (GIKI). Special thanks to my mentors for their technical guidance:

Mr. Ali Imran Sandhu (Course Instructor)

Mr. Asim Shah (Lab Instructor)

📬 Contact
Muhammad Usman Nazir

GitHub: Musmannazir

LinkedIn: Connect with me
