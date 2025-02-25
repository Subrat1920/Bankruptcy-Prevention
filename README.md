# Bankruptcy Prevention Using Machine Learning
#### 📌 Project Overview
Bankruptcy is a critical issue that can have severe economic and financial consequences for businesses. This project aims to develop a machine learning model to predict bankruptcy risk based on financial and operational indicators. By leveraging historical financial data, we can build a predictive model that helps organizations take proactive measures to avoid bankruptcy.

#### 🚀 Objectives
    - Analyze financial and operational risk factors associated with bankruptcy.
    - Clean and preprocess financial data for accurate model training.
    - Implement and compare multiple machine learning models.
    - Provide a dashboard or report to visualize predictions.
    - Offer actionable insights to businesses for bankruptcy prevention.

#### 📊 Dataset Description
The dataset contains financial and operational risk factors related to bankruptcy. 

- **Features:**
  - `industrial_risk`: Risk level in the industry (0-1 scale).
  - `management_risk`: Level of management risk (0-1 scale).
  - `financial_flexibility`: Financial adaptability score (0-1 scale).
  - `credibility`: Creditworthiness of the business (0-1 scale).
  - `competitiveness`: Market competitiveness (0-1 scale).
  - `operating_risk`: Operational risks associated with the business (0-1 scale).
  
- **Target Variable:**
  - `class`: Indicates whether the business is at risk of bankruptcy (`bankruptcy`) or not (`non-bankruptcy`).


### 📌 How to Run the Project
##### 1. Clone the Repository
        git clone https://github.com/Subrat1920/Bankruptcy-Prevention.git
##### 2. Create an Virtual Environment
        python -m venv venv
##### 3. Activate Virtual Environment
        venv\Scripts\activate
##### 4. Install all the Dependenices
        pip install -r requirements.txt
##### 5. Run the Data Ingestion module to get Data Ingested and Start Model Training
        python -m src.components.data_ingestion
##### 6. Run the Flask App to get the UI/UX 
        python app.py        
##### 7. Open the link below
        http://127.0.0.1:5000/

### How to Run with MLFLOW
##### 1. Open new terminal
##### 2. Activate Virtual Environment
        venv\Scripts\activate
##### 3. Start MLFlow UI
        mlflow ui
##### 4. Run MLFlow UI File for Tracing
        python train_with_mlflow.py
##### 5. Open the link
        http://localhost:5000/#/experiments/0

#### 🏗️ Project Architecture
        1️⃣ **Data Ingestion:** Load and preprocess financial data.
        2️⃣ **Feature Engineering:** Select and transform relevant features.
        3️⃣ **Model Training:** Train multiple machine learning models.
        4️⃣ **Evaluation:** Compare model performance using metrics.
        5️⃣ **Deployment:** Integrate with Flask for real-time predictions.

