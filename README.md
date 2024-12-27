# Capstone Project for 'Machine-Learning-Zoomcamp'

## Introduction
Employee attrition is a pressing issue for organizations, impacting efficiency, morale, and costs. ABC, a leading firm, has seen attrition rise from 14% to 25% in the past year, posing a risk to its stability. Using employee data, including attributes like age, department, and work-life balance, we aim to identify factors driving attrition and predict employees likely to leave.

Accurate attrition prediction allows companies to implement targeted retention strategies, reduce costs, and enhance workforce stability, ensuring sustainable growth.

For further details about the dataset, see [Employee-Attrition-Rate](https://www.kaggle.com/datasets/prachi13/employeeattritionrate)

## Setup & Project Details

### Local Setup

1. **Environment Setup**:
   Create a Python virtual environment (Python >= 3.10 recommended) and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. **Preliminary Analysis**:
   Preliminary exploration work is in `notebook.ipynb`, which includes:

   - **Data Cleaning & EDA**:
      - Clean the dataset and check the number of missing values in each column.
      - Visualize feature distributions using `seaborn`.
      - Analyze Point-biserial correlations between numeric features and the target variable.
      - Analyze mutual information between categorical features and the target variable.

   - **Feature Engineering & Modeling & Hyperparameter Tuning**:
      - Categorical features are encoded by `WOEEncoder` from the `category_encoders` library.
      - Train and evaluate Logistic Regression, Random Forest, and XGBoost models.
      - Use `CalibratedClassifierCV1 from 1scikit-learn` to calibrate prediction probabilities
      - Use `Pipeline` and `GridSearchCV` from `scikit-learn` for hyperparameter tuning.
      - Results indicate that the Random Forest model outperforms others in terms of negative log loss, achieving the best performance.

3. **Model Training**:
   Train the final Random Forest model using the optimal hyperparameters obtained from grid search and save the complete pipeline for inference:

   ```bash
   python train.py
   ```

4. **Launch Web Service**:
   Run the prediction service using Streamlit:

   ```bash
   streamlit run predict.py
   ```

   Access the service at [http://localhost:8501/](http://localhost:8501/).

---

### Run with Docker

1. **Build the Docker Image**:
   ```bash
   docker build -t ml-project .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -p 8501:8501 ml-project
   ```

   The web service will be available at [http://localhost:8501/](http://localhost:8501/).

---