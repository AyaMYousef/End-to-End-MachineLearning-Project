# End-to-End-MachineLearning-Project

This project was completed as a practical exercise for the **Udemy Machine Learning Diploma**. It demonstrates the complete life cycle of a machine learning regression task — from data exploration to model deployment.

🚀 **Live Demo**: _Coming soon 

---

## 📌 Project Overview

The goal is to predict **house prices** based on various features using multiple regression models. The project follows the standard end-to-end data science workflow:

- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Model Building & Tuning  
- Model Evaluation  
- Deployment with Flask

---

## 🛠 Technologies Used

- **Python**
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **Scikit-learn**, **XGBoost**, **LightGBM**
- **Flask** (for deployment)
- **HTML/CSS & Bootstrap** (frontend)
- **Git/GitHub**
---

## 📂 Project Structure

End-to-End-MachineLearning-Project/
│
├── static/ # Static frontend assets (CSS, JS, etc.)
├── templates/ # HTML templates for Flask
│ ├── base.html
│ └── predict.html
│
├── model/ # Saved model files
│ ├── best_model.pkl
│
├── app.py # Flask web application
├── model_building.ipynb # Jupyter notebook for EDA, training, and evaluation
├── requirements.txt # Python dependencies
└── README.md # Project description


---

## 📊 Models Implemented

- Linear Regression  
- Ridge & Lasso Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting (XGBoost, LightGBM)

After comparison, the best model was selected based on **R² Score**, **RMSE**, and **Cross-validation**.

---

## ⚙️ How to Run the Project Locally

1. **Clone the repository:**

```bash
git clone https://github.com/AyaMYousef/End-to-End-MachineLearning-Project.git
cd End-to-End-MachineLearning-Project

2. **Create a virtual environment & activate it:**

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

3. **Install dependencies:**

bash
Copy
Edit
pip install -r requirements.txt

4. **Run the Flask app:**
bash
Copy
Edit
python router.py

5. **Open your browser and navigate to:**

cpp
Copy
Edit
http://127.0.0.1:5000/

 **Features You’ll Learn**
✅ Exploratory Data Analysis
✅ Feature Engineering Techniques
✅ Handling Missing Values & Outliers
✅ Model Training and Evaluation
✅ Hyperparameter Tuning
✅ Model Serialization
✅ Building a Web App with Flask
✅ Deploying ML Apps



