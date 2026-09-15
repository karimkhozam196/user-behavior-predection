# 📱 User Behavior Prediction Using Machine Learning

A machine learning project for analyzing smartphone user behavior and predicting the user's **behavior class** based on device usage patterns, application usage, screen time, battery consumption, data usage, and demographic information.

The project covers a complete machine learning workflow, including **data preprocessing, feature engineering, exploratory data analysis, feature selection, clustering, classification, neural networks, model evaluation, and deployment with Gradio**.

---

## 📌 Project Overview

The goal of this project is to use smartphone usage data to identify and predict different levels of user behavior.

The dataset contains information such as:

* Operating System
* App Usage Time
* Screen On Time
* Battery Drain
* Number of Apps Installed
* Data Usage
* Age
* Gender
* Device Model
* User Behavior Class

The project applies several machine learning techniques to understand the dataset and build predictive models.

---

## 🚀 Features

The project includes:

* Data loading and preprocessing
* Missing-value handling
* Duplicate-value detection and removal
* Feature engineering
* Feature scaling
* Categorical feature encoding
* Exploratory Data Analysis (EDA)
* Correlation analysis
* Feature selection using Chi-Square
* K-Means clustering
* Decision Tree classification
* Gaussian Naive Bayes classification
* Confusion matrix evaluation
* PyTorch Convolutional Neural Network
* Multiple Naive Bayes experiments
* Model saving with Joblib
* User behavior prediction
* Interactive Gradio web interface

---

## 🗂️ Project Workflow

```text
Dataset
   │
   ▼
Data Preprocessing
   │
   ├── Missing Values
   ├── Duplicate Removal
   ├── Encoding
   └── Feature Scaling
   │
   ▼
Feature Engineering
   │
   ├── Usage Efficiency
   └── Binning
   │
   ▼
Exploratory Data Analysis
   │
   ├── Correlation Matrix
   ├── Feature Relationships
   └── Data Visualization
   │
   ▼
Feature Selection
   │
   └── Chi-Square
   │
   ▼
Machine Learning
   │
   ├── K-Means
   ├── Decision Tree
   ├── Naive Bayes
   └── CNN
   │
   ▼
Model Evaluation
   │
   ├── Accuracy
   ├── Confusion Matrix
   └── Classification Results
   │
   ▼
Model Saving
   │
   ▼
Gradio Application
   │
   ▼
User Behavior Prediction
```

---

## 📊 Dataset

The project uses a smartphone user behavior dataset containing user/device usage information.

### Main Features

| Feature                      | Description                      |
| ---------------------------- | -------------------------------- |
| `User ID`                    | Unique identifier for the user   |
| `Device Model`               | User's smartphone model          |
| `Operating System`           | Smartphone operating system      |
| `App Usage Time (min/day)`   | Daily application usage          |
| `Screen On Time (hours/day)` | Daily screen-on duration         |
| `Battery Drain (mAh/day)`    | Daily battery consumption        |
| `Number of Apps Installed`   | Number of installed applications |
| `Data Usage (MB/day)`        | Daily mobile data usage          |
| `Age`                        | User age                         |
| `Gender`                     | User gender                      |
| `User Behavior Class`        | Target behavior category         |

---

## 🧹 Data Preprocessing

Several preprocessing techniques are applied before training the models.

### Missing Values

Missing numerical values are handled using statistical values such as the median.

Categorical missing values are handled using the mode.

### Duplicate Removal

Duplicate records are identified and removed to improve data quality.

### Feature Engineering

A **Usage Efficiency** feature is calculated using application usage and screen-on time:

```text
Usage Efficiency =
App Usage Time / (Screen On Time × 60)
```

This provides an additional measurement of how application usage relates to total screen usage.

### Encoding

Categorical features such as:

* Gender
* Operating System

are converted into numerical representations using `LabelEncoder`.

### Feature Scaling

Numerical features are standardized using `StandardScaler`.

---

## 📈 Exploratory Data Analysis

The project performs exploratory analysis to understand relationships between the different variables.

Correlation analysis is used to identify relationships between numerical features and the target-related variables.

Visualizations include correlation matrices and other plots for understanding the dataset.

---

## 🎯 Feature Selection

The project uses the **Chi-Square statistical test** to evaluate categorical features.

The process includes:

1. Converting continuous variables into bins.
2. Encoding categorical variables.
3. Applying the Chi-Square test.
4. Comparing feature scores.

This helps identify features that have stronger relationships with the selected target variable.

---

## 🤖 Machine Learning Models

Several machine learning approaches are implemented.

### 1. K-Means Clustering

K-Means is used to group users according to their usage characteristics.

The workflow includes:

```text
Feature Selection
      ↓
Categorical Encoding
      ↓
Feature Scaling
      ↓
K-Means
      ↓
User Clusters
```

---

### 2. Decision Tree

A Decision Tree classifier is trained using selected user behavior features.

The model is evaluated using a train/test split and classification metrics.

---

### 3. Gaussian Naive Bayes

Gaussian Naive Bayes is used as one of the main classification models for predicting user behavior.

The project experiments with both selected features and the full feature set.

---

### 4. Convolutional Neural Network

A CNN is also implemented using **PyTorch**.

The neural network processes the numerical feature representation of the dataset and predicts the user behavior class.

The model includes:

* 1D convolutional layers
* Activation functions
* Pooling
* Fully connected layers
* Cross-entropy loss
* Adam optimizer

---

## 📊 Model Evaluation

The project evaluates classification models using techniques such as:

* Accuracy
* Confusion Matrix
* Prediction results
* Training and testing performance

Example evaluation workflow:

```text
Model
  ↓
Test Dataset
  ↓
Predictions
  ↓
Confusion Matrix
  ↓
Accuracy / Evaluation
```

---

## 💾 Model Saving

The trained Naive Bayes model and categorical encoders are saved using **Joblib**.

This allows the trained model to be loaded later without retraining it.

Example:

```python
import joblib

joblib.dump(nb_model, "naive_bayes_model.pkl")
joblib.dump(encoders, "encoders.pkl")
```

---

## 🌐 Gradio Application

The project includes an interactive **Gradio** interface for making predictions.

The user can enter:

* Operating System
* App Usage Time
* Screen On Time
* Battery Drain
* Number of Apps Installed
* Data Usage
* Age
* Gender

The application then uses the trained model to predict the user's behavior class.

### Example

```text
Operating System: Android
App Usage Time: 120 min/day
Screen On Time: 5 hours/day
Battery Drain: 800 mAh/day
Number of Apps: 40
Data Usage: 1500 MB/day
Age: 22
Gender: Male
```

After submitting the information, the application returns the predicted **User Behavior Class**.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Deep Learning

* PyTorch

### Model Deployment / Interface

* Gradio

### Model Serialization

* Joblib

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Move into the project directory:

```bash
cd YOUR_REPOSITORY
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch gradio joblib
```

---

## ▶️ Running the Project

Make sure the dataset is located in the project directory.

For example:

```text
project/
│
├── user_behavior_dataset.csv
├── pre13.py
├── naive_bayes_model.pkl
├── encoders.pkl
└── README.md
```

Run the Python script:

```bash
python pre13.py
```

The Gradio application will start and provide a local web interface.

Open the displayed Gradio URL in your browser.

---

## 🖥️ Application Interface

The Gradio application provides a simple interface where users can enter their smartphone usage information and receive a predicted behavior class.

```text
┌─────────────────────────────────────┐
│     User Behavior Prediction App    │
├─────────────────────────────────────┤
│ Operating System     [ Android ▼ ]  │
│ App Usage Time       [ 120       ]  │
│ Screen On Time       [ 5         ]  │
│ Battery Drain        [ 800       ]  │
│ Apps Installed       [ 40        ]  │
│ Data Usage           [ 1500      ]  │
│ Age                  [ 22        ]  │
│ Gender               [ Male ▼    ]  │
│                                     │
│             [ Submit ]               │
│                                     │
│ Prediction: User Behavior Class     │
└─────────────────────────────────────┘
```

---

## 📁 Suggested Project Structure

```text
User-Behavior-Prediction/
│
├── user_behavior_dataset.csv
├── pre13.py
├── naive_bayes_model.pkl
├── encoders.pkl
├── README.md
└── requirements.txt
```

---

## 📋 Requirements

A `requirements.txt` file can contain:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
torch
gradio
joblib
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

Possible improvements for the project include:

* Improve model hyperparameter tuning
* Compare additional classification algorithms
* Add cross-validation
* Improve CNN architecture
* Add more evaluation metrics
* Improve the Gradio UI
* Add interactive visualizations
* Deploy the application to Hugging Face Spaces
* Add automated model retraining
* Improve prediction reliability with additional user data

---

## 👨‍💻 Author

**AI Student | Machine Learning & Deep Learning Enthusiast**

This project was developed as part of an academic machine learning project to explore the complete process of preparing data, training machine learning models, evaluating results, and deploying a predictive application.

---

## ⭐ Project Highlights

* End-to-end machine learning workflow
* Multiple ML algorithms
* Deep learning implementation with PyTorch
* Statistical feature selection
* Unsupervised learning with K-Means
* Interactive Gradio prediction application
* Model persistence using Joblib
* Practical smartphone user behavior analysis

---

## 📄 License

This project is intended for educational and academic purposes.
