# Breast Cancer Classification

## Objective
The objective of this project is to apply supervised learning techniques to classify breast cancer cases using the Breast Cancer dataset from the `sklearn` library.

## Dataset
The dataset used is the Breast Cancer dataset available in `sklearn.datasets`. It contains features computed from digitized images of breast cancer biopsies to predict whether the cancer is malignant or benign.

## Key Steps

### 1. Loading and Preprocessing
- Loaded the dataset using `sklearn.datasets.load_breast_cancer()`.
- Checked for missing values.
- Scaled the features using `StandardScaler` to normalize the data.
- Split the data into training and testing sets (80% training, 20% testing).

### 2. Classification Algorithms
Implemented the following classification models:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **k-Nearest Neighbors (k-NN)**

Each model was trained and evaluated using accuracy, classification report, and confusion matrix.

### 3. Model Comparison
- Accuracy scores of all models were compared using a bar plot.
- The best and worst performing models were identified based on accuracy.

## Results
- The performance of each model was analyzed using accuracy metrics.
- The model with the highest accuracy was identified as the best performer.
- The model with the lowest accuracy was identified as the worst performer.

## Dependencies
To run this project, you need the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`

You can install missing dependencies using:
```
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage
Run the Jupyter Notebook (`breast_cancer_classification.ipynb`) to execute the classification pipeline and compare the performance of different models.

## Author
Abdul Basith



