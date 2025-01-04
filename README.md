# Credit Card Fraud Detection  

This project implements various machine learning models to detect fraudulent credit card transactions using a dataset. The models are trained and evaluated to compare their performance using metrics like confusion matrix and classification reports.  

---  

## Table of Contents  

- [Introduction](#introduction)  
- [Dataset](#dataset)  
- [Dependencies](#dependencies)  
- [Project Workflow](#project-workflow)  
- [Code Details](#code-details)  
- [How to Run the Code](#how-to-run-the-code)  
- [Results](#results)  
- [References](#references)  

---  

## Introduction  

Credit card fraud detection is a vital application of machine learning, aiming to identify fraudulent transactions from large datasets efficiently. This project evaluates multiple classification algorithms to determine the best-performing model.  

---  

## Dataset  

The dataset used in this project is `creditcard.csv`. It includes:  

- **Features**: Variables representing transaction details.  
- **Class**: Target variable (`1` for fraud, `0` for genuine).  

---  

## Dependencies  

Ensure you have the following Python libraries installed:  

- `pandas`  
- `numpy`  
- `matplotlib`  
- `seaborn`  
- `scikit-learn`  

Install them using:  

```bash  
pip install pandas numpy matplotlib seaborn scikit-learn  
```
## Project Workflow  

The project consists of the following steps:  

1. **Data Loading and Exploration**  
   - Load the dataset and inspect for null values, data types, and statistical summaries.  

2. **Data Visualization**  
   - Analyze data distribution and correlations using boxplots and heatmaps.  

3. **Data Splitting**  
   - Split the dataset into training and testing sets (70-30 split).  

4. **Model Building**  
   - Train the following machine learning models:  
     - Logistic Regression  
     - Decision Tree Classifier  
     - Random Forest Classifier  
     - Extra Trees Classifier  
     - K-Nearest Neighbors  
     - Support Vector Classifier  
     - Bagging Classifier  
     - Gradient Boosting Classifier  
     - Naive Bayes Classifier  

5. **Model Evaluation**  
   - Use confusion matrices and classification reports to evaluate models.  

---  

## Code Details  

### Data Loading  

Load the dataset and inspect its structure:  

```python  
df = pd.read_csv("creditcard.csv")  
print(df.info())  
print(df.describe())  
```  

### Data Visualization  

- **Boxplot**:  

```python  
sns.boxplot(data=df)  
plt.show()  
```  

- **Heatmap**:  

```python  
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")  
plt.title("Correlation Heatmap")  
plt.show()  
```  

### Splitting Data  

Split the dataset into features (`X`) and target (`y`), and then into training and testing sets:  

```python  
from sklearn.model_selection import train_test_split  

X = df.drop("Class", axis=1)  
y = df["Class"]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  
```  

### Model Training and Evaluation  

Train and evaluate multiple models:  

```python  
from sklearn.metrics import confusion_matrix, classification_report  

models = [  
    LogisticRegression(),  
    DecisionTreeClassifier(),  
    RandomForestClassifier(),  
    ExtraTreesClassifier(),  
    KNeighborsClassifier(),  
    SVC(),  
    BaggingClassifier(),  
    GradientBoostingClassifier(),  
    GaussianNB()  
]  

for model in models:  
    model.fit(X_train, y_train)  
    y_pred = model.predict(X_test)  
    print(f"Model: {model.__class__.__name__}")  
    print("Confusion Matrix:")  
    print(confusion_matrix(y_test, y_pred))  
    print("Classification Report:")  
    print(classification_report(y_test, y_pred))  
    print("-" * 50)  
```  

---  

## How to Run the Code  

1. Clone the repository:  

```bash  
git clone https://github.com/your-username/credit-card-fraud-detection.git  
cd credit-card-fraud-detection  
```  

2. Ensure the dataset (`creditcard.csv`) is in the project directory.  

3. Run the script:  

```bash  
python main.py  
```  

4. Check the terminal output for model evaluation results.  

---  

## Results  

The project compares the performance of various classification models. Key findings include:  

- **Random Forest Classifier** performed the best, achieving the highest accuracy with strong precision and recall.  
- Models like **SVC** and **KNN** underperformed due to the imbalanced dataset.  

---  

## References  

- **Dataset**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- **Scikit-learn documentation**: [https://scikit-learn.org/](https://scikit-learn.org/)  

---  

