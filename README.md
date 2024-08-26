# Heart Failure Prediction Assignment

This Jupyter notebook is dedicated to analyzing and predicting heart failure outcomes using various machine learning models. The dataset includes patient information and is cleaned and processed before applying different predictive models.

## Table of Contents

- [Dataset Information](#dataset-information)
- [Data Cleaning](#data-cleaning)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)

## Dataset Information

The dataset used in this notebook is `Heart_Failure_Training_Set.csv`. It includes various patient features, and the target variable is `1Yr_Death`, which indicates whether the patient died within one year of the data collection.

### Key Features:
- Number of Admissions Last Year
- Number of ED Visits in Last 6 Months
- Number of Outpatient Visits Last Year
- Length of Stay (Days)
- Max Potassium Result
- Last Creatinine Days From Admit
- First Sodium Result

## Data Cleaning

The notebook includes comprehensive data cleaning steps:
- Identification and handling of missing values.
- Removal or imputation of NaN values.
- Conversion and normalization of features.

## Exploratory Data Analysis

We explore the data using various visualizations:
- Histograms for numerical features.
- Correlation matrix to identify the relationship between features.
- Heatmap of the correlation matrix.

## Model Training

The following machine learning models were trained and evaluated:
1. Random Forest Classifier
2. Logistic Regression
3. Support Vector Machine (SVM)
4. Gradient Boosting Classifier
5. Neural Network (Deep Learning)

Each model's hyperparameters were optimized to improve prediction accuracy.

## Model Evaluation

Models were evaluated using:
- Accuracy
- Confusion Matrix
- ROC-AUC score
- Precision-Recall curves

The final model was selected based on overall performance metrics.

## Conclusion

The notebook concludes with a discussion on the best-performing model and potential next steps for improving the prediction of heart failure outcomes.

## Dependencies

This notebook requires the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow (for neural networks)

You can install these dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## Usage

To run this notebook:
1. Clone the repository.
2. Ensure all dependencies are installed.
3. Open the notebook in Jupyter and run the cells sequentially.

## Author

This notebook was created as part of a heart failure prediction assignment. For any questions or suggestions, please contact the author.

---

