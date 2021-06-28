# Credit_Risk_Analysis



## Overview

This goal of project was to learn how to employ different techniques to train and evaluate models with unbalanced classes. We used the imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. For each machine learning model, the steps are:

1. Prepare data (ETL, identfy input 'features' (aka factors') from output 'targets', and optionally encode)
2. Instantiate a machine learning model with a random seed and other parameters
3. Create training and testing datasets
4. Fit/train the model with training data
5. Make predictions with test data
6. Evaluate the model's performance
7. Repeat steps 2-5 until goals are satisfied

###### Technologies Used:

- Jupyter Notebook
- LendingClub peer-to-peer data (LoanStats_2019Q1.csv)
- Scikit-learn API
  - sklearn.preprocessing.StandardScaler
  - sklearn.linear_model.LogisticRegression
  - sklearn.ensemble.AdaBoostClassifier (with EasyEnsembleClassifier below)
- Imbalanced-learn API
  - imblearn.over_sampling.RandomOverSampler (over_sampling)
  - imblearn.over_sampling.SMOTE + (over_sampling)
  - imblearn.under_sampling.ClusterCentroids (under_sampling)
  - imblearn.combine.SMOTEENN ++ (combination of over_sampling and under_sampling)
  - imblearn.ensemble.BalancedRandomForestClassifier (reduces bias)
  - imblearn.ensemble.EasyEnsembleClassifier (reduces bias)

Notes:
+ SMOTE = synthetic minority oversampling technique
+ SMOTEEN = SMOTE and Edited Nearest Neighbors (ENN)

## Results

Please see **Table 1 in the Summary** section at the bottom of this README file for the performance and recommendation data. The detailed results (accuracy score, confusion matrix, and imbalanced classification report) for each ML model are shown below in Figures 3, 5, 6, 7, 8, 9, and 11. The sorted features for BalancedRandomForestClassifier (and probable bug) are shown in Figure 10 below.

#### Deliverable 1 - Use Resampling Models to Predict Credit Risk

Using our knowledge of the imbalanced-learn and scikit-learn libraries, we evaluated three machine learning models by using resampling to determine which is better at predicting credit risk. First, we used the oversampling RandomOverSampler and SMOTE algorithms, and then we used the undersampling ClusterCentroids algorithm. Using these algorithms, we resampled the dataset, viewed the count of the target classes, trained a logistic regression classifier, calculated the balanced accuracy score, generated a confusion matrix, and generated a classification report.

##### Credit Risk Resampling Techniques

###### Step 1: Read the CSV and Perform Basic Data Cleaning

Figure 1. Dataframe 'df' created by reading LoanStats_2019Q1.csv

![loan_data_dataframe.png](Images/loan_data_dataframe.png)

Figure 2. Create 'features' dataframe X, and 'target' Series/array y

![separate_features_targets.png](Images/separate_features_targets.png)

Figure 3. The X_enc dataframe after encoding with pandas.get_dummies()

![dataframe_after_encoding.png](Images/dataframe_after_encoding.png)

###### Step 2: Split the Data into Training and Testing

Figure 4. Calling sklearn.model_selection.train_test_split() to create train and test data

![split_data.png](Images/split_data.png)

##### Oversampling

###### Step 3: Naive Random Oversampling

Figure 5. Results for Naive Random Oversampling with imblearn.over_sampling.RandomOverSampler

![naive_random_oversampling.png](Images/naive_random_oversampling.png)

###### Step 4: SMOTE Oversampling

Figure 6. Results for SMOTE Oversampling with imblearn.over_samplng.SMOTE

![smote_oversampling.png](Images/smote_oversampling.png)

##### Step 5: Undersampling

Figure 7. Results for Undersampling with imblearn.under_sampling.ClusterCentroids

![undersampling.png](Images/undersampling.png)

#### Deliverable 2 - Use the SMOTEENN algorithm to Predict Credit Risk

Using our knowledge of the imbalanced-learn and scikit-learn libraries, we used a combinatorial approach of over- and undersampling with the SMOTEENN algorithm to determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms from Deliverable 1. Using the SMOTEENN algorithm, we resampled the dataset, viewed the count of the target classes, trained a logistic regression classifier, calculated the balanced accuracy score, generated a confusion matrix, and generated a classification report.

##### Step 6: Combination (Over and Under) Sampling

Figure 8. Results for Combination (Over and Under) Sampling with imblearn.combine.SMOTEEN

![smoteen.png](Images/smoteen.png)

#### Deliverable 3 - Use Ensemble Classifiers to Predict Credit Risk

Using our knowledge of the imblearn.ensemble library, we trained and compared two different ensemble classifiers, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk and evaluate each model. Using both algorithms, we resampled the dataset, viewed the count of the target classes, trained the ensemble classifier, calculated the balanced accuracy score, generated a confusion matrix, and generated a classification report.

##### Step 7a: Balanced Random Forest Classifier

Figure 9. Results for Ensemble Classifiers with imblearn.ensemble.BalancedRandomForestClassifier

![balanced_random_forest.png](Images/balanced_random_forest.png)

##### Step 7b: Sort the features_importances_ for BalancedRandomForestClassifier

The column name vs features_importances_ value are sorted in Figure 10. The list of names & values given in the sample code shows that the values were sorted in reverse numerical order, then zipped with the original column names. This is the 'wrong' way, since the values are rearranged, but the column names of the features are not. So the values do not match the name for the value. Also shown in Figure 10 is the 'correct' way to do it.

Note that the wrong way shows the most important feature to be the loan amount (loan_amnt), and the second most important feature is the interest rate (int_rate).

The correct way shows that the most important feature is total recovered principle (total_rec_prncp), and the second most important feature is the total payments made (total_pymnt).

Figure 10. Sorted Features, and probable bug in credit_risk_ensemble_starter_code.ipynb

![sort_vs_zip_first.png](Images/sort_vs_zip_first.png)

##### Step 8: Easy Ensemble AdaBoost Classifier

Figure 11. Results for Ensemble Classifiers with imblearn.ensemble.EasyEnsembleClassifier (with AdaBoost)

![easy_ensemble.png](Images/easy_ensemble.png)

## Summary

The summary results for all ML models is shown in Table 1.

The recommendation for **the overall best-performing Machine Learning (ML) model was the EasyEnsemble with AdaBoost classifier**. It's Balanced Accuracy Score (Acc) was .9317 (93.17%), which was the highest of any of the models. Precision (Prec), Sensitivity/Recall (Sens), and Harmonic Mean (F1) are shown for the 'High-Risk' (H) and 'Low-Risk' (L) categories, and Average (Avg) over both H & L.

Table 1. Comparison of ML Model Performance Metrics

| Model                          |   Acc |     Prec  (H/L/Avg)|     Sens (H/L/Avg) |        F1 (H/L/Avg)|
| :--                            |   --: |                --: |                --: |               --:  |
| Naive Random Oversampling      | .6648 | 0.01 / 1.00 / 0.99 | 0.73 / 0.60 / 0.60 | 0.02 / 0.75 / 0.74 |
| SMOTE Oversampling             | .6624 | 0.01 / 1.00 / 0.99 | 0.63 / 0.69 / 0.69 | 0.02 / 0.82 / 0.81 |
| ClusterCentroids Undersampling | .5447 | 0.01 / 1.00 / 0.99 | 0.69 / 0.40 / 0.40 | 0.01 / 0.57 / 0.56 |
| SMOTEEN Over and Undersampling | .6435 | 0.01 / 1.00 / 0.99 | 0.71 / 0.57 / 0.57 | 0.02 / 0.73 / 0.72 |
| BalancedRandomForest Ensemble  | .7885 | 0.03 / 1.00 / 0.99 | 0.70 / 0.87 / 0.87 | 0.06 / 0.93 / 0.93 |
| **EasyEnsemble AdaBoost**      | **.9317** | **0.09 / 1.00 / 0.99** | **0.92 / 0.94 / 0.94** | **0.16 / 0.97 / 0.97** |
