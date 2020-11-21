# Credit Risk Analysis 
UTMCC DataViz Module 17,  Appling machine learning to solve credit card risk decisions.

---

## Contents 
  * Overview
    - Purpose
    - Resources
  * Results
  * Summary
 
---  

## Overview 
  
  Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, the project is to understand and establish a credit card risk solution based on machine learning algorithm options. As comparison to select a best-fit accurate model, consideration is given to oversample the data using the RandomOverSampler and SMOTE algorithms, and to undersample the data using the ClusterCentroids algorithm. And, to use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm, and to compare two models that reduce bias, the BalancedRandomForestClassifier and the EasyEnsembleClassifier. 

   ### Purpose
   To employ different techniques to train and evaluate models with unbalanced classes, such as imbalanced-learn and scikit-learn libraries, to build and evaluate models using resampling, in order to evaluate the performance of these models, and to make a written recommendation on whether one is best, or if any should be used to predict credit risk.
  
   The deliverables are: 
   - Deliverable 1: Use Resampling Models to Predict Credit Risk
   - Deliverable 2: Use the SMOTEENN Algorithm to Predict Credit Risk
   - Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk
   - Deliverable 4: A Written Report on the Credit Risk Analysis (this README.md)


   ### Resources
  * Data/content source file: LoanStats_2019Q1.csv
  * Software: Windows10, Python 3.8.3, Pandas, Scikit-learn, imblearn.ensemble, Jupyter Notebook 

<br>

--- 

## Results
 In the tables below, we see a comparison of the printed outputs of running the six different machine learning models. 
 
 * When looking at the accuracy scores and recall (sensitivity) scores, it is apparent that there is a difference in sampling method results. SMOTE Oversampling has the best combination of high scores with accuracy of 65% and recall of 68%. The other sampling methods may have be a little higher in accuracy, but their recall is much lower in comparision.
 * The results indicate that the two Ensemble Classifiers, Random Forest and Ensemble Adaboost, are the two best methods with this dataset. Both are superior in accuracy and also average recall, and both are highly effective with sensitivity for low-risk situations, at recalls of 1.00. However, the recall scores are much lower for high-risk situations.  
 
.

   ### Deliverable 1: Resampling Models to Predict Credit Risk

   | **Naive Random Oversampling** | **SMOTE Oversampling** | **Undersampling, Cluster Centroids** |  
   | :--- | :--- | :--- |  
   | ![random_over.png](https://github.com/larrydodson/Credit_Risk_Analysis/blob/main/random_over.png) | ![smote_over.png](https://github.com/larrydodson/Credit_Risk_Analysis/blob/main/smote_over.png) | ![cluster_centroids_under.png](https://github.com/larrydodson/Credit_Risk_Analysis/blob/main/cluster_centroids_under.png) | 
   | Balanced Accuracy Score: 0.6603<br>Precision, Avg: 0.99<br>Recall (Sensitivity), Avg: 0.58<br>F1 Score, Avg: 0.73 | Balanced Accuracy Score: 0.6537<br>Precision, Avg: 0.99<br>Recall (Sensitivity), Avg: 0.68<br>F1 Score, Avg: 0.81 | Balanced Accuracy Score: 0.5474<br>Precision, Avg: 0.99<br>Recall (Sensitivity), Avg: 0.41<br>F1 Score, Avg: 0.58 |  

.


   ### Deliverable 2: SMOTEENN Algorithm to Predict Credit Risk

   | **SMOTEENN Combination Over & Under Sampling Algorithm** |
   | :--- |
   | ![smoteenn_combo.png](https://github.com/larrydodson/Credit_Risk_Analysis/blob/main/smoteenn_combo.png) |
   | Balanced Accuracy Score: 0.6448<br>Precision, Avg: 0.99<br>Recall (Sensitivity), Avg: 0.57<br>F1 Score, Avg: 0.72 | 

.

   
   ### Deliverable 3: Ensemble Classifiers to Predict Credit Risk

   | **Balanced Random Forest Classifier** | **Easy Ensemble AdaBoost Classifier** | 
   | :--- | :--- | 
   | ![RandomForest_classifier.png](https://github.com/larrydodson/Credit_Risk_Analysis/blob/main/RandomForest_classifier.png) | ![Ensemble_adaboost_classifier.pn](https://github.com/larrydodson/Credit_Risk_Analysis/blob/main/Ensemble_adaboost_classifier.png) | 
   | Balanced Accuracy Score: 0.6830<br>Precision, Avg: 1.00<br>Recall (Sensitivity), Avg: 1.00<br>F1 Score, Avg: 1.00<br> <br> **Feature Importances**: ![RandomForest_importances.png](https://github.com/larrydodson/Credit_Risk_Analysis/blob/main/RandomForest_importances.png)  | Balanced Accuracy Score: 0.7326<br>Precision, Avg: 1.00<br>Recall (Sensitivity), Avg: 1.00<br>F1 Score, Avg: 1.00 | 

.

 
   ### Deliverable 4: Written Report on the Credit Risk Analysis 
   (this README.md)
   
<br>

---

# Summary
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.


| Comparing Scores:  ................................ | **Naive Random Oversampling** | **SMOTE Oversampling** | **Undersampling, Cluster Centroids** | **SMOTEENN Combination Over&Under Sampling** | **Balanced Random Forest Classifier** | **Easy Ensemble AdaBoost Classifier** | 
| :--- | ---: | ---: | ---: |  ---: | ---: | ---: | 
| 1) Balanced Accuracy Score:<br>2) Precision, Avg:<br>3) Recall (Sensitivity), Avg:<br>4) F1 Score, Avg: | 0.6603<br>0.99<br>0.58<br>0.73 | 0.6537<br>0.99<br>0.68<br>0.81 | 0.5474<br>0.99<br>0.41<br>0.58 | 0.6448<br>0.99<br>0.57<br>0.72 | 0.6830<br>1.00<br>1.00<br>1.00 | 0.7326<br>1.00<br>1.00<br>1.00 |



### Recommendation

   Easy Ensemble AdaBoost Classifier 



.

.end
