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

   | **Naive Random Oversampling** | **SMOTE Oversampling** |   
   | :--- | :--- |  
   | `ros = RandomOverSampler(random_state=1)`<br>`X_resampled, y_resampled = ros.fit_resample(X_train, y_train)`<br><br>![random_over.png](https://github.com/larrydodson/Credit_Risk_Analysis/blob/main/random_over.png) | `X_resampled, y_resampled = SMOTE(random_state=1, sampling_strategy='auto')` <br> `.fit_resample(X_train, y_train)` <br><br>![smote_over.png](https://github.com/larrydodson/Credit_Risk_Analysis/blob/main/smote_over.png) | 
   | Balanced Accuracy Score: 0.6603<br>Precision, Avg: 0.99<br>Recall (Sensitivity), Avg: 0.58<br>F1 Score, Avg: 0.73 | Balanced Accuracy Score: 0.6537<br>Precision, Avg: 0.99<br>Recall (Sensitivity), Avg: 0.68<br>F1 Score, Avg: 0.81 |

   | **Undersampling, Cluster Centroids** |  
   | :--- | 
   | `cc = ClusterCentroids(random_state=1)` <br> `X_resampled, y_resampled = cc.fit_resample(X_train, y_train)`<br><br>![cluster_centroids_under.png](https://github.com/larrydodson/Credit_Risk_Analysis/blob/main/cluster_centroids_under.png) |
   | Balanced Accuracy Score: 0.5474<br>Precision, Avg: 0.99<br>Recall (Sensitivity), Avg: 0.41<br>F1 Score, Avg: 0.58 |

.


   ### Deliverable 2: SMOTEENN Algorithm to Predict Credit Risk

   | **SMOTEENN Combination Over & Under Sampling Algorithm** |
   | :--- |
   | `smote_enn = SMOTEENN(random_state=0)`<br>`X_resampled, y_resampled = smote_enn.fit_resample(X, y)`<br><br>![smoteenn_combo.png](https://github.com/larrydodson/Credit_Risk_Analysis/blob/main/smoteenn_combo.png) |
   | Balanced Accuracy Score: 0.6448<br>Precision, Avg: 0.99<br>Recall (Sensitivity), Avg: 0.57<br>F1 Score, Avg: 0.72 | 

.

   
   ### Deliverable 3: Ensemble Classifiers to Predict Credit Risk

   | **Balanced Random Forest Classifier** | **Easy Ensemble AdaBoost Classifier** | 
   | :--- | :--- | 
   | `rf_model = RandomForestClassifier(n_estimators=500, random_state=1)`<br> `rf_model = rf_model.fit(X_train, y_train)`<br><br>![RandomForest_classifier.png](https://github.com/larrydodson/Credit_Risk_Analysis/blob/main/RandomForest_classifier.png) | `adaboost = AdaBoostClassifier(n_estimators=1000, learning_rate=1,random_state=1)`<br>`model = adaboost.fit(X_train, y_train)`<br><br>![Ensemble_adaboost_classifier.pn](https://github.com/larrydodson/Credit_Risk_Analysis/blob/main/Ensemble_adaboost_classifier.png) | 
   | Balanced Accuracy Score: 0.6830<br>Precision, Avg: 1.00<br>Recall (Sensitivity), Avg: 1.00<br>F1 Score, Avg: 1.00<br> <br> **Feature Importances**: <br> ![RandomForest_importances.png](https://github.com/larrydodson/Credit_Risk_Analysis/blob/main/RandomForest_importances.png)  | Balanced Accuracy Score: 0.7326<br>Precision, Avg: 1.00<br>Recall (Sensitivity), Avg: 1.00<br>F1 Score, Avg: 1.00 <br><br><br><br> <br><br><br><br> <br><br><br><br> <br><br><br><br> <br><br><br><br><br> | 

.

 
   ### Deliverable 4: Written Report on the Credit Risk Analysis 
   (this README.md)
   
<br>

---

# Summary
  * In addition to the comments above in the Results section on the six different machine learning models, below we see a more focused comparison for the scores on accuracy, precision, recall and the respective F1 scores. 
  * The F1 score, as a combination score of precision and recall, with the calculation formula of 2(Precision * Sensitivity)/(Precision + Sensitivity), gives an additional determination metric.
  * Of the sampling methods, the F1 score for SMOTE Oversampling is at 81%, and reinforces that it is the best method among the group of the four sampling methods.
  * When including in comparison the Ensemble Classifier methods, each has an F1 average score of 1.0, and also reinforces these two methods as superior overall. 


| .................................... <br> Comparing Scores: | **Naive Random Oversampling** | **SMOTE Oversampling** | **Undersampling, Cluster Centroids** | **SMOTEENN Combination Over&Under Sampling** | **Balanced Random Forest Classifier** | **Easy Ensemble AdaBoost Classifier** | 
| :--- | ---: | ---: | ---: |  ---: | ---: | ---: | 
| 1) Balanced Accuracy Score:<br>2) Precision, Avg:<br>3) Recall (Sensitivity), Avg:<br>4) F1 Score, Avg: | 0.6603<br>0.99<br>0.58<br>0.73 | 0.6537<br>0.99<br>0.68<br>0.81 | 0.5474<br>0.99<br>0.41<br>0.58 | 0.6448<br>0.99<br>0.57<br>0.72 | 0.6830<br>1.00<br>1.00<br>1.00 | 0.7326<br>1.00<br>1.00<br>1.00 |



### Recommendation

   Recommended choice of machine learning models, of the six candidate evaluations:<br> **Easy Ensemble AdaBoost Classifier**. <br>
     Reason: This model has the best overall scores in this comparison of the six ML models. 

.

.end
