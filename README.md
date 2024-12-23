# ML_Churn_Prediction

Please see the coding file attached

## I. Introduction

### 1. Business question

- Customer churn prediction is a critical aspect of business management. It involves understanding and addressing customer attrition, which refers to the loss of clients or customers.
- For businesses in these sectors, measuring customer attrition is a vital business metric. This is because retaining an existing customer is significantly more cost-effective than acquiring a new one. As a result, these companies often have customer service branches dedicated to re-engaging customers who are considering leaving. This is because the long-term value of recovered customers far outweighs that of newly acquired ones.
- To address customer churn, predictive analytics comes into play. One ecommerce company has a project on predicting churned users in order to offer potential promotions.
- You will using these dataset to answer below questions:
  - What are the patterns/behavior of churned users? What are your suggestions to the company to reduce churned users.
  - Build the Machine Learning model for predicting churned users.

### 2. Dataset
The ecommerce dataset records customer information about their demographic and their behaviours. Each customer is labeled churned or not churned.

Dataset include these following main fields:

| Variable | Description |
| -- | -- |
| CustomerID | Unique customer ID |
| Churn | Churn Flag |
| Tenure | Tenure of customer in organization |
| PreferredLoginDevice | Preferred login device of customer |
| CityTier | City tier |
| WarehouseToHome | Distance in between warehouse to home of customer |
| PreferredPaymentMode | Preferred payment method of customer |
| Gender | Gender of customer |
| HourSpendOnApp | Number of hours spend on mobile application or website |
| NumberOfDeviceRegistered | Total number of deceives is registered on particular customer |
| PreferedOrderCat | Preferred order category of customer in last month |
| SatisfactionScore | Satisfactory score of customer on service |
| MaritalStatus | Marital status of customer |
| NumberOfAddress | Total number of added added on particular customer |
| Complain | Any complaint has been raised in last month |
| OrderAmountHikeFromlastYear | Percentage increases in order from last year |
| CouponUsed | Total number of coupon has been used in last month |
| OrderCount | Total number of orders has been places in last month |
| DaySinceLastOrder | Day Since last order by customer |

![image](https://i.imgur.com/ejvpwgP.png)


### 3. Method
Supervised learning with Scikit-learn on Python
- Supervised learning, also known as supervised machine learning, is a subcategory of machine learning and artificial intelligence. It is defined by its use of labeled datasets to train algorithms that to classify data or predict outcomes accurately.
- As input data is fed into the model, it adjusts its weights until the model has been fitted appropriately, which occurs as part of the cross validation process.
- Supervised learning helps organizations solve for a variety of real-world problems at scale, such as classifying spam in a separate folder from your inbox.
- In details:
  - Perform EDA to fidn out insightful data, discovered the pattern of churned and not churned users, then suggest the company to reduce churn users.
  - Preprocessing by filling missing data, checking for duplicates and correlation
  - Feature transforming with encoding, handling imbalanced data and normalizing
  - Training models with a number of model and select one to deploy

## II. Main Process

### 1. Cleaning and transforming dataset

- After investigating density and data field types of each column, I have provided some suitable data filling methods based on the distribution and context of the data.
  - tenure column uses bfills method
  - WTH column uses most frequent method
  - CouponUsed column uses KNNImputer
- There is no duplicated rows in dataset.
- Encoding data by pd.get_dummies(), handling imbalanced data using SMOTETomek, normalizing by MinMaxScaler

### 2. EDA and Feature Selection

![image](https://i.imgur.com/P3lHMVP.png)

- `Tenure`, `NumberofDeviceRegistered`, `PreferOrderCat`, `SatisfactionScore`, `MaritalStatus` and `Complains` columns have quite higher correlations than other columns
- Other columns I think that they might affect the target variable `Churn` : `DaySinceLastOrder`, `CashbackAmount`, `PreferredLoginDevice`, `WarehouseToHome`, `NumberOfAddress`,`OrderCount`   
-  However `OrderCount` and `NumberOfAddress` are correlated with some other columns above -> we can remove these two columns for next stages.

#### Feature Selection
After EDA we keep these features: 
-  `Tenure`
-  `NumberofDeviceRegistered`
-  `PreferOrderCat`
-  `SatisfactionScore`
-  `MaritalStatus`
-  `Complains`
-  `DaySinceLastOrder`
-  `CashbackAmount`
-  `PreferredLoginDevice`
-  `WarehouseToHome`

## III. Model training and evaluation

Apply to models: Logistic Regression, Decision Tree and Random Forest

![image](https://i.imgur.com/dZRiZa4.png)

![image](https://i.imgur.com/aKp1v2V.png)

![image](https://i.imgur.com/hIFZ8aY.png)

Comparing the balance accuracy of 3 models, we can see that RandomForest has the highest test set's `balanced accuracy score`(**0.929**). 
RandomForest also has the highest `precision`, `recall` and `F1-score` of 2 classes 0 and 1. => Choose **RandomForest** as the final model used to predict churners for this Company.

## IV. Enhance Machine Leanring Model

### Feature Importances

![image](https://i.imgur.com/EckNcEJ.png)

### Hyperparameter tuning
![image](https://i.imgur.com/rgvMpXs.png)

![image](https://i.imgur.com/jpreUD7.png)

### Probability Threshold selection

As default, model will get probability threshold = 0.5 for the model. If we change the probability threshold, the accuracy of model can improve? We will use ROC curve to check which probability threshold will have highest True Positive Rate and loweest False Positive rate.

![image](https://i.imgur.com/tRe1V5E.png)

![image](https://i.imgur.com/mVIy8XG.png)

After hyperparameter tuning & probability threshold selection, we will choose the model Random Forest with the hyperparameter as below and the probability = 0.5235714285714286.
- `n_estimators= 50`,
- `max_depth= 20`,
- `min_samples_split= 2`,
- `min_samples_leaf= 1`,
- `bootstrap= False`









