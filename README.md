# detection-of-financial-fraud-in-transactions

## **README: Financial Fraud Detection**

**Project Title:** Credit Card Transaction Fraud Detection using Logistic Regression

**Purpose:**
This project aims to develop a machine learning model capable of identifying fraudulent credit card transactions based on historical data. The model utilizes Logistic Regression for classification.

**Dataset:**
* **Name:** credit_card.csv (assumed to be present in the same directory)
* **Description:** Contains characteristics of various credit card transactions, including:
    - Time: Transaction timestamp.
    - V1 to V28: Numerical features related to the transaction.
    - Amount: Transaction amount.
    - Class: Label indicating legitimate transaction (0) or fraudulent transaction (1).

**Methodology:**

1. **Data Loading and Exploration:**
   - Load the dataset using pandas.
   - Explore the data to understand its structure, features, and potential missing values.

2. **Data Preprocessing:**
   - Handle missing values (if any).
   - Analyze the distribution of fraudulent and legitimate transactions (highly imbalanced datasets require addressing).
   - Consider techniques like under-sampling or over-sampling to balance the dataset (optional in this example).

3. **Feature Engineering (Optional):**
   - Create new features based on existing ones to potentially improve model performance (not implemented in this example).

4. **Model Training and Evaluation:**
   - Split the data into training and testing sets for model evaluation.
   - Use Logistic Regression to build a classification model that predicts the probability of a transaction being fraudulent.
   - Train the model on the training data.
   - Evaluate the model's performance on the testing data using accuracy score.

**Implementation Highlights:**

- This example demonstrates under-sampling to balance the dataset (limited fraudulent transactions).
- Logistic Regression is chosen for its simplicity and interpretability. 

**Instructions:**

1. Ensure the `credit_card.csv` dataset is in the same directory as the script.
2. Run the script.
3. The script will:
    - Load and explore the data.
    - Perform under-sampling to balance the dataset (optional).
    - Split the data into training and testing sets.
    - Train a Logistic Regression model.
    - Evaluate the model's accuracy on the testing data.

**Dependencies:**

* pandas
* numpy
* scikit-learn

**Note:**

- This is a basic example of fraud detection using Logistic Regression.
- For real-world applications, consider more advanced algorithms, feature engineering techniques, and evaluation metrics.
- Imbalanced datasets require specific handling approaches.
- This model should not be used as a substitute for professional fraud detection systems.

**OUTPUTS**

1]First 5 rows of the dataset
Time	V1	V2	V3	V4	V5	V6	V7	V8	V9	...	V21	V22	V23	V24	V25	V26	V27	V28	Amount	Class
0	0	-1.359807	-0.072781	2.536347	1.378155	-0.338321	0.462388	0.239599	0.098698	0.363787	...	-0.018307	0.277838	-0.110474	0.066928	0.128539	-0.189115	0.133558	-0.021053	149.62	0.0
1	0	1.191857	0.266151	0.166480	0.448154	0.060018	-0.082361	-0.078803	0.085102	-0.255425	...	-0.225775	-0.638672	0.101288	-0.339846	0.167170	0.125895	-0.008983	0.014724	2.69	0.0
2	1	-1.358354	-1.340163	1.773209	0.379780	-0.503198	1.800499	0.791461	0.247676	-1.514654	...	0.247998	0.771679	0.909412	-0.689281	-0.327642	-0.139097	-0.055353	-0.059752	378.66	0.0
3	1	-0.966272	-0.185226	1.792993	-0.863291	-0.010309	1.247203	0.237609	0.377436	-1.387024	...	-0.108300	0.005274	-0.190321	-1.175575	0.647376	-0.221929	0.062723	0.061458	123.50	0.0
4	2	-1.158233	0.877737	1.548718	0.403034	-0.407193	0.095921	0.592941	-0.270533	0.817739	...	-0.009431	0.798278	-0.137458	0.141267	-0.206010	0.502292	0.219422	0.215153	69.99	0.0
5 rows × 31 columns

2]Information of the data
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20414 entries, 0 to 20413
Data columns (total 31 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Time    20414 non-null  int64  
 1   V1      20414 non-null  float64
 2   V2      20414 non-null  float64
 3   V3      20414 non-null  float64
 4   V4      20414 non-null  float64
 5   V5      20414 non-null  float64
 6   V6      20414 non-null  float64
 7   V7      20414 non-null  float64
 8   V8      20414 non-null  float64
 9   V9      20414 non-null  float64
 10  V10     20414 non-null  float64
 11  V11     20414 non-null  float64
 12  V12     20414 non-null  float64
 13  V13     20414 non-null  float64
 14  V14     20414 non-null  float64
 15  V15     20414 non-null  float64
 16  V16     20414 non-null  float64
 17  V17     20414 non-null  float64
 18  V18     20414 non-null  float64
 19  V19     20414 non-null  float64
 20  V20     20414 non-null  float64
 21  V21     20414 non-null  float64
 22  V22     20414 non-null  float64
 23  V23     20414 non-null  float64
 24  V24     20413 non-null  float64
 25  V25     20413 non-null  float64
 26  V26     20413 non-null  float64
 27  V27     20413 non-null  float64
 28  V28     20413 non-null  float64
 29  Amount  20413 non-null  float64
 30  Class   20413 non-null  float64
dtypes: float64(30), int64(1)
memory usage: 4.8 MB

3]Checking for null Values

	0
Time	0
V1	0
V2	0
V3	0
V4	0
V5	0
V6	0
V7	0
V8	0
V9	0
V10	0
V11	0
V12	0
V13	0
V14	0
V15	0
V16	0
V17	0
V18	0
V19	0
V20	0
V21	0
V22	0
V23	0
V24	1
V25	1
V26	1
V27	1
V28	1
Amount	1
Class	1
dtype: int64

4]Distribution of fraudulent and legit transactions in the dataset
count
Class	
0.0	20327
1.0	86
dtype: int64

5]Splitting the data into
--> For X
  Time        V1        V2        V3        V4        V5        V6  \
17660  28842 -0.045645  0.760969  1.831294  0.824596  0.069321 -0.229525   
11803  20253 -0.909521  1.257555  1.322797  0.004725  0.203739  0.347177   
541      406 -2.312227  1.951992 -1.609851  3.997906 -0.522188 -1.426545   
623      472 -3.043541 -3.157307  1.088463  2.288644  1.359805 -1.064823   
4920    4462 -2.303350  1.759247 -0.359745  2.330243 -0.821628 -0.075788   
...      ...       ...       ...       ...       ...       ...       ...   
18466  29526  1.102804  2.829168 -3.932870  4.707691  2.937967 -1.800904   
18472  29531 -1.060676  2.608579 -2.971679  4.360089  3.738853 -2.728395   
18773  29753  0.269614  3.549755 -5.810353  5.809370  1.538808 -2.269219   
18809  29785  0.923764  0.344048 -2.880004  1.721680 -3.019565 -0.639736   
20198  30852 -2.830984  0.885657  1.199930  2.861292  0.321669  0.289966   

             V7        V8        V9  ...       V20       V21       V22  \
17660  0.509600 -0.227317 -0.187402  ...  0.140194  0.157098  0.693093   
11803  0.001500  0.679063  0.375989  ... -0.063859 -0.218769 -0.408103   
541   -2.537387  1.391657 -2.770089  ...  0.126911  0.517232 -0.035049   
623    0.325574 -0.067794 -0.270953  ...  2.102339  0.661696  0.435477   
4920   0.562320 -0.399147 -0.238253  ... -0.430022 -0.294166 -0.932391   
...         ...       ...       ...  ...       ...       ...       ...   
18466  1.672734 -0.300240 -2.783011  ... -0.030880 -0.106994 -0.250050   
18472  1.987616 -0.357345 -2.757535  ... -0.089062 -0.063168 -0.207385   
18773 -0.824203  0.351070 -3.759059  ...  0.310525  0.371121 -0.322290   
18809 -3.801325  1.299096  0.864065  ...  0.170872  0.899931  1.481271   
20198  1.767760 -2.451050  0.069736  ... -1.016923  0.546589  0.334971   

            V23       V24       V25       V26       V27       V28  Amount  
17660 -0.249261  0.056832 -0.071876 -0.267198 -0.055993 -0.164270    9.99  
11803  0.082860 -0.352805 -0.353738  0.074643  0.147512  0.028380    1.98  
541   -0.465211  0.320198  0.044519  0.177840  0.261145 -0.143276    0.00  
623    1.375966 -0.293803  0.279798 -0.145362 -0.252773  0.035764  529.00  
4920   0.172726 -0.087330 -0.156114 -0.542628  0.039566 -0.153029  239.93  
...         ...       ...       ...       ...       ...       ...     ...  
18466 -0.521627 -0.448950  1.291646  0.516327  0.009146  0.153318    0.68  
18472 -0.183261 -0.103679  0.896178  0.407387 -0.130918  0.192177    0.68  
18773 -0.549856 -0.520629  1.378210  0.564714  0.553255  0.402400    0.68  
18809  0.725266  0.176960 -1.815638 -0.536517  0.489035 -0.049729   30.30  
20198  0.172106  0.623590 -0.527114 -0.079215 -2.532445  0.311177  104.81  

[88 rows x 30 columns]

--> For Y
17660    0.0
11803    0.0
541      1.0
623      1.0
4920     1.0
        ... 
18466    1.0
18472    1.0
18773    1.0
18809    1.0
20198    1.0
Name: Class, Length: 88, dtype: float64

6] Accuracy Score
Accuracy on Training Data: 1.0
Accuracy score on Test data: 0.9629629629629629
