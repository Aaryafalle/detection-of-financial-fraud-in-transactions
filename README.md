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

5]Splitting the data

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
