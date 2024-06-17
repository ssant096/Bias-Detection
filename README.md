# Bias Detection in Diabetes Dataset
## Project Problem
The goal of our project is to build a model to predict whether a person is diabetic or prediabetic and to identify any biases in the model that could compromise protected classes. Our dataset includes the protected classes of age and sex. We will also look to identify biases in other categories that may unfairly penalize a person, such as education and income. If we encounter bias in our model, we plan to make attempts to correct it.

## Data Description
Our data came clean with no missing values. The dataset, sourced from the UCI Machine Learning Repository, contains 253,680 rows and 22 features, including 14 binary variables, 7 categorical variables, and 1 response variable.

### Binary Variables
- **HighBP**: 0 = no high BP, 1 = high BP
- **HighChol**: 0 = no high cholesterol, 1 = high cholesterol
- **CholCheck**: 0 = no cholesterol check in 5 years, 1 = yes cholesterol check in 5 years
- **Smoker**: Have you smoked at least 100 cigarettes in your life? 0 = no, 1 = yes
- **Stroke**: Ever had a stroke? 0 = no, 1 = yes
- **HeartDiseaseorAttack**: Coronary heart disease (CHD) or myocardial infarction (MI)? 0 = no, 1 = yes
- **PhysActivity**: Physical activity in the past 30 days (excluding job)? 0 = no, 1 = yes
- **Fruits**: Consume fruit 1 or more times per day? 0 = no, 1 = yes
- **Veggies**: Consume vegetables 1 or more times per day? 0 = no, 1 = yes
- **HvyAlcoholConsump**: Heavy drinkers (adult men > 14 drinks/week and women > 7 drinks/week)? 0 = no, 1 = yes
- **AnyHealthcare**: Have any health care coverage? 0 = no, 1 = yes
- **NoDocbcCost**: Needed to see a doctor but couldn't because of cost in the past 12 months? 0 = no, 1 = yes
- **DiffWall**: Serious difficulty walking or climbing stairs? 0 = no, 1 = yes
- **Sex**: 0 = female, 1 = male

### Categorical Variables
- **BMI**: Body Mass Index
- **GenHlth**: General health (1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor)
- **MentHlth**: Days mental health was not good in the past 30 days (scale 1-30 days)
- **PhysHlth**: Days physical health was not good in the past 30 days (scale 1-30 days)
- **Age**: Age categories (1 = 18-24, 9 = 60-64, 13 = 80 or older)
- **Education**: Education level (1 = Never attended school/kindergarten, 2 = Grades 1-8, 3 = Grades 9-11, 4 = Grade 12/GED, 5 = College 1-3 years, 6 = College 4+ years)
- **Income**: Income categories (1 = less than $10,000, 5 = less than $35,000, 8 = $75,000 or more)

### Response Variable
- **Diabetes_binary**: 0 = no diabetes, 1 = prediabetes or diabetes

## Exploratory Data Analysis (EDA)
![image](https://github.com/ssant096/Bias-Detection/assets/102336530/82c2f719-1fb9-452a-9363-04acc5b350cc)

### Categorical Variables
- **Skewed Right**: BMI, Mental Health, Physical Health
- **Skewed Left**: Education, Income
- **Normally Distributed**: Age, General Health
- Slightly more females than males.

![image](https://github.com/ssant096/Bias-Detection/assets/102336530/e1b37874-36cd-4cfa-9538-594d7818203b)

### Binary Variables
- Roughly even distribution: High blood pressure, high cholesterol, smoked at least 100 cigarettes, and males.
- More people: Physical activity outside of work in the past 30 days, eat fruit 1+ times/day, eat vegetables 1+ times/day.
- Majority: Checked cholesterol in the past 5 years, have health care coverage.
- Minority: Had a stroke, CHD or MI, heavy drinkers, couldn’t afford to see a doctor, serious difficulty walking/climbing stairs.

![image](https://github.com/ssant096/Bias-Detection/assets/102336530/e5d2aabf-cfff-4752-b865-46c4d8dcfe4b)


## Ethical Considerations
Our dataset includes sensitive information that may contain bias (Age, Sex, Education, and Income). After building the model, we will test it for fairness in these protected categories.

## Approach
Since our data is mostly categorical, we will build a logistic regression model. We first one-hot encoded our categorical data to binary to increase model performance.

![image](https://github.com/ssant096/Bias-Detection/assets/102336530/f5eb40dc-a840-4ed5-9b87-c2031c62d521)

We then built our model and ran it with an 80:20 train-test split. 
![image](https://github.com/ssant096/Bias-Detection/assets/102336530/fd7566bb-562b-44f1-83f5-cd49f6aa10de)

![image](https://github.com/ssant096/Bias-Detection/assets/102336530/ff3ec4b1-0093-42d6-a354-cead520ff6eb)

![image](https://github.com/ssant096/Bias-Detection/assets/102336530/50ba4f3a-c26c-452d-89ba-e3a563173eec)

### Initial Model Results
- **Training Accuracy**: 86.1%
- **Testing Accuracy**: 85.9%
- **True Positive Rate (TPR)**: 0.1025

### Challenges
The TPR was only 10.25%, indicating poor performance in identifying those with diabetes. We tried different models (decision tree, Multinomial Naive Bayes) but TPR remained unsatisfactory (0.1700 for decision tree, 0.3898 for Multinomial Naive Bayes). 
![image](https://github.com/ssant096/Bias-Detection/assets/102336530/63aeb62e-301a-41bf-8387-b631aba422bd)

### Resampling Approach
We undersampled the non-diabetic rows to create a balanced dataset and re-ran logistic regression.
![image](https://github.com/ssant096/Bias-Detection/assets/102336530/ec05f402-7ad1-40e2-9dfe-aeaf59b6c67d)

- **New Dataset Size**: 70,692 rows
- **Training Accuracy**: 73.9%
- **Testing Accuracy**: 73.8%
- **TPR**: 0.7674

## Analyzing Fairness and Bias
We used the Aequitas python library to analyze fairness, focusing on age, sex, education, and income. We created a dataset compatible with Aequitas and added label_value and score columns for actual and predicted results, respectively.

![image](https://github.com/ssant096/Bias-Detection/assets/102336530/ffe1d3b2-0610-46e8-9456-c5c7a2138143)

### Fairness Metrics of Unchanged Model

![image](https://github.com/ssant096/Bias-Detection/assets/102336530/4859113b-ec21-4c5c-ad57-cc8c9ed13903)

- **Unfair**: Age, Education, Income (failed parity tests for FNR)
- **Fair**: Sex (passed parity tests)

### Attempted Bias Mitigation
We tried dropping columns (Age, Education, Income) individually and together, but bias remained.

![image](https://github.com/ssant096/Bias-Detection/assets/102336530/adc04088-9025-4ad2-a982-15d5a707857d)

![image](https://github.com/ssant096/Bias-Detection/assets/102336530/e3bd31d7-a407-4ac1-96d5-3b13ec8ccdf3)

![image](https://github.com/ssant096/Bias-Detection/assets/102336530/0d3e72e7-013d-4cf4-8066-ff3610012045)

![image](https://github.com/ssant096/Bias-Detection/assets/102336530/e86af79b-d36b-4752-8515-00b3e7fe85fb)

## Closing Thoughts
We could not remove the bias from our model. It is possible that the data itself, rather than the model, was biased. Age, education, and income are important factors in diabetes risk. Dropping variables is not a rigorous enough approach to guarantee fairness. Further tuning of the model may be needed to address bias effectively.
