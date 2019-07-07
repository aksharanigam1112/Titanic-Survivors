# Step1 importing libraries

import numpy as np
import pandas as pd

pd.set_option('display.width', 1000)
pd.set_option('display.max_column', 16)
pd.set_option('precision', 2)

import matplotlib.pyplot as plt
import seaborn as sbn
import warnings

warnings.filterwarnings(action="ignore")



# Step2 Reading data



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Take a look at the training data
print(train.describe())
print("\n", train.describe(include='all'), "\n")



# Step 3 Data Analysis


# We're going to consider the features in the
# dataset and how complete they are.

# See a sample of the dataset to get an idea of the variables
print("\n", train.columns)
print("\n", train.head())
print("\n", train.sample(5))

print("\nData type of each feature : ", train.dtypes)

# see a summary of the training dataset
print(train.describe(include="all"))

print("\n", pd.isnull(train).sum())  # To know the cont of the missing data



# Step 4 Data visualization



# 4 A) Sex Feature


# Draw a bar plot of survival by sex
sbn.barplot(x='Sex', y='Survived', data=train)
plt.show()

print("\n", train['Survived'][train['Sex'] == 'female'])
print("\n", train['Survived'][train['Sex'] == 'female'].value_counts())
# Counts the female passengers who survived
print("\n", train['Survived'][train['Sex'] == 'female'].value_counts(normalize=True)[1])
# Gives the normalized value(0-1) based on the survival of female passengers

print("Percentage of females who survived:",train["Survived"][train["Sex"] == 'female'].value_counts(normalize=True)[1] * 100)
print("Percentage of males who survived:",train["Survived"][train["Sex"] == 'male'].value_counts(normalize=True)[1] * 100)



# B) Pclass Feature


sbn.barplot(x='Pclass', y='Survived', data=train)
plt.show()
print("\nPercentage of Pclass 1 who survived : ",train['Survived'][train['Pclass'] == 1].value_counts(normalize=True)[1] * 100)
print("\nPercentage of Pclass 2 who survived : ",train['Survived'][train['Pclass'] == 2].value_counts(normalize=True)[1] * 100)
print("\nPercentage of Pclass 3 who survived : ",train['Survived'][train['Pclass'] == 3].value_counts(normalize=True)[1] * 100)



# C) SibSp Feature


print("Percentage of SibSp = 0 who survived:",train["Survived"][train["SibSp"] == 0].value_counts(normalize=True)[1] * 100)
print("Percentage of SibSp = 1 who survived:",train["Survived"][train["SibSp"] == 1].value_counts(normalize=True)[1] * 100)
print("Percentage of SibSp = 2 who survived:",train["Survived"][train["SibSp"] == 2].value_counts(normalize=True)[1] * 100)

sbn.barplot(x='SibSp', y='Survived', data=train)
plt.show()



# D) Parch Feature


sbn.barplot(x='Parch', y='Survived', data=train)
plt.show()



# E) Age Feature


train['Age'] = train['Age'].fillna(-0.5)
test['Age'] = test['Age'].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['Agegroup'] = pd.cut(train['Age'], bins, labels=labels)
test['Agegroup'] = pd.cut(test['Age'], bins, labels=labels)
print(train.sample(10))

sbn.barplot(x='Agegroup', y='Survived', data=train)
plt.show()



# F) Cabin Feature



train['CabinBool'] = (train['Cabin'].notnull().astype('int'))
test['CabinBool'] = (test['Cabin'].notnull().astype('int'))

print(train.sample(10))
print("\nPercentage of CabinBool = 1 who survived:",train["Survived"][train["CabinBool"] == 1].value_counts(normalize=True)[1] * 100)
print("\nPercentage of CabinBool = 0 who survived:",train["Survived"][train["CabinBool"] == 0].value_counts(normalize=True)[1] * 100)

sbn.barplot(x='CabinBool', y='Survived', data=train)
plt.show()



# Cleaning Further unessesary columns



print(test.describe(include="all"))

train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)

train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)



# Embarked Feature



shmt = train[train['Embarked'] == 'S'].shape[0]
print("\nPeople who boarded from southampton", shmt)

chbg = train[train['Embarked'] == 'C'].shape[0]
print("\nPeople who boarded from southampton", chbg)

qtwn = train[train['Embarked'] == 'Q'].shape[0]
print("\nPeople who boarded from southampton", qtwn)

#  After observing from this data we realize max people boarded from Southampton
# So we fill the missing values with this

train = train.fillna({'Embarked': 'S'})



# Agegroup feature



# Filling unknown values according to the salutation 
# If salutation is Mr. or anything similar we fill with mode value

combine = [train, test]
print(combine[0])

# extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset['Name'].str.extract(', ([A-Za-z]+)\.', expand=False)

print("\n\n","~"*100,"\n\n")
print(train.sample(10))


print(pd.crosstab(train['Title'], train['Sex']))

# replace various titles with more common names
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'],'Rare')

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print("\n\nAfter grouping rare title : \n", train)

print(train[['Title', 'Survived']].groupby(['Title'],as_index=True).count())

print("\nMap each of the title groups to a numerical value.")
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}

for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

print("\n\nAfter replacing title with neumeric values.\n")
print(train)

# Next, we'll try to predict the missing Age values from the most common age for their Title.

# fill missing age with mode age group for each title
mr_age = train[train["Title"] == 1]["Agegroup"].mode()  # Mr.= Young Adult
print("mode() of mr_age : ", mr_age)
print("\n\n")

miss_age = train[train["Title"] == 2]["Agegroup"].mode()  # Miss.= Student
print("mode() of miss_age : ", miss_age)
print("\n\n")

mrs_age = train[train["Title"] == 3]["Agegroup"].mode()  # Mrs.= Adult
print("mode() of mrs_age : ", mrs_age)
print("\n\n")

master_age = train[train["Title"] == 4]["Agegroup"].mode()  # Baby
print("mode() of master_age : ", master_age)
print("\n\n")

royal_age = train[train["Title"] == 5]["Agegroup"].mode()  # Adult
print("mode() of royal_age : ", royal_age)
print("\n\n")

rare_age = train[train["Title"] == 6]["Agegroup"].mode()  # Adult
print("mode() of rare_age : ", rare_age)

print(train.describe(include="all"))
print(train)

print("\n\ntrain[Agegroup][0] :  \n\n")

for x in range(10):
    print(train["Agegroup"][x])

age_title_mapping = {1: "Young Adult", 2: "Student",3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

for x in range(len(train["Agegroup"])):
    if train["Agegroup"][x] == "Unknown":  # x=5 ( means for 6th record )
        train["Agegroup"][x] = age_title_mapping[train["Title"][x]]

for x in range(len(test["Agegroup"])):
    if test["Agegroup"][x] == "Unknown":
        test["Agegroup"][x] = age_title_mapping[test["Title"][x]]

print("\n\nAfter replacing Unknown values from AgeGroup column : \n")
print(train.sample(10))

# map each Age value to a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3,'Student': 4, 'Young Adult': 5,'Adult': 6, 'Senior': 7}

train['Agegroup'] = train['Agegroup'].map(age_mapping)
test['Agegroup'] = test['Agegroup'].map(age_mapping)
print("\n\n",train.sapmle(10))



# dropping the Age & Name features for now, might change

train = train.drop(['Age'], axis=1)
test = test.drop(['Age'], axis=1)

print("\n\nAge column dropped.\n\n")
print(train.sample(10))

train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)



# Mapping sex feature value



sex_mapping = {'male': 0, 'female': 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

print("\nSex feature: \n\n", train.sample(10))



# Mapping embarked feature



embarked_mapping = {'S': 0, 'C': 2, 'Q': 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)
print(train.sample(10))



# Fare Feature



# fill in missing Fare value in test set based on mean fare for that Pclass
for x in range(len(test['Fare'])):
    if (pd.isnull(test['Fare'][x])):
        pclass = test['Pclass'][x]
        test['Fare'][x] = round(train[train['Pclass'] == pclass]['Fare'].mean(), 2)


train['Fareband'] = pd.qcut(train['Fare'], 4, labels=[1, 2, 3, 4])
test['Fareband'] = pd.qcut(test['Fare'], 4, labels=[1, 2, 3, 4])


train = train.drop(['Fare'], axis=1)
test = test.drop(['Fare'], axis=1)
print("\n\nFare Column Dropped\n\n", train.head(10))



# Check test data

print("\n\nTest data checking:\n\n", test.head(10))



# Step 6) Choosing the best model



# We use 20% of historic data as testing data

from sklearn.model_selection import train_test_split

input = train.drop(['Survived', 'PassengerId'], axis=1)
output = train['Survived']

xtrain, xtest, ytrain, ytest = train_test_split(input, output, test_size=0.20, random_state=7)



# Testing Different Models

# We will be testing the following models with my training data (got the list from here):

# 1) Logistic Regression
# 2) Gaussian Naive Bayes
# 3) Support Vector Machines
# 4) Linear SVC
# 5) Perceptron
# 6) Decision Tree Classifier
# 7) Random Forest Classifier
# 8) KNN or k-Nearest Neighbors
# 9) Stochastic Gradient Descent
# 10) Gradient Boosting Classifier
# 11) Linear Discriminant Analysis


from sklearn.metrics import accuracy_score



# 1) LOGISTIC REGRESSIION



from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression()
model1.fit(xtrain, ytrain)
y_pred = model1.predict(xtest)
acc1 = round(accuracy_score(y_pred, ytest) * 100, 2)
print("\nModel 1 Logistic Regression : ", acc1)



# 2) GAUSSIAN NAIVE BAYES



from sklearn.naive_bayes import GaussianNB

model2 = GaussianNB()
model2.fit(xtrain, ytrain)
y_pred = model2.predict(xtest)
acc2 = round(accuracy_score(y_pred, ytest) * 100, 2)
print("\nModel 2 Gaussian Naive Bayes : ", acc2)



# 3) SVM



from sklearn.svm import SVC

model3 = SVC()
model3.fit(xtrain, ytrain)
y_pred = model3.predict(xtest)
acc3 = round(accuracy_score(y_pred, ytest) * 100, 2)
print("\nModel 3 SVM : ", acc3)



# 4) LINEAR SVC



from sklearn.svm import LinearSVC

model4 = LinearSVC()
model4.fit(xtrain, ytrain)
y_pred = model4.predict(xtest)
acc4 = round(accuracy_score(y_pred, ytest) * 100, 2)
print("\nModel 4 Linear SVC : ", acc4)



# 5) Perceptron



from sklearn.linear_model import Perceptron

model5 = Perceptron()
model5.fit(xtrain, ytrain)
y_pred = model5.predict(xtest)
acc5 = round(accuracy_score(y_pred, ytest) * 100, 2)
print("\nModel 5 Perceptron : ", acc5)



# 6) Decision Tree Classifier



from sklearn.tree import DecisionTreeClassifier

model6 = DecisionTreeClassifier()
model6.fit(xtrain, ytrain)
y_pred = model6.predict(xtest)
acc6 = round(accuracy_score(y_pred, ytest) * 100, 2)
print("\nModel 6 CART : ", acc6)



# 7) Random Forest



from sklearn.ensemble import RandomForestClassifier

model7 = RandomForestClassifier()
model7.fit(xtrain, ytrain)
y_pred = model7.predict(xtest)
acc7 = round(accuracy_score(y_pred, ytest) * 100, 2)
print("\nModel 7 Random Forest : ", acc7)



# 8) KNN



from sklearn.neighbors import KNeighborsClassifier

model8 = KNeighborsClassifier()
model8.fit(xtrain, ytrain)
y_pred = model8.predict(xtest)
acc8 = round(accuracy_score(y_pred, ytest) * 100, 2)
print("\nModel 8 KNN : ", acc8)



# 9) Stochastic Gradient Descent



from sklearn.linear_model import SGDClassifier

model9 = SGDClassifier()
model9.fit(xtrain, ytrain)
y_pred = model9.predict(xtest)
acc9 = round(accuracy_score(y_pred, ytest) * 100, 2)
print("\nModel 9 Stochastic Gradient Descent : ", acc9)



# 10) Gradient Boosting Classifier



from sklearn.ensemble import GradientBoostingClassifier

model10 = GradientBoostingClassifier()
model10.fit(xtrain, ytrain)
y_pred = model10.predict(xtest)
acc10 = round(accuracy_score(y_pred, ytest) * 100, 2)
print("\nModel 10 Gradient Boosting Claasifier : ", acc10)



# 11) Linear Discriminant Analysis



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model11 = LinearDiscriminantAnalysis()
model11.fit(xtrain, ytrain)
y_pred = model11.predict(xtest)
acc11 = round(accuracy_score(y_pred, ytest) * 100, 2)
print("\nModel 11 Linear Discriminant Analysis : ", acc11)




# Comparing accuracies of each model



models = pd.DataFrame({
    'Model': ['Logistic Regression', 'GaussianNB', 'SVM',
              'Linear SVC', 'Perceptron', 'CART',
              'Random Forest', 'KNN', 'Stochastic Gradient Descent',
              'Gradient Boosting Classifier', 'LDA'],
    'Score': [acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc8, acc10, acc11]

})

print("\n", models.sort_values(by='Score', ascending=False))

# According to the report we find that Gradient Boosting Classifier has the best accuracy




# Step 7) Cresting submission.csv

ids = test['PassengerId']
pred = model10.predict(test.drop(['PassengerId'], axis=1))

output = pd.DataFrame({'PassengerId': ids, 'Survived': pred})
output.to_csv('submission.csv', index=False)

print("\nAll Survival predictions done !!")
print(output)
