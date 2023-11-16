""" --------------------Fraud Transaction Detection with LogisticRegression --------------------------"""

# import required library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#%%
# Load creditcard dataset from Kaggle [link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud]

credit_card_data = pd.read_csv('C:\\Users\\Salim Raza\\\Desktop\\UTEP Research\\Code\\Fraud Transaction Detection (FTD)\\creditcard.csv')
print(credit_card_data.head()) # First 5 rows of the dataset
print(credit_card_data.tail()) # Last 5 rows of the dataset

#%% 
# Shows sahape of dataset 

credit_card_data.info() # Dataset information
print(credit_card_data.isnull()) # Checking missing values as a True and False
print(credit_card_data.isnull().sum()) #Suming  the True values (Missing values) in each column  
print(credit_card_data['Class'].value_counts()) # Showing distribution of transactions in terms of Class

#%% 
# Separating the data for analysis

legit = credit_card_data[credit_card_data['Class'] == 0]
fraud = credit_card_data[credit_card_data['Class'] == 1]
print(legit.shape)
print(fraud.shape)

#%% 
# Preprocessing dataset to fit them with the model (1)

print(legit['Amount'].describe()) # Statistical measures of legit the dataset
print(fraud['Amount'].describe()) # Statistical measures of fraud the dataset
print(credit_card_data.groupby('Class').mean())  # Showig mean values for both class for each feature

#%% 
#Preprocessing dataset to fit them with the model (2)

legit_sample = legit.sample(n=492)  # Select 492 random rows from the legit, this is equal to fraud dataset
new_dataset = pd.concat([legit_sample, fraud], axis=0) # Combines the two datasets (legit+fraud) vertically

print(new_dataset.head())
print(new_dataset.tail())

print(new_dataset['Class'].value_counts())
print(new_dataset.groupby('Class').mean())

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

print(X)
print(Y)

#%% 
#Preprocessing dataset to fit them with the model (3)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#%% 
# Modle Selection for Classification from sklearn and Training 

model = LogisticRegression()
model.fit(X_train, Y_train)# Training the Logistic Regression Model with Training Data

#%% 
# Accuracy Measurement on Training Data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data:', training_data_accuracy)

#%% 
# Accuracy Measurement on Test Data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score on Test Data:', test_data_accuracy)

""" ------------------------------------------- End-----------------------------------------------------------"""