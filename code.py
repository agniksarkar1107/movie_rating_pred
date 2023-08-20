import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
data_movie=pd.read_csv("movies.csv",sep='::',engine='python')
data_movie.columns =['MovieIDs','MovieName','Category']
data_movie.dropna(inplace=True)
data_movie.head()
data_rating = pd.read_csv("ratings.csv",sep='::', engine='python')
data_rating.columns =['ID','MovieID','Ratings','TimeStamp']
data_rating.dropna(inplace=True)
data_rating.head()
data_user = pd.read_csv("users.csv",sep='::',engine='python')
data_user.columns =['UserID','Gender','Age','Occupation','Zip-code']
data_user.dropna(inplace=True)
data_user.head()
data = pd.concat([data_movie, data_rating,data_user], axis=1)
data.head()
features = ["MovieID","Age","Occupation"]
#Use rating as label
labels = data['Ratings']
#Create train and test data set
x_train, x_test, y_train, y_test = train_test_split(data,labels,test_size=0.5)

len(x_train)
print(y_train)
data.dropna(inplace=True)
features = ["MovieID","Age","Occupation"]
y_train=data["Ratings"]
x_train = pd.get_dummies(data[features])
x_test = pd.get_dummies(data[features])
my_model= RandomForestClassifier(n_estimators=100)
my_model.fit(x_train,y_train)
Y_pred = my_model.predict(x_test)

output = pd.DataFrame({'MovieName':data.MovieName, 'Ratings': Y_pred})
#output.to_csv('movie_predict_model.csv', index=False)
#print("The file is created")
print(output)
output.to_csv('movie_predict_model.csv', index=False)
