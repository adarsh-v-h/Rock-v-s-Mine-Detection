# We have something called SONAR used in submarines,
# where the data from that SONAR is used to tell if the object detected by the SONAR is a rock or a mine
# kinda military use case.... 
# work flow : The Sonar data from lab is collected -> Then the data is undergone some preprocessing
# -> now we will split our data to train data and test data -> this data is feed into a model(for this case we will use Logistic Regression model)
# -> then we will take this trained model and take some real world or new data that the model is not trained with and we check how accurate it is...

import pandas as pd # for dataframe
import numpy as np # for data handling
from sklearn.model_selection import train_test_split # for spliting the data
from sklearn.linear_model import LogisticRegression # the model, this is good for 2 class
from sklearn.metrics import accuracy_score # to find out the accuracy of the model
import os
def main():
    print("cwd: ", os.getcwd())
    print("Start to initialize data")
    Sonar_dataframe = pd.read_csv("Copy of sonar data.csv", header=None)
    # our dataset has not header, we have to let pandas know that, so that it doesnt confuse the actual first row of data as header
    # now pandas has assigned indexes to all the columns(like it used to do for rows)

    print(f"data set consistes of {Sonar_dataframe.shape[0]} rows and {Sonar_dataframe.shape[1]}")
    X = Sonar_dataframe.drop(columns=60, axis=1) # except the last column() which has the labelled data
    Y = Sonar_dataframe[60] # here we will store the result

    # now spliting data our data set for training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.15, stratify=Y, random_state=3)
    # 15% for test, stratify=Y makes sure the proportion Y had before spliting(the proportion of labels(0 or l)) stays the same even after split, both in Y_train and Y_test
    print("Data initialization done")

    # Now lets start the work with the model
    print("Start to train the model")
    model_LR = LogisticRegression()
    # LogisticRegression is a class here, but in theory its actually a model type
    model_LR.fit(X_train,Y_train)
    # here we are done with the training part, this where the parameters of the model are determined based on X_train and Y_train
    # better the data(bigger in size and more accurate), more accurate the parameters will be
    print("Done with training")

    # now we will test the model's accuracy
    X_test_predict = model_LR.predict(X_test)
    # Now we are predicting the Y_test values with X_test, the parameters of the model mainly do that

    # after predicting we compare that with actual Y_test value
    acc_score = accuracy_score(X_test_predict, Y_test)
    print(f"The accuracy of the model is {acc_score*100}%")

    print("Done with testing")

if __name__ == "__main__":
    main()