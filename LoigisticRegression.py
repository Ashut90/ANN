
#Logistic Regression 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV

from sklearn.datasets import load_breast_cancer 
df = load_breast_cancer()
#Indipendent feature
X = pd.DataFrame(df['data'], columns = df["feature_names"])

X.head() #show up the indipendent feature 

#dependent feature 

y = pd.DataFrame(df['target'],columns = ["Target"])
y      #show up the dependent feature 


#check if dataset is balenced or not 

y['Target'].value_counts()

#train test split 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.33, random_state=42)

params = [{'C':[1,5,10]} ,{'max_iter':[100,150]}]

model1 = LogisticRegression(C = 100 , max_iter = 100)

model=GridSearchCV(model1, param_grid = params, scoring = 'f1', cv = 5)

model.fit(X_train , y_train)

model.best_params_

model.best_score_

model.predict(X_test)

y_pred = model.predict(X_test)

y_pred

from sklearn.metrics import confusion_matrix, classification_report,accuracy_score 

confusion_matrix(y_test, y_pred)

accuracy_score(y_test , y_pred)

print(classification_report(y_test,y_pred))


