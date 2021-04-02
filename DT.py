import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import tree

name = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12'
        ,'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24'
        ,'V25', 'V26', 'V27', 'V28', 'Amount', 'Lable']

data = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12'
        ,'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24'
        ,'V25', 'V26', 'V27', 'V28', 'Amount']


ds = pd.read_csv(r"C:\Users\User\OneDrive - KMITL\Documents\credit_normalize.csv", header=0, names=name)


x = ds[data]
y = ds.Lable
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.3,random_state=0)
model = DecisionTreeClassifier()
model = model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
# print(y_pred)
# confusion_matrix(Y_test, y_pred)
# print(confusion_matrix(Y_test, y_pred))

# accuracy
print("Accuracy:", 100*(metrics.accuracy_score(Y_test, y_pred)))

