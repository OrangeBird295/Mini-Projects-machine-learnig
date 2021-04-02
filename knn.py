import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

name = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12'
        ,'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24'
        ,'V25', 'V26', 'V27', 'V28', 'Amount', 'Lable']

data = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12'
        ,'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24'
        ,'V25', 'V26', 'V27', 'V28', 'Amount']


ds = pd.read_csv(r"C:\Users\User\OneDrive - KMITL\Documents\credit_normalize.csv", header=0, names=name)

x = ds[data]
y = ds.Lable

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)


# from sklearn.metrics import classification_report
# print(classification_report(y_test, answer))

print("Accuracy:", 100*(metrics.accuracy_score(y_test, y_pred)))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train,y_train)#ฝึกโมเดล
#ทดสอบโมเดล
y_pred = knn.predict(X_test)
print("Accuracy:", 100*(metrics.accuracy_score(y_test, y_pred)))

