#Machine learning mini project

import pandas as pd
import sklearn.tree as dtree
import sklearn.neighbors as knn
import sklearn.metrics as ac

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Load Dataset
df=pd.read_csv('C:\\Users\\kiman\\Downloads\\diabetes.csv')


# Features and Target
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split Dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.50,
    random_state=1
)

# Create KNN Model
knnx = knn.KNeighborsClassifier(n_neighbors=5)

# Train Model
knnx.fit(x_train, y_train)

# Predict
y_pred = knnx.predict(x_test)

# Accuracy
a1 = ac.accuracy_score(y_pred, y_test)

# Confusion Matrix
c1 = ac.confusion_matrix(y_pred, y_test)

cv1 = cross_val_score(knnx, x, y, cv=2)

print("The KNN Results are: ")
print("Accuracy:\t", a1)
print("Confusion Matrix:\n", c1)
print("2-Fold Cross Validation:", cv1.mean())

#DECISION TREES
# Create Decision Tree
dt = dtree.DecisionTreeClassifier()

# Train Model
dt.fit(x_train, y_train)

# Predict
y_pred = dt.predict(x_test)

# Accuracy
a2 = ac.accuracy_score(y_pred, y_test)

# Confusion Matrix
c2 = ac.confusion_matrix(y_pred, y_test)

cv2 = cross_val_score(dt, x, y, cv=2)

# Display Tree Rules
rules = dtree.export_text(dt, feature_names=list(x.columns))

print("Decision Tree Results")
print("Accuracy:\t", a2)
print("Confusion Matrix:\n", c2)
print("2-Fold Cross Validation:", cv2)
print("Average CV Accuracy:", cv2.mean())
print("Decision Tree Rules:\n")
print(rules)

#ANN

ann = MLPClassifier(
     hidden_layer_sizes=(10,),
     max_iter=1000,
     random_state=1
    )
ann.fit(x_train, y_train)

y_pred = ann.predict(x_test)

a3 = ac.accuracy_score(y_pred, y_test)

c3 = ac.confusion_matrix(y_pred, y_test)

cv3 = cross_val_score(ann, x, y, cv=2)

print("ANN Results")    
print("Accuracy:", a3)
print("confusion_matrix:", c3)
print("2-Fold Cross Validation:", cv3)
print("Average cv Accuracy:", cv3.mean())


#COMPARISON OF ALL MODELS
print("\nCOMPARISON OF THE MODELS")
print("KNN Accuracy           :", a1)
print("Decision Tree Accuracy :", a2)
print("ANN Accuracy           :", a3)


print("\nAverage 2-Fold Cross Validation Accuracy")
print("KNN           :", cv1.mean())
print("Decision Tree :", cv2.mean())
print("ANN           :", cv3.mean())