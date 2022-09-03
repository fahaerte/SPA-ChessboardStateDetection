import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from joblib import dump, load

col_names = ['white', 'black', 'blue', 'yellow', 'label']
data = pd.read_csv("decTreeData.csv", sep=";", header=None, names=col_names)
feature_cols = ['white', 'black', 'blue', 'yellow']
X = data[feature_cols]  # Features
y = data.label  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
model = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = model.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

dump(model, 'decision_tree.joblib')

text_representation = tree.export_text(model, feature_names=feature_cols)
print(text_representation)

print(X_test)
