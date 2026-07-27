import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# read iris dataset with python
iris = pd.read_csv("iris.csv")

# define features (x) and target (y)
x = iris[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
y = iris["variety"]

# call train_test_split function to split the data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=100
)

# use kneighborsclassifier to find the optimal k value
scores = []

for k in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    scores.append(knn.score(x_test, y_test))

top_score = max(scores)
optimal_k = scores.index(top_score) + 1

print("Optimal k value:", optimal_k)