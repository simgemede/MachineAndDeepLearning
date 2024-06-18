import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# output = 1 diabetic
# output = 0 healthy
data = pd.read_csv("diabetes.csv")
data.head()
"""print(data)

diabetic = data[data.Outcome == 1]
healthy = data[data.Outcome == 0]

plt.scatter(healthy.Age, healthy.Glucose, color="green", label="healthy", alpha=0.4)
plt.scatter(diabetic.Age, diabetic.Glucose, color="red", label="diabetic", alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend()
plt.show()"""

y = data.Outcome.values
x_raw_data = data.drop(["Outcome"], axis = 1)

x = (x_raw_data - np.min(x_raw_data))/(np.max(x_raw_data) - np.min(x_raw_data))

"""print("Raw data before normalization:\n")
print(x_raw_data.head())

print("\n\n Data after normalization: \n")
print(x.head())"""

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("Verification test result of our test data for k=3:", knn.score(x_test, y_test))