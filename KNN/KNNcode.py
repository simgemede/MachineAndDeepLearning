import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# çıktı = 1 şeker hastası
# çıktı = 0 sağlıklı
data = pd.read_csv("diabetes.csv")
data.head()
print(data)

seker_hastaları = data[data.Outcome == 1]
saglikli_insanlar = data[data.Outcome == 0]

plt.scatter(saglikli_insanlar.Age, saglikli_insanlar.Glucose, color="green", label="sağlıklı", alpha=0.4)
plt.scatter(seker_hastaları.Age, seker_hastaları.Glucose, color="red", label="şeker hastası", alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend()
plt.show()