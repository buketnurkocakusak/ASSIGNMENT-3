import io
import requests

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


"""
As a result of the code block below is accuracy is found suppliying error rate of 10%. Knn only works with labels not with
linear numbers. When f(5)) = 100, estimates that are between 90 <  f(5) < 110  would be acceptable. 
The get_y_val() method can only be applied to have predicted y value for supplied x. It should be called when classified is applied.

"""


def get_y_val(knn_model,x_value):
    return knn_model.predict(np.array(x_value).reshape(-1,1))[0]

# The exact path of the csv file is given, content of it is supplied to getUrlContent object
url = "https://raw.githubusercontent.com/boragungoren-portakalteknoloji/ATILIM-ECON484-Spring2022/main/Homework%20Assignments/HW3%20-%20First%20KNN%20Model/Training%20Data%20with%20Outliers.csv"
getUrlContent = requests.get(url).content
#Decoded to "utf-8" format.
df = pd.read_csv(io.StringIO(getUrlContent.decode('utf-8')))
x_data = np.array(df["X"].tolist()).reshape(-1,1)
y_data = np.array(df["Y"].tolist()).reshape(-1,1)
#Values have been converted to list.
listofValues = df.values.tolist()
#print(listofValues)

url = "https://raw.githubusercontent.com/boragungoren-portakalteknoloji/ATILIM-ECON484-Spring2022/main/Homework%20Assignments/HW3%20-%20First%20KNN%20Model/Validation%20Data.csv"
getUrlContent = requests.get(url).content
#Url content has been decoded and separated by semi column.
df = pd.read_csv(io.StringIO(getUrlContent.decode('utf-8')), sep=";")
x_val = np.array(df["X"].tolist()).reshape(-1,1)
y_val = np.array(df["Y"].tolist()).reshape(-1,1)
#Values have been converted to list.
ValidatedList = df.values.tolist()
#print(ValidatedList)


# Knn algorith is applied in a loop that k is 1-10
# The best accuracy is resulted as 7, so that 7 for k is chosen.
classified = KNeighborsClassifier(n_neighbors=7)

# Classified applied by having training sets
classified.fit(x_data,y_data.ravel())
#Predicted Result
prediction = classified.predict(x_val)
# It is possible to call get_y_val() function after these line
#print(get_y_val(classified,x_value))
print("Guesssed f(5) is : ".format(classified,5))
error_rate = 0.1
count = 0
#For loop in prediction result applied, item is the element of the range. count is incremented by 1 when the if condition is supported.
for item in range(len(prediction)):
    if (y_val[item] + y_val[item] * error_rate) > prediction[item] > (y_val[item] - y_val[item] * error_rate):
        count +=1

print("Accuracy is {}".format(count / len(prediction)))
