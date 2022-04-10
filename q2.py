from openpyxl import load_workbook
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Read data from excel and convert then train set applied
loadedExcel = load_workbook("Milkshake.xlsx")
sheetofExcel = loadedExcel.active
rows = sheetofExcel.rows
#Get the names of the rows
head = [cell.value for cell in next(rows)] 

#List set to empty. The result of nested loop will be appended to empty list.
data = []
for row in rows:
    inner = []
    for name,val in zip(head,row):
        inner.append(val.value)
    data.append(inner)

# generate data and labels. Data list is supplied 7 indexed element of the array is chosen.
np_data = np.array(data)
labels = [element[7] for element in np_data]
np_data = np.delete(np_data,7,1)
np_data = np.delete(np_data,0,1)
np_labels = np.array(labels)
#print(np_labels)

"""
Training data set used for test cases.
First take 7 rows from testing data.
"""
testData = np_data[0:7]
testLabels = np_labels[0:7]

# knn operations from the link given in the hw description
classified = KNeighborsClassifier(n_neighbors=7)
classified.fit(np_data, np_labels)
prediction = classified.predict(testData)

# Accuracy can be calculated after for loop. Test labels index set to predition index value. 
count = 0
for i in range(len(testData)):
    if testLabels[i] == prediction[i]:
        count +=1
print("Accuracy is : {:.2f}% for my test set".format((count / len(testLabels))*100))
