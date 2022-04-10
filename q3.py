from openpyxl import load_workbook
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

# Read data from excel and convert then train set applied
loadedExcel = load_workbook("Milkshake.xlsx")
# Get all values from excel
sheet = loadedExcel.active
rows = sheet.rows
#Get all names of the rows
head = [cell.value for cell in next(rows)] 

#List set to empty. The result of nested loop will be appended to empty list.
data = []
for row in rows:
    inner = []
    for name,val in zip(head,row):
        inner.append(val.value)
    data.append(inner)

# generate data and labels. Data list is supplied 7th indexed element of the array is chosen.
np_data = np.array(data)
labels = [element[7] for element in np_data]
np_data = np.delete(np_data,7,1)
np_data = np.delete(np_data,0,1)
np_labels = np.array(labels)


# Between 2 and 9 loop elements. 
for i in range(2, 9):
    k_fold = KFold(n_splits=i,shuffle=False)
# Empty accuracyList has been created to be appended  with accuracy
    accuracyList = []
    for train_index, test_index in k_fold.split(np_data):
        x_train = np.array([np_data[a] for a in train_index])
        x_test = np.array([np_labels[a] for a in train_index])
        y_train = np.array([np_data[a] for a in test_index])
        y_test = np.array([np_labels[a] for a in test_index])

        classified = KNeighborsClassifier(n_neighbors=3)
        classified.fit(x_train, x_test)
        prediction = classified.predict(y_train)
        accurate = 0
        for j in range(len(prediction)):
            if prediction[j] == y_test[j]:
                accurate +=1
        accuracy = accurate/len(prediction)
        accuracyList.append(accuracy)
# {} is used to get variables values.
    print("Average accuracy is : {} for {} fold validation".format(sum(accuracyList)/len(accuracyList),i))

""""
The ultimate result of the code block showed that the best performance is with 5 fold cross validation.

"""

