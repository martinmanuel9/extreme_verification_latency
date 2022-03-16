from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

rnd = np.random.RandomState(42)
# load the cancer dataset
cancer = datasets.load_breast_cancer()

# Randomly unlabel some records in the dataset
#unlabeled points are marked as -1
random_unlabeled_points = rnd.rand(len(cancer.target)) < 0.6

labels = np.copy(cancer.target)

labels_orig = np.copy(cancer.target)

labels[random_unlabeled_points] = -1

X=cancer.data

tot_unlabled=labels[labels==-1]


print(f"Total Records in dataset is {len(X)} and unlabeled records is  {len(tot_unlabled)}")

# define model
model = LabelPropagation(kernel='knn',n_neighbors=5, gamma=30, max_iter=2000)
# fit model on training dataset
print
model.fit(X, labels)
# make predictions
print(len(X[random_unlabeled_points]))
predicted_labels = model.predict(X[random_unlabeled_points])
true_labels = labels_orig[random_unlabeled_points]

#print the classification report and confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=model.classes_)
print("Label propagation model: %d labeled & %d unlabeled points (%d total)" %
      (len(labels[labels!=-1]), len(tot_unlabled) , len(X)))

print(classification_report(true_labels, predicted_labels))
print("Confusion matrix")
print(cm)