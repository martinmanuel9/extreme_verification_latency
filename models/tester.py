
# from skmultiflow.data import SEAGenerator
# from skmultiflow.bayes import NaiveBayes
# import numpy as np
# # Setup a data stream
# stream = SEAGenerator(random_state=1)
# # Setup Naive Bayes estimator
# naive_bayes = NaiveBayes()
# # Setup variables to control loop and track performance
# n_samples = 0
# correct_cnt = 0
# max_samples = 200
# print(stream.next_sample())
# # Train the estimator with the samples provided by the data stream
# while n_samples < max_samples and stream.has_more_samples():
#     X, y = stream.next_sample()
#     y_pred = naive_bayes.predict(X)
#     if y[0] == y_pred[0]:
#         correct_cnt += 1
#     naive_bayes.partial_fit(X, y)
#     n_samples += 1
# # Display results
# print('{} samples analyzed.'.format(n_samples))
# print('Naive Bayes accuracy: {}'.format(correct_cnt / n_samples))

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
print(X_test)
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
       % (X_test.shape[0], (y_test != y_pred).sum()))