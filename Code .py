# MNIST Classification - chapter 3 Hands-on Machine Learning

# import the libraries
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np 

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X, y = mnist['data'], mnist['target'] 

# data contains 70,000 images, 28x28 (784 features), pixel intensity 0 to 255

# Visualising an instance in dataset
random_digit = X[30000]
random_digit_image = random_digit.reshape(28, 28)
plt.imshow(random_digit_image, cmap = matplotlib.cm.binary, interpolation = 'nearest')
plt.axis('off')
plt.show()
y[30000]

# Splitting data into training set and test set
# mnist data is already split into training and test set. (first 60,000 images form the training set, test set is the last 10000)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# shuffle dataset
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Identifying 1 digit (5-detector) - binary classifier
y_train_5 = (y_train == 5) # True for 5, false for all other digits
y_test_5 = (y_test == 5)


# SGDclassifier (binary) - 5 or not 5
from sklearn.linear_model import SGDClassifier
sgd_classifier = SGDClassifier(random_state = 0)
sgd_classifier.fit(X_train, y_train_5)

y_pred_5 = sgd_classifier.predict(X_train)

# Measuring performance
# k-fold cross validation (cv specifies number of folds)
from sklearn.model_selection import cross_val_score
SGD_score = cross_val_score(estimator = sgd_classifier, X = X_train, y = y_train_5, cv = 3, scoring = 'accuracy')
SGD_score # Gives score across 3 folds

# skewed dataset, accuracy not a good measure of performance
# importing confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train_5, y_pred_5)
cm
precision_score = 4895/(4895+2587) # when claiming an image is a 5, only right 65% of the time
recall_score = 4895/(4895+526) # detects 90% of 5s

from sklearn.metrics import f1_score
f1_score(y_train_5, y_pred_5) # 75.6%

y_scores = sgd_classifier.decision_function([30000])

# Deciding threshold precision/recall tradeoff
from sklearn.model_selection import cross_val_predict
y_scores = cross_val_predict(sgd_classifier, X_train, y_train_5, cv = 3, method = 'decision_function')

# Now that scores have been computed can compute precision and recall for all possible thresholds
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# Plotting precision and recall as a function of the threshold
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label = 'Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label = 'Recall')
    plt.xlabel('Threshold')#
    plt.legend(loc = 'center left')
    plt.ylim([0, 1])
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

def  plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.show()

# Plot ROC curve - Must compute the false positive rate and true positive rate 
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label = None):
    plt.plot(fpr, tpr, label = label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
plot_roc_curve(fpr, tpr)
plt.show()

# ROC area under curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)

# Implementing Random forest classifier and comparing ROC curve and ROC AUC to SGDClassifier. Obtain scores using method = predict proba (probability an instance is 5)
from sklearn.ensemble import RandomForestClassifier
forest_classifier = RandomForestClassifier(random_state = 42)
y_probas_forest = cross_val_predict(forest_classifier, X_train, y_train_5, cv = 3, method = 'predict_proba')

# Need scores not probabilities. Use probability as the score
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

# Plotting ROC curve against SGDClassifier
plt.plot(fpr, tpr, 'b:', label = 'SGD')
plot_roc_curve(fpr_forest, tpr_forest, 'Random Forest')
plt.legend(loc = 'lower right')
plt.show()

# Calculate ROC AUC for forest classifier
roc_auc_score(y_train_5, y_scores_forest)
forest_classifier.fit(X_train, y_train_5)
y_pred_5_forest = forest_classifier.predict(X_train)

# Confusion matrix for forest
cm_forest = confusion_matrix(y_train_5, y_pred_5_forest)
cm_forest
precision_forest = 5368/(5368 + 1)
recall_forest = 5368/(5368 + 53)

# Now using SGDClassifier on training set
sgd_classifier.fit(X_train, y_train)
y_pred_sgd = sgd_classifier.predict(X_train)

# Decision function scores
some_digit_scores = sgd_classifier.decision_function([X_train])

# RF classifier
forest_classifier.fit(X_train, y_train)
y_train_forest = forest_classifier.predict(X_train)

cross_val_score(sgd_classifier, X_train, y_train, cv = 5, scoring = 'accuracy')

# Standardscaling will improve accuracy
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

cross_val_score(sgd_classifier, X_train_scaled, y_train, cv = 5, scoring = 'accuracy')


# ERROR ANALYSIS
y_train_pred = cross_val_predict(sgd_classifier, X_train_scaled, y_train, cv = 3)
confusion_matrix = confusion_matrix(y_train, y_train_pred)

# Graphing confusion matrix
plt.matshow(confusion_matrix, cmap = plt.cm.gray)
plt.show()

# Graphing for errors - first, get error rate by dividing each value in CM by number of images in corresponding class
row_sums = confusion_matrix.sum(axis = 1, keepdims = True)
norm_confusion_matrix = confusion_matrix / row_sums

np. fill_diagonal(norm_confusion_matrix, 0)
plt.matshow(norm_confusion_matrix, cmap = plt.cm.gray)
plt.show()

# KNN classifier, multilabel >7, odd
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_multilabel)

train_knn_predict = knn_classifier.predict(X_train)
