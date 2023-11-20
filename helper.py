from model.naivebayes import NaiveBayes, features1, features2, features3
from model.logreg import LogReg, featurize
from evaluation import accuracy, f_1
from utils import plot_grafs


def train_smooth(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) Re-train Naive Bayes while varying smoothing parameter k,
    #         then evaluate on test_data.
    #         2) Plot a graph of the accuracy and/or f-score given
    #         different values of k and save it, don't forget to include
    #         the graph for your submission.

    ######################### STUDENT SOLUTION #########################

    k_values = [0.1, 0.5, 1.0, 2.0, 3.5, 5.0]
    # k_values = np.linspace(0.1,3.1,31)
    tested_accuracies = []
    tested_f1_scores = []

    for k in k_values:
        print(f"(Re)training naive bayes classifier with k = {k}")
        nb = NaiveBayes.train(train_data, k)

        acc = accuracy(nb, test_data)
        tested_accuracies.append(acc)
        print("Accuracy: ", acc)

        f1 = f_1(nb, test_data)
        tested_f1_scores.append(f1)
        print("F_1: ", f1)

    plot_grafs(k_values, tested_accuracies, "Accuracy")
    plot_grafs(k_values, tested_f1_scores, "F1-score")

    # DISCUSSION:
    # If we increase k, both accuracy and F1 score increase for a moment until they exponentially decrease.
    # From the graphs, the accuracy seems to reach its maximum by about k ~0.8
    # From the graphs, the F1 score seems to reach its maximum by about k ~1.0
    # In this case, I would choose a k in between of -for example- K = 0.9 to improve the whole accuracy and f1 score

    pass
    ####################################################################


def train_feature_eng(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) Improve on the basic bag of words model by changing
    #         the feature list of your model. Implement at least two
    #         variants using feature1 and feature2
    ########################### STUDENT SOLUTION ########################

    def retrain_bayes(train_data, test_data, feature):
        print(f"(Re)training naive bayes classifier with feature {feature}")
        nb = NaiveBayes.train(train_data)
        print("Accuracy: ", accuracy(nb, test_data))
        print("F_1: ", f_1(nb, test_data))

    retrain_bayes(features1(train_data), test_data, "Preprocessing")

    retrain_bayes(features2(train_data), test_data, "Removing stop words")

    retrain_bayes(features3(train_data), test_data, "Porter Stemming")

    pass
    #####################################################################


def train_logreg(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) First, assign each word in the training set a unique integer index
    #         with `buildw2i()` function (in model/logreg.py, not here). DONE!
    #         2) Now that we have `buildw2i`, we want to convert the data into
    #         matrix where the element of the matrix is 1 if the corresponding
    #         word appears in a document, 0 otherwise with `featurize()` function. DONE!
    #         3) Train Logistic Regression model with the feature matrix for 10
    #         iterations with default learning rate eta and L2 regularization
    #         with parameter C=0.1.
    #         4) Evaluate the model on the test set.
    ########################### STUDENT SOLUTION ########################

    obj = LogReg()

    X_train, Y_train = featurize(train_data, test_data)

    obj.train(X_train, Y_train)

    pass
    #####################################################################
