from model.naivebayes import NaiveBayes, features1, features2, features3
from model.logreg import LogReg, featurize
from evaluation import accuracy, f_1
from utils import plot_grafs


def train_smooth(train_data, test_data):
    ######################### STUDENT SOLUTION #########################
    """
    Retrain the Naive Bayes classifier while varying the smoothing parameter k, 
    then evaluate on the test_data.

    Args:
        train_data: Training data.
        test_data: Test data.

    Returns:
        None.

    Notes:
        This function re-trains the Naive Bayes classifier with different values of
        the smoothing parameter k, evaluates the accuracy and F1-score on the test_data,
        and plots graphs of the accuracy and F1-score for different values of k.
    """

    k_values = [0.1, 0.5, 1.0, 2.0, 3.5, 5.0]
    # k_values = np.linspace(0.1,3.1,31)
    tested_accuracies = []
    tested_f1_scores = []

    for k in k_values:
        print(f"(Re)training naive bayes classifier with k = {k}")
        nb = NaiveBayes.train(train_data, k)

        acc = accuracy(nb, test_data)
        tested_accuracies.append(acc)
        print(f"Accuracy with k = {k}: ", acc)

        f1 = f_1(nb, test_data)
        tested_f1_scores.append(f1)
        print(f"F_1 with k = {k}: ", f1)

    plot_grafs(k_values, tested_accuracies, "Accuracy")
    plot_grafs(k_values, tested_f1_scores, "F1-score")

    '''
    DISCUSSION:
    If we increase k, both accuracy and F1 score increase for a moment until they exponentially decrease.
    From the graphs, the accuracy seems to reach its maximum at about k ~0.8
    From the graphs, the F1 score seems to reach its maximum at about k ~1.0
    In this case, choosing a k between, for example, K = 0.9 could improve both accuracy and F1 score.    
    '''

    return None
    ####################################################################


def train_feature_eng(train_data, test_data):
    ######################### STUDENT SOLUTION #########################
    """
    Improve on the basic bag of words model by changing the feature list of the model.
    Implement at least two variants using feature1 and feature2.

    Args:
        train_data: Training data.
        test_data: Test data.

    Returns:
        None.

    Notes:
        This function re-trains the Naive Bayes classifier with different feature sets,
        evaluates the accuracy and F1-score on the test_data, and prints the results for each variant.
    """

    def retrain_bayes(train_data, test_data, feature):
        print(f"(Re)training naive bayes classifier with feature: {feature}")
        nb = NaiveBayes.train(train_data)
        print("Accuracy: ", accuracy(nb, test_data))
        print("F_1: ", f_1(nb, test_data))

    retrain_bayes(features1(train_data), test_data, "Preprocessing")

    retrain_bayes(features2(train_data), test_data, "Removing stop words")

    retrain_bayes(features3(train_data), test_data, "Porter Stemming")

    '''
    DISCUSSION:
    The first feature has an accuracy of ~0.7344 and a F1-score of ~0.4180. This feature showed worse results than the original trained model.
    The second feature has an accuracy of ~0.848 and a F1-score of ~0.5426. This feature showed slightly better results than the original trained model.
    The third feature has an accuracy of ~0.7849 and a F1-score of ~0.3569. This feature showed worse results than the original trained model.
    
    The original results were an accuracy of ~0.8468 and a F1-score of ~0.5397.
    '''

    return None
    #####################################################################


def train_logreg(train_data, test_data):
    ######################### STUDENT SOLUTION #########################
    """
    Train a Logistic Regression model and evaluate it on the test set.

    Args:
        train_data: Training data.
        test_data: Test data.

    Returns:
        None.

    Notes:
        This function performs the following steps:
        1. Assign each word in the training set a unique integer index with `buildw2i()` function.
        2. Convert the data into a matrix where the element of the matrix is 1 if the corresponding
           word appears in a document, 0 otherwise with `featurize()` function.
        3. Train the Logistic Regression model with the feature matrix for 10 iterations with
           the default learning rate eta and L2 regularization with parameter C=0.1.
        4. Evaluate the model on the test set.
    """
    obj = LogReg()

    X_train, Y_train = featurize(train_data, test_data)

    # The evaluation is included in the train()-function
    obj.train(X_train, Y_train)

    return None
    #####################################################################
