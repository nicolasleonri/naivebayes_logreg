import random


def accuracy(classifier, data):
    """Compute the accuracy of a classifier on reference data.

    Args:
        classifier: An instance of the classifier.
        data: Reference data in the form of a list of tuples, where each tuple contains
              a list of words and the corresponding true label.

    Returns:
        The accuracy of the classifier on the provided data, represented as a float
        in the range [0.0, 1.0].
    """
    ##################### STUDENT SOLUTION ########################

    ''' 
    THIS WAS MY FIRST ANSWER (BEFORE FINISHING THE PREDICTION FUNCION)
    # I'll be working with the mathematical formula of accuracy given in Figure 4.4 of Chapter 4
    # The formula is: (tp+tn)/(tp+fp+tn+fn)
    # with tp = true positive, tn = true negative, fp = false positive and fn = false negative

    def test_prediction():
        return random.choice(["nonoffensive", "offensive"])

    tp_tn = 0 
    tp_tn_fp_fn = 0 

    for element in data:
        if test_prediction() == element[-1]:
            tp_tn += 1
            tp_tn_fp_fn += 1
        else:
            tp_tn_fp_fn += 1

    accuracy = float(tp_tn/tp_tn_fp_fn)

    return accuracy
    '''

    # I'll be working with the mathematical formula of accuracy given in Figure 4.4 of Chapter 4
    # The formula is: (tp+tn)/(tp+fp+tn+fn)
    # with tp = true positive, tn = true negative, fp = false positive and fn = false negative

    # Initialize variables for true positive + true negative and total instances
    tp_tn = 0
    tp_tn_fp_fn = len(data)

    # Iterate through the data to check if the predictions match the true labels
    for words, trained_label in data:
        prediction = classifier.predict(words)
        if prediction == trained_label:
            tp_tn += 1

    # Calculate accuracy, handling the case of an empty dataset
    accuracy = tp_tn / tp_tn_fp_fn if tp_tn_fp_fn > 0 else 0.0

    return accuracy

    ################################################################


def f_1(classifier, data):
    """Compute the F1-score of a classifier on reference data.

    Args:
        classifier: An instance of the classifier.
        data: Reference data in the form of a list of tuples, where each tuple contains
              a list of words and the corresponding true label.

    Returns:
        The F1-score of the classifier on the provided data, represented as a float.
    """
    ##################### STUDENT SOLUTION #########################

    # To calculate F1-score, we need to define what a false negative is.
    # In this case, we use the first key of the self.log_priors dictionary as the "correct" answer,
    # indicating that our system should identify whether a sentence is offensive or not.

    # Little trick to get the first key of a dictionary
    correct_answer = next(iter(classifier.log_priors))

    # Initialize variables for true positive (tp), true negative (tn), false positive (fp), and false negative (fn)
    tp, tn, fp, fn = 0, 0, 0, 0

    # Iterate through the data to calculate tp, tn, fp, fn
    for words, trained_label in data:
        prediction = classifier.predict(words)

        if prediction == trained_label and trained_label == correct_answer:
            tp += 1
        elif prediction == trained_label and trained_label != correct_answer:
            tn += 1
        elif prediction != trained_label and trained_label == correct_answer:
            fn += 1
        elif prediction != trained_label and trained_label != correct_answer:
            fp += 1
        else:
            raise ValueError("Something is wrong, pal.")

    # Calculate precision, recall, and F1-score
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) != 0 else 0

    return f1
    ################################################################
