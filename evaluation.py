import random


def accuracy(classifier, data):
    """Computes the accuracy of a classifier on reference data.

    Args:
        classifier: A classifier.
        data: Reference data.

    Returns:
        The accuracy of the classifier on the test data, a float.
    """
    ##################### STUDENT SOLUTION ########################

    ''' THIS WAS MY FIRST ANSWER (BEFORE FINISHING THE PREDICTION FUNCION)
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
    tp_tn = 0
    tp_tn_fp_fn = len(data)

    # We loop through the data and check if the prediction is correct
    for words, trained_label in data:
        prediction = classifier.predict(words)
        if prediction == trained_label:
            tp_tn += 1

    accuracy = tp_tn / tp_tn_fp_fn if tp_tn_fp_fn > 0 else 0.0

    return accuracy

    ################################################################


def f_1(classifier, data):
    """Computes the F_1-score of a classifier on reference data.

    Args:
        classifier: A classifier.
        data: Reference data.

    Returns:
        The F_1-score of the classifier on the test data, a float.
    """
    ##################### STUDENT SOLUTION #########################

    # As we have to define what a false negative is, we will need a "correct" answer.
    # In this case, I've decided that it will be the first key of the self.log_priors dictionary, in this case: offensive.
    # This means that our system should find out if a sentence is offensive and not if it's nonoffensive

    # Little trick to get the first key of a dictionary
    correct_answer = next(iter(classifier.log_priors))

    # We initialize the integers
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    # And then -based in each situation- we add any counts necesarry
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

    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    # Calculate F1-score
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) != 0 else 0

    return f1
    ################################################################
