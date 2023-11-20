import numpy as np
from collections import Counter


class LogReg:
    def __init__(self, eta=0.01, num_iter=10, C=0.1):
        """
        Initialize a Logistic Regression classifier.

        Args:
            eta (float): The learning rate. Default is 0.01.
            num_iter (int): The number of iterations for training. Default is 10.
            C (float): The regularization parameter. Default is 0.1.
        """

        self.eta = eta
        self.num_iter = num_iter
        self.C = C

    def softmax(self, inputs):
        """
        Calculate the softmax for the given inputs.

        Args:
            inputs: numeric array.

        Returns: 
            Softmax probabilities as array.

        Notes:
            The adjustment exp_inputs was used to improve numerical stability
        """

        # Shift inputs to avoid numerical instability
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Calculate softmax probabilities
        softmax_probs = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)

        return softmax_probs

    def train(self, X, Y, ):
        """
        Train the Logistic Regression model.

        Args:
            X: Feature matrix
            Y: Label matrix

        Returns:
            None
        """

        #################### STUDENT SOLUTION ###################

        # Initialize weights and bias
        self.weights = np.zeros((X.shape[1], Y.shape[1]), dtype=np.float128)
        self.bias = np.zeros(Y.shape[1], dtype=np.float128)

        for i in range(self.num_iter):
            # Create minibatches for training
            minibatches = self.create_minibatches(X, Y, 100)

            for X_batch, Y_batch in minibatches:
                # Calculate predictions and gradient
                predictions = self.softmax(
                    np.dot(X_batch, self.weights) + self.bias) - Y_batch
                gradient = np.dot(X_batch.T, predictions)

                # Update weights and bias with regularization term
                self.weights -= self.eta * gradient + self.C
                self.bias -= self.eta * np.sum(predictions, axis=0)

            # Evaluate and print metrics after each epoch
            predictions = self.predict(X)
            accuracy = accuracy_logreg(predictions, Y)
            f1_score = f1_score_logreg(predictions, Y)

            print(
                f"For epoch {i + 1}, the accuracy was: {accuracy} and the F1 score was: {f1_score}")

        print("Training done!")

        return None
        #########################################################

    def p(self, X):
        """
        Calculate the log probability prediction.

        Args:
            X: Matrix.

        Returns:
            Log Softmax probabilites predictions.
        """
        ################## STUDENT SOLUTION ########################
        # Calculate dot product and apply softmax
        z = np.dot(X, self.weights)
        softmax_result = self.softmax(z)

        # Calculate log probabilities
        log_probabilities = np.log(softmax_result)

        return log_probabilities
        ############################################################

    def predict(self, X):
        """
        Predict the class for a set of input features.

        Args:
            X (numpy.ndarray): Input feature matrix.

        Returns:
            numpy.ndarray: Predicted classes.
        """
        ####################### STUDENT SOLUTION ####################
        # Get the log probability predictions
        log_probabilities = self.p(X)

        # Identify the class with the highest probability for each input
        predictions = log_probabilities.argmax(axis=1)

        return predictions
        #############################################################

    def create_minibatches(self, X, Y, batch_size):
        """
        Create minibatches from input features and labels.

        Args:
            X: Input features matrix.
            Y: Labels matrix.
            batch_size: Size of each minibatch.

        Returns:
            List of tuples, each containing a minibatch of X and Y.
        """
        num_samples = X.shape[0]
        minibatches = []

        # Shuffle the data to ensure randomness in minibatches
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        # Iterate over the data, creating minibatches of the specified size
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            # Create minibatch for X and Y
            X_batch = X[batch_indices, :]
            Y_batch = Y[batch_indices, :]

            minibatches.append((X_batch, Y_batch))

        return minibatches


def buildw2i(vocab):
    """
    Create indexes for 'featurize()' function.

    Args:
        vocab: vocabulary constructed from the training set.

    Returns:
        Dictionaries with word as the key and index as its value.
    """
    # YOUR CODE HERE
    #################### STUDENT SOLUTION ######################

    # Extract all unique words from the vocabulary
    vocabulary = set(word for words, _ in vocab for word in words)

    # Create a dictionary mapping each word to its index
    word_index_dict = {word: index for index, word in enumerate(vocabulary)}

    return word_index_dict
    ############################################################


def featurize(data, train_data=None):
    """
    Convert data into X and Y where X is the input and
    Y is the label.

    Args:
        data: Training or test data.
        train_data: Reference data to build vocabulary from.

    Returns:
        Matrix X and Y.
    """
    # YOUR CODE HERE
    ##################### STUDENT SOLUTION #######################

    # First, we create our dictionary with the unique words and their indeces
    word_index_dict = buildw2i(data)  # Using train_data to build vocabulary

    # Initialize empty matrices
    # Matrix X is an N × F matrix (N: number of data instances, F: number of features)
    X = np.empty((len(data), len(word_index_dict)))

    # Initialize an empty numpy matrix (N × i), where i is the number of labels, containing only zeros
    # Extract unique labels from the data
    documents, labels = zip(*[(document, label)
                              for document, label in data])
    unique_labels = list(set(labels))

    # Initialize Matrix Y
    Y = np.zeros((len(labels), len(unique_labels)))

    # Ensure matrices have equal length
    if len(X) != len(Y):
        raise ValueError("The matrices must have equal length.")

    # Populate matrices
    # Matrix X is populated with 1 if any of the words in the document is in the unique words dictionary
    for idx, val in enumerate(documents):
        for word in val:
            if word in word_index_dict.keys():
                X[idx][word_index_dict[word]] = int(1)

    # Matrix Y is populated with binary values based on the presence of each label in the corresponding document
    for idx, label in enumerate(labels):
        label_index = unique_labels.index(label)
        Y[idx, label_index] = int(1)

    return X, Y
    ##############################################################


def accuracy_logreg(predictions, labels):
    """
    Computes the accuracy of a Logistic Regression classifier on reference data.

    Args:
        predictions (numpy.ndarray): Predicted labels from the classifier.
        labels (numpy.ndarray): True labels from the reference data.

    Returns:
        float: The accuracy of the classifier on the test data.
    """
    # Assign classes corresponding to the returned index
    predictions = [np.array([1, 0]) if index == 0 else np.array(
        [0, 1]) for index in predictions]

    # Calculate accuracy
    accuracy = np.sum(predictions == labels) / labels.size * 100

    return accuracy


def f1_score_logreg(predictions, labels):
    """
    Computes the F1 score of a Logistic Regression classifier on reference data.

    Args:
        predictions (numpy.ndarray): Predicted labels from the classifier.
        labels (numpy.ndarray): True labels from the reference data.

    Returns:
        float: The F1 score of the classifier on the test data.
    """

    # Assign classes corresponding to the returned index
    predictions = [np.array([1, 0]) if index == 0 else np.array(
        [0, 1]) for index in predictions]

    # Calculate true positive, false positive, and false negative
    tp = np.sum([np.array_equal(pred, label)
                for pred, label in zip(predictions, labels)])
    fp = len(predictions) - tp
    fn = len(labels) - tp

    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) != 0 else 0

    return f1
