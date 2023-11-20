import numpy as np
from collections import Counter


class LogReg:
    def __init__(self, eta=0.01, num_iter=10, C=0.1):
        self.eta = eta
        self.num_iter = num_iter
        self.C = C

    def softmax(self, inputs):
        """
        Calculate the softmax for the give inputs (array)
        :param inputs:
        :return:
        """
        # return np.exp(inputs) / float(np.sum(np.exp(inputs)))

        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)

    def train(self, X, Y, ):
        #################### STUDENT SOLUTION ###################

        # weights initialization
        self.weights = np.zeros((X.shape[1], Y.shape[1]), dtype=np.float128)
        self.bias = np.zeros(Y.shape[1], dtype=np.float128)

        for i in range(self.num_iter):
            minibatches = self.create_minibatches(X, Y, 100)

            for X_batch, Y_batch in minibatches:
                predictions = self.softmax(
                    np.dot(X_batch, self.weights) + self.bias) - Y_batch
                gradient = np.dot(X_batch.T, predictions)

                self.weights -= self.eta * gradient + self.C
                self.bias -= self.eta * np.sum(predictions, axis=0)

            predictions = self.predict(X)

            print(f"For epoch {i+1}, the accuracy was:", accuracy_logreg(
                predictions, Y), "and the F1 scores:", accuracy_logreg(predictions, Y))

        print("Training done!")

        return None
        #########################################################

    def p(self, X):
        # YOUR CODE HERE
        #     TODO:
        #         1) Fill in (log) probability prediction
        ################## STUDENT SOLUTION ########################
        return np.log(self.softmax(np.dot(X, self.weights)))
        ############################################################

    def predict(self, X):
        # YOUR CODE HERE
        #     TODO:
        #         1) Replace next line with prediction of best class
        ####################### STUDENT SOLUTION ####################
        predictions = self.p(X).argmax(axis=1)

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

        # Shuffle the data
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

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

    # First we need to get all words in the vocabulary
    vocabulary = set(word for words, _ in vocab for word in words)

    # Then we create dictionary looping through the vocabulary and adding the index as value and the word as key
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
    word_index_dict = buildw2i(data)

    # Now, we can initialize our empty matrices
    # The Matrix X is an N × F matrix (N : number of data instances, F : number of features)
    # In this case, the number of features is equal to the number unique words
    X = np.empty((len(data), len(word_index_dict)))

    # Then we can initialize an empty numpy matrix (N x i) being i the number of labels and containing only zeros
    # As we need to know how many features we have, we'll isolate the unique labels like this...
    documents, labels = zip(*[(document, label)
                              for document, label in data])
    unique_labels = []
    for x in set(labels):
        unique_labels.append(x)
    # ... to get our Marix Y:
    Y = np.zeros((len(labels), len(unique_labels)))

    # In this point, I want to be sure that they're the same length. Otherwise, it doesn't make sense to continue:
    if len(X) != len(Y):
        raise ("The matrices have to have equal length.")

    # Now, we'll populate the matrices
    # The first matrix has to be populated with a 1 if any of the words found in the document is also in the unique words dictionary
    # This means that each row has to have a 0 or a 1 depending if any of the unique words are found in the tweet using the indexation of the unique words dictionray
    for idx, val in enumerate(documents):
        for word in val:
            if word in word_index_dict.keys():
                X[idx][word_index_dict[word]] = int(1)

    # The second matrix has to be populated with binary values based on the presence of each label in the corresponding document
    for idx, label in enumerate(labels):
        label_index = unique_labels.index(label)
        Y[idx, label_index] = int(1)

    return X, Y
    ##############################################################


def accuracy_logreg(predictions, labels):
    # assign classes corresponding to returned index
    predictions = [np.array([1, 0]) if index == 0 else np.array(
        [0, 1]) for index in predictions]
    return np.sum(predictions == labels) / labels.size * 100


def f1_score_logreg(predictions, labels):
    # assign classes corresponding to returned index
    predictions = [np.array([1, 0]) if index == 0 else np.array(
        [0, 1]) for index in predictions]

    # calculate true positive, false positive, and false negative
    tp = np.sum([np.array_equal(pred, label)
                for pred, label in zip(predictions, labels)])
    fp = len(predictions) - tp
    fn = len(labels) - tp

    # calculate precision, recall, and f1 score
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) != 0 else 0

    return f1
