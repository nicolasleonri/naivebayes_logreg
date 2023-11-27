import re
import numpy as np
from math import log
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class NaiveBayes(object):

    ######################### STUDENT SOLUTION #########################
    def __init__(self, log_priors, log_likelihoods):
        """Initialises a new classifier."""
        self.log_priors = log_priors
        self.log_likelihoods = log_likelihoods
    ####################################################################

    def predict(self, x):
        """Predicts the class for a document.

        Args:
            x: A document, represented as a list of words.

        Returns:
            The predicted class, represented as a string.
        """
        ################## STUDENT SOLUTION ########################

        # We create temporary dictionary that will give us the final answer
        tmp_dict = {key: 0 for key in self.log_priors}

        for word in x:
            # Then we look if the word is to be found in our log_likelihoods-dictionary
            if word in self.log_likelihoods.keys():
                # If so, we create a new temporary dictionary with the values (two lists) already found out during the training of data
                tmp_input_dict = {label: value for label,
                                  value in self.log_likelihoods[word]}

                # We add it to the first temporary dictionary
                for key, value in tmp_input_dict.items():
                    tmp_dict[key] += value

            else:
                # Else, we pass
                pass

        # And we get the key with the maximum
        max_key = str(max(tmp_dict, key=tmp_dict.get))

        return max_key
        ############################################################

    @classmethod
    def train(cls, data, k=1):
        """Train a new classifier on training data using maximum
        likelihood estimation and additive smoothing.

        Args:
            cls: The Python class representing the classifier.
            data: Training data.
            k: The smoothing constant.

        Returns:
            A trained classifier, an instance of `cls`.
        """
        ##################### STUDENT SOLUTION #####################

        # To calculate the number of documents in data:
        n_doc = int(len(data))

        # To calculate the number of documents in each class, we create a dictionary:
        n_c = Counter(label for _, label in data)

        # To test that everything is working smoothly, I added the following raise error in case the labels are more frequent than the documents:
        if sum(n_c.values()) > n_doc:
            raise Exception(
                "Sorry, labels cannot be more frequent than examples given.")

        # We create a dictionary with the log prior information of each class:
        log_priors = {label: np.log(count / n_doc)
                      for label, count in n_c.items()}

        # And we calculate the size of the vocabulary
        vocabulary = len(set(word for words, _ in data for word in words))

        # Similarly, I added this raise error to control that the vocabulary isn't bigger than the corpus.
        list_words = [word for word, _ in data]
        corpus = [word for sublist in list_words for word in sublist]
        if vocabulary > len(corpus):
            raise Exception(
                "Sorry, the vocabulary cannot be bigger than the corpus.")

        # We also create a Counter for each label that has all words belonging to that label and how often each word appears
        # This will improve the perfomance when calculating the log likelihood by not having to count again each time
        label_word_counts = {label: Counter(
            word for words, searched_label in data for word in words if searched_label == label) for label in n_c}

        # Now we can create a dictionary that has all words to their respective label
        big_doc = {label: [] for label in n_c}
        for words, label in data:
            big_doc[label].extend(words)

        # Afterwards, we create a dictionary to save all the likelihood calculations. It only contains the unique words and it's empty.
        log_likelihoods = {word: [] for word in set(
            word for words, _ in data for word in words)}

        # We loop through the dictionary...
        for word in log_likelihoods:
            # Loop through the labels and the word count in another dictionary...
            for label, word_count in label_word_counts.items():
                # Get how often that word was found in the specific label...
                count = word_count[word]
                # Calculate the likelihood with log...
                likelihood = log(
                    (count + k) / (len(big_doc[label]) + (vocabulary*k)))
                # And save it in the dictionary!
                log_likelihoods[word].append([label, likelihood])

        # Now, we have a dictionary with all words and a list of two lists which contain the label and its calculated likelihood
        return cls(log_priors, log_likelihoods)
        ############################################################


def features1(data):
    """
    Preprocesses text data for a Naive Bayes classifier.

    Args:
        data (list): Training data, where each element is a tuple containing a list of words (sentence) and its label.

    Returns:
        list: Preprocessed data, where each element is a tuple containing a list of cleaned words and its label.

    Notes:
        This function performs the following preprocessing steps on the input text data:
        1. Converts words to lowercase.
        2. Removes non-alphanumeric characters.
        3. Removes hyperlinks.
        4. Removes old-style retweet text ("RT").
        5. Removes hashtags from words.
        The preprocessed data is returned and can be used to initialize the NaiveBayes() class.
    """
    ###################### STUDENT SOLUTION ##########################
    def preprocess_text(data):

        # Frist we separate our sentences (tweets) and labels
        sentences, labels = zip(*[(sentence, label)
                                for sentence, label in data])

        # And an empty list which will save our preprocessed data
        cleaned_sentences = []

        for sentence in sentences:
            tmp_cleaned_sentence = []
            for word in sentence:
                # lower case words_
                word = word.lower()
                # remove non alphanumeric
                word = re.sub(r'[^a-z0-9\s]', '', word)
                # remove hyperlinks
                word = re.sub(r'https?:\/\/.*[\r\n]*', '', word)
                # remove old style retweet text "RT"
                word = re.sub(r'^RT[\s]+', '', word)
                # remove the hashtags from the words
                word = re.sub(r'#*', '', word)
                tmp_cleaned_sentence.append(word)
            cleaned_sentences.append(tmp_cleaned_sentence)

        # Finally we create a cleaned data variable that will replace our train_data
        clean_data = list(zip(cleaned_sentences, labels))

        return clean_data

    return preprocess_text(data)
    ##################################################################


def features2(data):
    """
    Removes stop words from text data for a Naive Bayes classifier.

    Args:
        data (list): Training data, where each element is a tuple containing a list of words (sentence) and its label.

    Returns:
        list: Data with stop words removed, where each element is a tuple containing a list of words without stop words and its label.

    Notes:
        This function removes common English stop words from the input text data. Stop words are words that are considered to be of little value for text classification as they occur frequently across texts.
        The processed data is returned and can be used to initialize the NaiveBayes() class.
    """
    ###################### STUDENT SOLUTION ##########################
    def remove_stop_words(data):
        sentences, labels = zip(*[(sentence, label)
                                for sentence, label in data])

        cleaned_sentences = []

        # We get the set of english stopwords
        stops = set(stopwords.words('english'))

        for sentence in sentences:
            tmp_cleaned_sentence = []
            for word in sentence:
                if word not in stops:  # remove stopwords
                    tmp_cleaned_sentence.append(word)
            cleaned_sentences.append(tmp_cleaned_sentence)

        clean_data = list(zip(cleaned_sentences, labels))

        return clean_data

    return remove_stop_words(data)
    ##################################################################


def features3(data):
    """
    Applies stemming to words in text data for a Naive Bayes classifier.

    Args:
        data (list): Training data, where each element is a tuple containing a list of words (sentence) and its label.

    Returns:
        list: Data with words stemmed, where each element is a tuple containing a list of stemmed words and its label.

    Notes:
        This function applies stemming to each word in the input text data using the Porter stemmer. Stemming reduces words to their root or base form, helping to simplify and normalize the vocabulary.
        The processed data is returned and can be used to initialize the NaiveBayes() class.
    """
    ###################### STUDENT SOLUTION ##########################
    def stemming_words(data):

        # We will use the Porter stemmer
        stemmer = PorterStemmer()

        sentences, labels = zip(*[(sentence, label)
                                for sentence, label in data])

        cleaned_sentences = []

        for sentence in sentences:
            tmp_cleaned_sentence = []
            for word in sentence:
                stem_word = stemmer.stem(word)  # steeming the words
                tmp_cleaned_sentence.append(stem_word)
            cleaned_sentences.append(tmp_cleaned_sentence)

        clean_data = list(zip(cleaned_sentences, labels))

        return clean_data

    return stemming_words(data)
    ##################################################################
