import csv
import json
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer


def read_hate_tweets(annofile, jsonfile):
    """Reads in hate speech data."""
    all_data = {}
    annos = {}
    with open(annofile) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            if row[0] in annos:
                # if duplicate with different rating, remove!
                if row[1] != annos[row[0]]:
                    del (annos[row[0]])
            else:
                annos[row[0]] = row[1]

    tknzr = TweetTokenizer()

    with open(jsonfile) as jsonfile:
        for line in jsonfile:
            twtjson = json.loads(line)
            twt_id = twtjson['id_str']
            if twt_id in annos:
                all_data[twt_id] = {}
                all_data[twt_id]['offensive'] = "nonoffensive" if annos[twt_id] == 'none' else "offensive"
                all_data[twt_id]['text_tok'] = tknzr.tokenize(twtjson['text'])

    # split training and test data:
    all_data_sorted = sorted(all_data.items())
    items = [(i[1]['text_tok'], i[1]['offensive']) for i in all_data_sorted]
    splititem = len(all_data)-3250
    train_dt = items[:splititem]
    test_dt = items[splititem:]
    print('Training data:', len(train_dt))
    print('Test data:', len(test_dt))

    # print(train_dt)
    # print(test_dt)

    return (train_dt, test_dt)


def plot_grafs(k_values, tested_list, type):
    """
    Plots a graph of the specified type against smoothing parameter (k).

    Args:
        k_values (list): List of smoothing parameter values.
        tested_list (list): List of values corresponding to the specified type.
        type (str): Type of values being plotted.

    Returns:
        None
    """
    # Create a new figure with a specified size
    plt.figure(figsize=(10, 6))

    # Plot the data points with markers
    plt.plot(k_values, tested_list, marker='o')

    # Set the title and labels for the axes
    plt.title(f"{type} vs Smoothing Parameter (k)")
    plt.xlabel('Smoothing Parameter (k)')
    plt.ylabel(str(type))

    # Adjust layout for better presentation
    plt.tight_layout()

    # Display the plot
    plt.show()
