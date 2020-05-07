import numpy as np
import math


def load_spamham_dataset():
    import string

    with open('spamham.txt', mode='r', encoding='utf-8') as f:
        rows = [l.strip().split('\t')[:2] for l in f]

    y, X = zip(*rows)
    X = [x.translate(str.maketrans('', '', string.punctuation)
                     ).lower().split() for x in X]

    return X, y


X, y = load_spamham_dataset()

print('Sample:')
print(f'{y[0]}: {X[0]}')
print(f'{y[2]}: {X[2]}')

# Implement your solution here.


class NaiveBayesSpamClassifier(object):
    def __init__(self):
        self.priors = {}
        self.hamLikehoods = {}
        self.spamLikehoods = {}
        self.word_total = {}
        self.ham_total=0
        self.spam_total=0

    def fit(self, X, y):
        """
        X is a list of `n` text messages. Each text message is a list of strings with at least length one.
        y is a list of `n` labels either the string 'spam' or the string 'ham'.
        """
        total_doc_cnt=len(y)

        label_doc_cnt={}
        bigdoc_word={}

        vocabulary=set()
        for i in range(len(y)):
            if y[i] not in label_doc_cnt:
                label_doc_cnt[y[i]]=0
                bigdoc_word[y[i]]=[]
            label_doc_cnt[y[i]]+=1
            for w in X[i]:
                vocabulary |=set(w)
            
        V=len(vocabulary)
        print(vocabulary)
        log_prior={label: math.log(1.0 * cnt / total_doc_cnt) for label, cnt in label_doc_cnt.items()}


    def predict(self, X):
        """
        X is a list of `n` text messages. Each text message is a list of strings with at least length one.
        The method returns a list of `n` strings, i.e. classification labels ('spam' or 'ham').
        """
      
     
        return result

# The following code will evaluate your classifier.


class HamClassifier(object):
    """
    This classifier is a primitive baseline, which just predicts the most common class each time.
    Naive Bayes should definitely beat this.
    """

    def fit(self, X, y): pass
    def predict(self, X): return len(X)*['ham']


def train_evaluate(classifier, X, y):
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split

    # Apply train-test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=2020)
    # Inititialize and train classifier.
    classifier.fit(X_train, y_train)
    # Evaluate classifier on test data.
    yhat_test = classifier.predict(X_test)
    cmatrix = confusion_matrix(y_test, yhat_test, labels=['ham', 'spam'])

    return cmatrix


def plot_confusion_matrix(cmatrix, classifier_name):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)
    ax.matshow(cmatrix, cmap='Greens')
    for x in (0, 1):
        for y in (0, 1):
            ax.text(x, y, cmatrix[y, x])
    ax.set_xlabel('predicted label')
    ax.set_ylabel('true label')
    ax.set_xticklabels(['', 'ham', 'spam'])
    ax.set_yticklabels(['', 'ham', 'spam'])
    ax.set_title(classifier_name)
    plt.show()


ham_classifier = HamClassifier()
your_classifier = NaiveBayesSpamClassifier()
ham_cmatrix = train_evaluate(ham_classifier, X, y)
your_cmatrix = train_evaluate(your_classifier, X, y)

plot_confusion_matrix(ham_cmatrix, 'HamClassifier')
plot_confusion_matrix(your_cmatrix, 'NaiveBayesSpamClassifier')
