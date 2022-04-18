import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

class Classifier:
  def __init__(self, spacy_nlp):
    '''Loads NLP model.'''
    self.nlp = spacy.load(spacy_nlp)
  def to_vector(self, txt):
    '''Converts a string to a vector.'''
    doc = self.nlp(txt)
    return doc.vector
  def file_to_matrix(self, filename):
    '''Loads a file. Returns matrix where each row is a line.'''
    rows = []
    with open(filename, 'r') as f:
      for line in f:
        rows.append(self.to_vector(line))
      return np.row_stack(rows)
  def train(self, pos_file, neg_file):
    '''Trains model with files containing positive and negative lines.'''
    pos = self.file_to_matrix(pos_file)
    neg = self.file_to_matrix(neg_file)
    # vector of Y values
    y_pos = np.ones(pos.shape[0])
    y_neg = np.zeros(neg.shape[0])
    # Set up Numpy objects
    x = np.concatenate((pos, neg))
    y = np.concatenate((y_pos, y_neg))
    # Split to training and test sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
    self.model = LogisticRegression()
    # Learn the good and bad lines
    y_pred = self.model.fit(X_train, y_train).predict(X_test)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
  def predict(self, txt):
    '''Predicts whether the given string is positive (1) or negative (0).'''
    x = self.to_vector(txt).reshape(1, -1)
    return self.model.predict(x)
