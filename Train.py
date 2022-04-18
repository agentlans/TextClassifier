import argparse
from Classifier import Classifier
import pickle

parser = argparse.ArgumentParser(description='Trains model for classifying lines of text.')
parser.add_argument('pos', help='input file containing "positive" lines')
parser.add_argument('neg', help='input file containing "negative" lines')
parser.add_argument('--nlp', default='en_core_web_sm', help='natural language model')
parser.add_argument('out', help='where to save the classifier')
args = parser.parse_args()

cls = Classifier(args.nlp)
cls.train(args.pos, args.neg)

# Save to file
with open(args.out, 'wb') as f:
  pickle.dump(cls, f)
