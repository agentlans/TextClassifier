import argparse
from Classifier import Classifier
import pickle

parser = argparse.ArgumentParser(description='Filters a file by printing only the "positive" lines.')
parser.add_argument('classifier', help='file containing the classifier')
parser.add_argument('infile', help='file containing positive and negative lines')
parser.add_argument('outfile', help='where to save the filtered file containing only positive lines')
args = parser.parse_args()

# Load the model
cls = None
with open(args.classifier, 'rb') as f:
  cls = pickle.load(f)

if cls is None:
  raise "Couldn't load model."

# For writing a string to a file object
def write_to(file_obj, txt):
  file_obj.write(txt)
  file_obj.write("\n")

# Class for writing to a file or standard output
class Writer:
  def __init__(self, filename):
    if filename == '-':
      self.file = None
      self.writer = print
    else:
      self.file = open(filename, 'w')
      self.writer = lambda txt: write_to(self.file, txt)
  def write(self, txt):
    self.writer(txt)
  def close(self):
    if self.file is not None:
      self.file.close()

# Print every line that's classified as positive
w = Writer(args.outfile)
with open(args.infile, 'r') as f:
  for line in f:
    if cls.predict(line) == 1:
      w.write(line)
w.close()
