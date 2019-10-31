#!/usr/bin/python3

import sys;
from os.path import exists, isdir;
import pickle;
import numpy as np;

def main(file_path):

  f = open(file_path, 'r');
  ticks = list();
  for line in f:
    line = line.strip();
    tokens = line.split('\t');
    assert len(tokens) == 3;
    ticks.append([float(tokens[1]), float(tokens[2])]);
  f.close();
  dataset = np.array(ticks, dtype = np.float32);
  print(dataset.shape);
  with open('dataset.pkl', 'wb') as f:
    f.write(pickle.dumps(dataset));

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <file>");
    exit(1);
  if False == exists(sys.argv[1]) or True == isdir(sys.argv[1]):
    print("invalid file!");
    exit(1);
  main(sys.argv[1]);
