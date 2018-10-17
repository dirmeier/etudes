#!/usr/bin/env python3

import pandas
import numpy
import pydot
import graphviz
import keras
import click
from sklearn import ensemble
from sklearn import linear_model
from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.layers import Dropout
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(fl):
  dat = pandas.read_csv(fl, sep="\t")
  del dat["cells.children_invasomes_count"]

  cell_header = list(filter(lambda x: x.startswith("cells"), dat.columns))

  dat = dat.replace([numpy.inf, -numpy.inf], numpy.nan)
  dat = dat.loc[:, cell_header]
  dat = dat.dropna(how='any')

  for _, c in enumerate(cell_header):
      dat.loc[:, c] = dat.loc[:, c].astype('float64')

  Y = dat["cells.children_bacteria_count"]
  Y = Y.astype("float")
  Y = Y.values.reshape([len(Y), 1])

  Y[numpy.where(Y[:,0] != 0), 0] = 1

  X = dat
  del X["cells.children_bacteria_count"]
  X = X.as_matrix()

  return X, Y


def create_deep_net(X, Y):
  model = Sequential()

  model.add(Dense(10, input_dim=X.shape[1], activation='relu'))
  model.add(Dropout(.2))

  model.add(Dense(20, activation='relu'))
  model.add(Dropout(.2))

  model.add(Dense(30, activation='relu'))
  model.add(Dropout(.2))

  model.add(Dense(20, activation='relu'))
  model.add(Dropout(.2))

  model.add(Dense(1, activation='sigmoid'))


  sgd = keras.optimizers.SGD(lr=0.01,
                             decay=1e-4,
                             momentum=0.9,
                             nesterov=True)

  model.compile(loss='binary_crossentropy',
                optimizer=sgd,
                metrics=["accuracy"])

  return model


def validate(X, Y, deep_net, rf, log_reg, out):
  folds = numpy.tile(range(10), int(X.shape[0] / 9))
  folds = folds[:X.shape[0]]
  numpy.random.shuffle(folds)

  cross_val_accuracies = {"deep-net": [], "random-forest": [], "logit": []}
  for i in range(10):
    train_idxs = numpy.where(folds != i)[0]
    test_idxs =  numpy.where(folds == i)[0]
    # train neural net
    deep_net.fit(X[train_idxs, :], Y[train_idxs, :],
              epochs=5, 
              batch_size=1000,
              validation_split=0.2, 
              verbose=1)
    score = deep_net.evaluate(X[test_idxs, :], Y[test_idxs, :])[1]
    cross_val_accuracies["deep-net"].append(score)
    
    trees = rf.fit(X=X[train_idxs, :], y=Y[train_idxs, :].flatten())
    score = trees.score(X=X[test_idxs, :], y=Y[test_idxs, :].flatten())
    cross_val_accuracies["random-forest"].append(score)
    
    lrfit = logreg.fit(X=X[train_idxs, :], y=Y[train_idxs, :].flatten())
    score = lrfit.score(X=X[test_idxs, :], y=Y[test_idxs, :].flatten())
    cross_val_accuracies["logit"].append(score)    

  data_to_plot = [
    cross_val_accuracies["deep-net"],
    cross_val_accuracies["logit"],
    cross_val_accuracies["random-forest"]
  ] 

  sns.set_style("whitegrid")
  ax = sns.boxplot(data=data_to_plot, palette=sns.light_palette("navy"), orient="h", showfliers=False)
  ax.set_yticklabels(["deep-net", "logit", "random-forest"])
  ax.set_xlabel('Accuracy')
  plt.savefig(out + "-training.pdf", dpi=720)
  plt.savefig(out + "-training.png", dpi=720)


@click.command()
@click.argument("file", type=str)
@click.argument("out", type=str)
def run(file, out):
  X, Y = load_data(file)
  
  deep_net = create_deep_net(X, Y)
  rf = ensemble.RandomForestClassifier(n_estimators=100, max_features='sqrt', n_jobs=-1)
  logreg = linear_model.LogisticRegression(solver="sag", max_iter=1000)

  validate(X, Y, deep_net, rf, logreg, out)


if __name__ == '__main__':
    run()
