import sys
import os
path = os.getcwd()
sys.path.append(path)

import random

# Related third party imports
import matplotlib.pyplot as plt
import numpy as np

# Local application/library specific imports
import nn.initialize
from nn.commons.maths import relu, softmax
from nn.commons.library import (
    configure_directory,
    read_model,
)
from nn.network.models import nn
from nn.embedding.models import Embedding
from nn.convolution.models import Convolution
from nn.pooling.models import Pooling
from nn.flatten.models import Flatten
from nn.dropout.models import Dropout
from nn.dense.models import Dense
from nn_train.image.settings import se_hPars

from nn_train.image.prepare_dataset import load_mnist_data

########################## CONFIGURE ##########################
random.seed(1)
np.random.seed(1)

np.set_printoptions(threshold=10)

np.seterr(all='warn')

configure_directory()

X_train, y_train, X_test, y_test = load_mnist_data(limit=1000)

embedding = Embedding(X_data=X_train,
                      Y_data=y_train,
                      X_scale=True,
                      Y_encode=True,
                      batch_size=16,
                      relative_size=(1, 0, 0))

name = 'NUMBER-Convolution-16-2_Pooling-3-Max_Flatten_Dense-64-relu_Dense-10-softmax'

se_hPars['learning_rate'] = 0.001
se_hPars['softmax_temperature'] = 5

layers = [
    embedding,
    Convolution(unit_filters=16, filter_size=(3, 3), strides=(1, 1), activate=relu),
    Pooling(pool_size=(2, 2)),
    Convolution(unit_filters=32, filter_size=(3, 3), strides=(1, 1), activate=relu),
    Convolution(unit_filters=32, filter_size=(3, 3), strides=(1, 1), activate=relu),
    Pooling(pool_size=(2, 2)),
    Flatten(),
    Dense(128, relu),
    Dropout(0.2),
    Dense(10, softmax)
]

# layers = [
#     embedding,
#     Convolution(unit_filters=16, filter_size=(3, 3), strides=(1, 1), activate=relu),
#     Convolution(unit_filters=16, filter_size=(3, 3), strides=(1, 1), activate=relu),
#     Pooling(pool_size=(2, 2)),
#     Dropout(0.2),
#     Flatten(),
#     Dense(128, relu),
#     Dropout(0.2),
#     Dense(10, softmax)
# ]

model = nn(layers=layers, name=name)

model.initialize(loss='CCE', se_hPars=se_hPars.copy(), end='\n')

model.train(epochs=10, init_logs=True, report=False)

#model.plot(path=False)

model.write()

predicted = model.predict(X_test, X_scale=True).P
x = (predicted == y_test)
print(x.sum() / X_test.shape[0])