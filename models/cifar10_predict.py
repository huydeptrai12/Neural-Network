import sys
sys.path.append("C:\\Users\\nguye\\OneDrive\\Desktop\\CNN\\EpyNN")

from epynn.commons.library import read_model
from epynnlive.dummy_image.prepare_dataset import load_cifar10_data
from epynn.commons.metrics import metrics_functions

name = "1728527935_CIFAR10-OVER-Convolution-16-2_Pooling-3-Max_Flatten_Dense-64-relu_Dense-10-softmax.pickle"
name_complex = "1728530588_CIFAR10-Convolution-16-2_Pooling-3-Max_Flatten_Dense-64-relu_Dense-10-softmax.pickle"
path = "models\\" + name_complex
print(path)
model = read_model(path)
    
X_train, y_train, X_test, y_test = load_cifar10_data(limit=100)

predicted = model.predict(X_test, X_scale = True).P
print(X_train.shape)
print(predicted.shape)
print(y_test.shape)
accuracy = metrics_functions(key = 'accuracy')(y_test, predicted)
accuracy = accuracy.mean()
print(accuracy)

import matplotlib.pyplot as plt
import math
def foo(y_validation_re, predictions, x_validation_normalized):
    numbers_to_display = 25
    num_cells = math.ceil(math.sqrt(numbers_to_display))
    plt.figure(figsize=(20, 20))
    for plot_index in range(numbers_to_display): 
        predicted_label = predictions[plot_index]
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        color_map = 1 if predicted_label == y_validation_re[plot_index] else 0
        plt.subplot(num_cells, num_cells, plot_index + 1)
        plt.imshow(x_validation_normalized[plot_index])
        plt.xlabel(predicted_label)
        plt.ylabel(color_map)
    plt.subplots_adjust(hspace=1, wspace=0.5)   
    plt.show()

foo(y_test, predicted, X_test)
# plt.figure(figsize = (12, 6))
# plt.imshow(X_test[0])
# plt.show()
