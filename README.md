# nn

![](https://img.shields.io/github/languages/top/synthaze/nn) ![](https://img.shields.io/github/license/synthaze/nn) ![](https://img.shields.io/github/last-commit/synthaze/nn)

![](https://img.shields.io/github/stars/synthaze/nn?style=social) ![](https://img.shields.io/twitter/follow/nn_synthaze?label=Follow&style=social)

**nn is written in pure Python/NumPy.**

If you use nn in academia, please cite:

Malard F., Danner L., Rouzies E., Meyer J. G., Lescop E., Olivier-Van Stichelen S. [**nn: Educational python for Neural Networks**](https://www.softxjournal.com/article/S2352-7110(22)00090-5/fulltext), *SoftwareX* 19 (2022).

## Documentation

Please visit https://nn.net/ for extensive documentation.

### Purpose

nn is intended for **teachers**, **students**, **scientists**, or more generally anyone with minimal skills in Python programming **who wish to understand** and build from basic implementations of Neural Network architectures.

Although nn can be used for production, it is meant to be a library of **homogeneous architecture templates** and **practical examples** which is expected to save an important amount of time for people who wish to learn, teach or **develop from scratch**.

### Content

nn features **scalable**, **minimalistic** and **homogeneous** implementations of major Neural Network architectures in **pure Python/Numpy** including:

* [Embedding layer (Input)](https://nn.net/Embedding.html)
* [Fully connected layer (Dense)](https://nn.net/Dense.html)
* [Recurrent Neural Network (RNN)](https://nn.net/RNN.html)
* [Long Short-Term Memory (LSTM)](https://nn.net/LSTM.html)
* [Gated Recurrent Unit (GRU)](https://nn.net/GRU.html)
* [Convolution (CNN)](https://nn.net/Convolution.html)
* [Pooling (CNN)](https://nn.net/Pooling.html)
* [Dropout - Regularization](https://nn.net/Dropout.html)
* [Flatten - Adapter](https://nn.net/Flatten.html)

Model and function rules and definition:

* [Architecture Layers - Model](https://nn.net/nn_Model.html)
* [Neural Network - Model](https://nn.net/Layer_Model.html)
* [Data - Model](https://nn.net/Data_Model.html)
* [Activation - Functions](https://nn.net/activation.html)
* [Loss - Functions](https://nn.net/loss.html)

While not enhancing, extending or replacing nn's documentation, series of live examples in Python and Jupyter notebook formats are offered online and within the archive, including:

* [Data preparation - Examples](https://nn.net/data_examples.html)
* [Network training - Examples](https://nn.net/run_examples.html)

### Reliability

nn has been cross-validated against TensorFlow/Keras API and provides identical results for identical configurations in the limit of float64 precision.

Please see [Is nn reliable?](https://nn.net/index.html#is-nn-reliable) for details and executable codes.

### Recommended install

* **Linux/MacOS**

```bash

# Use bash shell
bash

# Clone git repository
git clone https://github.com/synthaze/nn

# Change directory to nn
cd nn

# Install nn dependencies
pip3 install -r requirements.txt

# Export nn path in $PYTHONPATH for current session
export PYTHONPATH=$PYTHONPATH:$PWD

# Alternatively, not recommended
# pip3 install nn
# nn
```

**Linux:** Permanent export of nn directory path in ```$PYTHONPATH```.

```bash
# Append export instruction to the end of .bashrc file
echo "export PYTHONPATH=$PYTHONPATH:$PWD" >> ~/.bashrc

# Source .bashrc to refresh $PYTHONPATH
source ~/.bashrc
```

**MacOS:** Permanent export of nn directory path in ```$PYTHONPATH```.

```bash
# Append export instruction to the end of .bash_profile file
echo "export PYTHONPATH=$PYTHONPATH:$PWD" >> ~/.bash_profile

# Source .bash_profile to refresh $PYTHONPATH
source ~/.bash_profile
```

* **Windows**

```bash
# Clone git repository
git clone https://github.com/synthaze/nn

# Change directory to nn
chdir nn

# Install nn dependencies
pip3 install -r requirements.txt

# Show full path of nn directory
echo %cd%

# Alternatively, not recommended
# pip3 install nn
# nn
```

Copy the full path of nn directory, then go to:
``Control Panel > System > Advanced > Environment variable``

If you already have ``PYTHONPATH`` in the ``User variables`` section, select it and click ``Edit``, otherwise click ``New`` to add it.

Paste the full path of nn directory in the input field, keep in mind that paths in ``PYTHONPATH`` should be comma-separated.

ANSI coloring schemes do work on native Windows10 and later. For prior Windows versions, users should configure their environment to work with ANSI coloring schemes for optimal experience.

## Current release

### 1.2 Publication release

* Minor revisions for peer-review process.

See [CHANGELOG.md](CHANGELOG.md) for past releases.



## Project tree

**nn**
 * [convolution](nn/convolution)
   * [backward.py](nn/convolution/backward.py)
   * [forward.py](nn/convolution/forward.py)
   * [models.py](nn/convolution/models.py)
   * [parameters.py](nn/convolution/parameters.py)
 * [dense](nn/dense)
   * [backward.py](nn/dense/backward.py)
   * [forward.py](nn/dense/forward.py)
   * [models.py](nn/dense/models.py)
   * [parameters.py](nn/dense/parameters.py)
 * [dropout](nn/dropout)
   * [backward.py](nn/dropout/backward.py)
   * [forward.py](nn/dropout/forward.py)
   * [models.py](nn/dropout/models.py)
   * [parameters.py](nn/dropout/parameters.py)
 * [embedding](nn/embedding)
   * [backward.py](nn/embedding/backward.py)
   * [dataset.py](nn/embedding/dataset.py)
   * [forward.py](nn/embedding/forward.py)
   * [models.py](nn/embedding/models.py)
   * [parameters.py](nn/embedding/parameters.py)
 * [flatten](nn/flatten)
   * [backward.py](nn/flatten/backward.py)
   * [forward.py](nn/flatten/forward.py)
   * [models.py](nn/flatten/models.py)
   * [parameters.py](nn/flatten/parameters.py)
 * [gru](nn/gru)
   * [backward.py](nn/gru/backward.py)
   * [forward.py](nn/gru/forward.py)
   * [models.py](nn/gru/models.py)
   * [parameters.py](nn/gru/parameters.py)
 * [lstm](nn/lstm)
   * [backward.py](nn/lstm/backward.py)
   * [forward.py](nn/lstm/forward.py)
   * [models.py](nn/lstm/models.py)
   * [parameters.py](nn/lstm/parameters.py)
 * [pooling](nn/pooling)
   * [backward.py](nn/pooling/backward.py)
   * [forward.py](nn/pooling/forward.py)
   * [models.py](nn/pooling/models.py)
   * [parameters.py](nn/pooling/parameters.py)
 * [rnn](nn/rnn)
   * [backward.py](nn/rnn/backward.py)
   * [forward.py](nn/rnn/forward.py)
   * [models.py](nn/rnn/models.py)
   * [parameters.py](nn/rnn/parameters.py)
 * [template](nn/template)
     * [backward.py](nn/template/backward.py)
     * [forward.py](nn/template/forward.py)
     * [models.py](nn/template/models.py)
     * [parameters.py](nn/template/parameters.py)
 * [network](nn/network)
   * [backward.py](nn/network/backward.py)
   * [evaluate.py](nn/network/evaluate.py)
   * [forward.py](nn/network/forward.py)
   * [hyperparameters.py](nn/network/hyperparameters.py)
   * [initialize.py](nn/network/initialize.py)
   * [models.py](nn/network/models.py)
   * [report.py](nn/network/report.py)
   * [training.py](nn/network/training.py)
 * [commons](nn/commons)
   * [io.py](nn/commons/io.py)
   * [library.py](nn/commons/library.py)
   * [logs.py](nn/commons/logs.py)
   * [loss.py](nn/commons/loss.py)
   * [maths.py](nn/commons/maths.py)
   * [metrics.py](nn/commons/metrics.py)
   * [models.py](nn/commons/models.py)
   * [plot.py](nn/commons/plot.py)
   * [schedule.py](nn/commons/schedule.py)

**nn_train**
 * [author_music](nn_train/author_music)
 * [captcha_mnist](nn_train/captcha_mnist)
 * [dummy_boolean](nn_train/dummy_boolean)
 * [image](nn_train/image)
 * [dummy_string](nn_train/dummy_string)
 * [dummy_time](nn_train/dummy_time)
 * [ptm_protein](nn_train/ptm_protein)
