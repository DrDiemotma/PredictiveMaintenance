# Predictive Maintenance Submodule

This is a package for predictive maintenance tools written in Python.
This package can be included in projects free of charge according to the MIT license.

In order to use this module, clone the repository into your project.
After that, you can use the implemented methods.

## Installation
Clone the repository: 
```bash 
git clone git@github.com:DrDiemotma/PredictiveMaintenance.git
```

As a submodule to include in your projects: 
```bash
git submodule git@github.com:DrDiemotma/PredictiveMaintenance.git
```

## Models
In the folder Models, you will find tools to design statistical signal processing base methods for event detection.
To include those, use:
* `import PredictiveMaintenance.Models.Autoencoder` for implemented Autoencoder. Please notice that they are specially designed for event detection and might not be useful otherwise.
* `import PredictiveMaintenance.Models.Predictors` for the predictor models.

## How to Use
Examples can be found in the `Tests` folder.
Most present classes are tested here.

The package assumes that a generator is provided to present sequences.
Those generators can be wrapped using `PredictiveMaintenance.Models.AutoEncoder.make_tf_dataset`.
Notice that the generator should yield both data sequence and label sequence.
For an autoencoder, those sequences are the same.

Once the TensorFlow dataset is generated, one of the autoencoder in `PredictiveMaintenance.Models.Autoencoder` can be used.
Furthermore, the package provides a CUSUM and a EWMA test for patient even detection.
Those are in `PredictiveMaintenance.Models.Aggregation`.

### Predictors
The main tools are provided in `PredictiveMaintenance.Models.Predictors`.
Here, you will find the `AutoencoderPredictor` class.
The `AutoencoderPredictor` configures the autoencoder type and provides the two methods `fit` and `predict`.
This tool allows to transform a sequence into a prediction of deviations and applies a threshold to the results.
That means, it detects deviations of the sequence is not to be predicted using the underlying autoencoder.

Notice that those predictors "predict" a sequence after a reconstruction.
The autoencoder predicts sequences of the same length, which is very commonly the same as the input sequence.
Because of the latent dimension of the bottleneck layers, the values are not simply passed forward through the model.
That said, make sure that `latent_dim` is set smaller than your dimension times sequence length in this case.

Autoencoder in this setup are:
 * GRU
 * LSTM
 * Transformer

The default is GRU.
This should work for most series.
Use the Transformer autoencoder for very long and high dimensional series only, since they tend to overfit your training data.

## Applications
The application is for measuring deviations.
Prepare your data with a generator callback with signature 

```python
from collections.abc import Callable, Generator
import numpy as np

generator_func: Callable[[], Generator[tuple[np.typing.NDArray, np.typing.NDArray], None, None]]
```
That is a function which returns a generator which then yields to sequences of equal type `dtype=np.float32`.
Notice that the generator is consumed by training and threshold estimation which is used several times.
Hence, the function must return fresh generators:

```python
from collections.abc import Generator
import numpy as np

def func() -> Generator[tuple[np.typing.NDArray, np.typing.NDArray], None, None]:
    ...

```

With this, you use `func`, not `func()`, when using the autoencoder predictor.
