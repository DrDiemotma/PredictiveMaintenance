This is a package for predictive maintenance tools written in Python.
This package can be included in projects free of charge according to the MIT license.

In order to use this module, clone the repository into your project.
After that, you can use the implemented methods.

# Installation
Clone the repository: `git clone git@github.com:DrDiemotma/PredictiveMaintenance.git`

As a submodule to include in your projects: `git submodule git@github.com:DrDiemotma/PredictiveMaintenance.git`

# Models
In the folder Models, you will find tools to design statistical signal processing base methods for event detection.
To include those, use:
* `import PredictiveMaintenance.Models.Autoencoder` for implemented Autoencoder. Please notice that they are specially designed for event detection and might not be useful otherwise.
* `import PredictiveMaintenance.models.Predictors` for the predictor models.

# How to Use