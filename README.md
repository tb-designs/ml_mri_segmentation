# Machine Learning Model for Glioma Detection

The code in this repository implements a U-Net CNN architecture
to perform image segmentation on MRI scans. Because the implementation uses a number of libraries and the Keras DataGenerator to handle the data, ensuring proper package installation as well as correct directory structure is key for proper functioning of the code. This document will provide instructions for how to set up and run the model.

## Libraries

First, this code was written using Python version 3.8.3 and so ensure that this is the enabled Python release on your machine, this program will not run using the 2.0 versions.

Use the package manager of your choice to install the following packages in the environment (I use [pip](https://pip.pypa.io/en/stable/))

* numpy==1.19.5
* opencv-python==4.5.2.54
* pandas==1.2.4
* matplotlib==3.4.2
* nilearn==0.7.1
* nibabel==3.2.1
* keras==2.4.3
* tensorflow==2.5.0
* sklearn==0.24.2

Here's an example of the install command:

```bash
pip install tensorflow==2.5.0
```

## Data

This model makes use of the 2020 [BRaTS](https://www.kaggle.com/awsaf49/brats20-dataset-training-validation) dataset (available by clicking the link and choosing "Download"). The dataset is quite large at 40 GB and is downloaded in the form of a .zip file. 

## Directory Structure

The model requires a particular directory structure for the Keras DataGenerator to function correctly. Thus, care must be taken when unzipping the BRaTS dataset. The following is the correct directory tree structure

```bash
ryanr@LAPTOP-0T9QK8SP /cygdrive/c/Users/ryanr/Desktop/ECE470/project
$ tree -L 2
.
├── BraTS2020_TrainingData
│   └── MICCAI_BraTS2020_TrainingData
├── BraTS2020_ValidationData
│   └── MICCAI_BraTS2020_ValidationData
└── code
    ├── README.md
    ├── training.log
    ├── main.py
    ├── model_x1_1.h5
    └── utils.py
```

Note that the name of the directory "code" is not relevant, only that the training and validation dataset directories be at the same level as the folder containing the code and model.

## Running the Model

Once all the necessary libraries are intalled and the proper directory structure is set up, the model can be run by navigating to the "code" directory and running the following command:
```bash
py main.py
```
Depending on the choice of script options, the relevant results will be displayed in the terminal

## Script Options

The main.py script contains some options at the beginning of the file to control its behaviour:

```python
# Enable/Disable various sections of the script
PLOT_SHOW = True
TRAIN = False
TEST = True
SANITY_TEST = False
```
These options allow for:
* enabling/disabling of plots
* enabling/disabling training the model
* enabling/disabling of running the model on test data
* enabling/disabling the sanity testing of retrieving data

The script defaults are the settings shown above.

## Other
When training the model, checkpoints are used to save model progress so don't be surprised by additional files popping up in the directory when training.

For any questions on getting the script to run (in reference to ECE470 project marking) feel free to contact me at:
ryanblais@uvic.ca 

## License
[MIT](https://choosealicense.com/licenses/mit/)