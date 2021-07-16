# TODO Liscense stuff here!



# for file access to use data generator
import os

# math operations and image resizing
import numpy as np
import cv2

# pandas used for reading .csv
import pandas as pd

# for plotting results
import matplotlib.pyplot as plt

# neural imaging library: NiBabel
# for NlfTI file handling
# Available: https://nipy.org/nibabel/
import nilearn as nl
import nibabel as nib
import nilearn.plotting as nlplt

# Keras/Tensorflow imports
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

# convenient split function imported from sk-learn
from sklearn.model_selection import train_test_split

# Callback functionality
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard

# from my files
from utils import *


############################
# TEST/DEBUG PARAMS
PLOT_SHOW = True
TRAIN = False
TEST = True
SANITY_TEST = False
############################

############################
#IMPORTANT PARAMETERS 
IMG_SIZE=128 
VOLUME_SLICES = 100 
VOLUME_START_AT = 22 # first slice of volume that we will include, early slices often have no useful information
EPOCHS = 35
############################

############################
# SETUP
np.set_printoptions(precision=3, suppress=True) # Make numpy printouts easier to read.

# DEFINE segmetation-areas  
SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3 later
}

TRAIN_DATASET_PATH = '../BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
VALIDATION_DATASET_PATH = '../BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'

############################
# SANITY TESTING

# Sanity test to check the dataset paths and imaging ability
if SANITY_TEST:
    test_image_flair=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii').get_fdata()
    test_image_t1=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1.nii').get_fdata()
    test_image_t1ce=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1ce.nii').get_fdata()
    test_image_t2=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t2.nii').get_fdata()
    test_mask=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii').get_fdata()


    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (20, 10))
    slice_w = 25
    ax1.imshow(test_image_flair[:,:,test_image_flair.shape[0]//2-slice_w], cmap = 'gray')
    ax1.set_title('Image flair')
    ax2.imshow(test_image_t1[:,:,test_image_t1.shape[0]//2-slice_w], cmap = 'gray')
    ax2.set_title('Image t1')
    ax3.imshow(test_image_t1ce[:,:,test_image_t1ce.shape[0]//2-slice_w], cmap = 'gray')
    ax3.set_title('Image t1ce')
    ax4.imshow(test_image_t2[:,:,test_image_t2.shape[0]//2-slice_w], cmap = 'gray')
    ax4.set_title('Image t2')
    ax5.imshow(test_mask[:,:,test_mask.shape[0]//2-slice_w])
    ax5.set_title('Mask')

    niimg = nl.image.load_img(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii')
    nimask = nl.image.load_img(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii')

    fig, axes = plt.subplots(nrows=4, figsize=(30, 40))


    nlplt.plot_anat(niimg,
                    title='BraTS20_Training_001_flair.nii plot_anat',
                    axes=axes[0])

    nlplt.plot_epi(niimg,
                   title='BraTS20_Training_001_flair.nii plot_epi',
                   axes=axes[1])

    nlplt.plot_img(niimg,
                   title='BraTS20_Training_001_flair.nii plot_img',
                   axes=axes[2])

    nlplt.plot_roi(nimask, 
                   title='BraTS20_Training_001_flair.nii with mask plot_roi',
                   bg_img=niimg, 
                   axes=axes[3], cmap='Paired')

    print("Ran sanity check")

    plt.show()



############################
#
#   @fname: build_unet
#   @param: inputs   --> the input layer for the model
#           ker_init --> layer weights initialization method 
#           dropout  --> percentage of nodes dropped in the
#                        dropout layer
#   @desc:  Builds the u-net CNN model based on the following source:
#           https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
#           Note: number of filters at each layer adjusted for 128x128
#           image size
#
def build_unet(inputs, ker_init, dropout):
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv1)
    
    pool = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool)
    conv = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv3)
    
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv5)
    drop5 = Dropout(dropout)(conv5)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(drop5))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv9)
    
    up = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv9))
    merge = concatenate([conv1,up], axis = 3)
    conv = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge)
    conv = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv)
    
    conv10 = Conv2D(4, (1,1), activation = 'softmax')(conv)
    
    return Model(inputs = inputs, outputs = conv10)


# Define the input layer 
input_layer = Input((IMG_SIZE, IMG_SIZE, 2))

# Build u-net model
print("Building u-net")
model = build_unet(input_layer, 'he_normal', 0.2)

#compile u-net model
print("Compiling u-net")
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing] )

# Obtain directories containing the datasets (as each sample is stored in a seperate folder)
train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]

# file BraTS20_Training_355 has ill formatted name for for seg.nii file
# so we remove it if it exists (hasn't already been removed)
train_and_val_directories.remove(TRAIN_DATASET_PATH+'BraTS20_Training_355')

# convert directories list into id's
train_and_test_ids = pathListIntoIds(train_and_val_directories); 
    
# split training and testing datasets
train_test_ids, val_ids = train_test_split(train_and_test_ids,test_size=0.2) 
train_ids, test_ids = train_test_split(train_test_ids,test_size=0.15) 

# Create the DataGenerator objects for the training, testing, and validation datasets 
training_generator = DataGenerator(train_ids)
valid_generator = DataGenerator(val_ids)
test_generator = DataGenerator(test_ids)

# define csv_logger for callback use
csv_logger = CSVLogger('training.log', separator=',', append=False)

# define callbacks
callbacks = [
    # Early stopping used for debug, disabled for full training
    #keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
                                  #patience=2, verbose=1, mode='auto'),

    # Reduce the learning rate if reached a plateau in the loss function
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=2, min_lr=0.000001, verbose=1),
    
    # Save model checkpoints incase training is interrupted
    keras.callbacks.ModelCheckpoint(filepath = 'model_.{epoch:02d}-{val_loss:.6f}.m5',
                                    verbose=1, save_best_only=True, save_weights_only = True),

    # log training data as CSV
    csv_logger
]
print("Built Data Generator")
print("Clearing Keras Session")

# Clear previous Keras session
K.clear_session()

# Training Process
if TRAIN:

    print("Beginning model training")
    history =  model.fit(training_generator,
                         epochs=EPOCHS,
                         steps_per_epoch=len(train_ids),
                         callbacks= callbacks,
                         validation_data = valid_generator
                         )  
    print("Saving Trained Model")
    model.save("model_x1_1.h5")

############ load trained model ################
print("Loading Trained Model")
model = keras.models.load_model('model_x1_1.h5', 
                                   custom_objects={ 'accuracy' : tf.keras.metrics.MeanIoU(num_classes=4),
                                                   "dice_coef": dice_coef,
                                                   "precision": precision,
                                                   "sensitivity":sensitivity,
                                                   "specificity":specificity,
                                                   "dice_coef_necrotic": dice_coef_necrotic,
                                                   "dice_coef_edema": dice_coef_edema,
                                                   "dice_coef_enhancing": dice_coef_enhancing,
                                                   "dice_coef_none": dice_coef_none
                                                  }, compile=False)

history = pd.read_csv('training.log', sep=',', engine='python')

hist=history

#######################################
# Results Visualizaiton Section

print("Retreiving perfomance metrics")

# retrieve performance metrics
acc=hist['accuracy']
val_acc=hist['val_accuracy']
epoch=range(len(acc))
loss=hist['loss']
val_loss=hist['val_loss']
train_dice=hist['dice_coef']
val_dice=hist['val_dice_coef']

# Set up plots
f,ax=plt.subplots(1,4,figsize=(16,8))

# Plot accuracy over epochs
ax[0].plot(epoch,acc,'b',label='Training Accuracy')
ax[0].plot(epoch,val_acc,'r',label='Validation Accuracy')
ax[0].legend()

# Plot loss over epochs
ax[1].plot(epoch,loss,'b',label='Training Loss')
ax[1].plot(epoch,val_loss,'r',label='Validation Loss')
ax[1].legend()

# Plot overall dice loss over epochs
ax[2].plot(epoch,train_dice,'b',label='Training dice coef')
ax[2].plot(epoch,val_dice,'r',label='Validation dice coef')
ax[2].legend()

# Plot mean IoU over epochs
ax[3].plot(epoch,hist['mean_io_u'],'b',label='Training mean IOU')
ax[3].plot(epoch,hist['val_mean_io_u'],'r',label='Validation mean IOU')
ax[3].legend()

if PLOT_SHOW: plt.show()


############################
#
#   @fname: predictByPath
#   @param: case_path --> Path to data sample directory 
#           case -------> numeric ID of data sample
#   @desc:  Function for running the u-net model on a specific image case
#
def predictByPath(case_path,case):
    files = next(os.walk(case_path))[2]
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
    #y = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE))
    
    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_flair.nii');
    flair=nib.load(vol_path).get_fdata()
    
    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_t1ce.nii');
    ce=nib.load(vol_path).get_fdata() 
    
    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_seg.nii');
    seg=nib.load(vol_path).get_fdata()  

    
    for j in range(VOLUME_SLICES):
        X[j,:,:,0] = cv2.resize(flair[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))
        X[j,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))
        #y[j,:,:] = cv2.resize(seg[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))
        
    #model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema, dice_coef_enhancing] )
    #model.evaluate(x=X,y=y[:,:,:], callbacks= callbacks)
    return model.predict(X/np.max(X), verbose=1)

############################
#
#   @fname: showPredictsById
#   @param: case  --------> numeric ID of data sample 
#           start_slice --> slice of the image to begin at
#                           used because first 50 or so
#                           slices are usually empty
#   @desc:  Function to plot the model predictions.
#           Used to generate visualizations of model
#           performance using the training set data
#
def showPredictsById(case, start_slice = 60):
    # Obtain image from the dataset and run the model
    path = f"../BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{case}"
    gt = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
    origImage = nib.load(os.path.join(path, f'BraTS20_Training_{case}_flair.nii')).get_fdata()
    p = predictByPath(path,case)

    # define classes
    core = p[:,:,:,1]
    edema= p[:,:,:,2]
    enhancing = p[:,:,:,3]

    # set up plots
    plt.figure(figsize=(18, 50))
    f, axarr = plt.subplots(1,6, figsize = (18, 50)) 

    # for each image, add brain background
    for i in range(3):
        axarr[i].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray", interpolation='none')
    
    # plot original flair image
    axarr[0].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
    axarr[0].title.set_text('Original image flair')
    # plot ground truth
    curr_gt=cv2.resize(gt[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_NEAREST)
    axarr[1].imshow(curr_gt, cmap="Reds", interpolation='none', alpha=0.3)
    axarr[1].title.set_text('Ground truth')
    #plot scan with all classes present
    axarr[2].imshow(p[start_slice,:,:,1:4], cmap="Reds", interpolation='none', alpha=0.3)
    axarr[2].title.set_text('all classes')
    #plot edema prediction
    axarr[3].imshow(edema[start_slice,:,:], cmap="RdPu", interpolation='none', alpha=0.3)
    axarr[3].title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
    # plot necrotic core prediciton
    axarr[4].imshow(core[start_slice,:,], cmap="RdPu", interpolation='none', alpha=0.3)
    axarr[4].title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
    # plot enhancing prediction
    axarr[5].imshow(enhancing[start_slice,:,], cmap="RdPu", interpolation='none', alpha=0.3)
    axarr[5].title.set_text(f'{SEGMENT_CLASSES[3]} predicted')
    if PLOT_SHOW: plt.show()

# Run the model on a number of test samples to obtain visualizations    
print("Displaying predictions")
showPredictsById(case=test_ids[0][-3:])
showPredictsById(case=test_ids[1][-3:])
showPredictsById(case=test_ids[2][-3:])
showPredictsById(case=test_ids[3][-3:])
showPredictsById(case=test_ids[4][-3:])
showPredictsById(case=test_ids[5][-3:])
showPredictsById(case=test_ids[6][-3:])

# Evaluate model on the test data
if TEST: 
    print("Compiling model for test data")
    model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema, dice_coef_enhancing] )
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(test_generator, batch_size=100, callbacks= callbacks)
    print("test loss, test acc:", results)