#   This file contains utility functions used in the
#   u-net.py script. 
# 

import keras.backend as K

# Segment Classes Reference
SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3 later
}

############################
#
#   @fname: dice_coef
#   @param: y_true --> ground truth lables
#           y_pred --> predicted labels
#           smooth --> added to avoid division by 0
#   @desc:  Implementation of the Sorensen-Dice coefficient. 
#           Is a measurement of similarity between sets with
#           a smoothing factor to prevent divide-by 0 errors.
#           Measures Dice coef including all 4 classes
#
def dice_coef(y_true, y_pred, smooth=1.0):
    n_classes = 4
    for i in range(n_classes):
        y_t_flat = K.flatten(y_true[:,:,:,i])
        y_p_flat = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_t_flat * y_p_flat)
        loss = ((2. * intersection + smooth) / (K.sum(y_t_flat) + K.sum(y_p_flat) + smooth))
        K.print_tensor(loss, message='loss value for class {} : '.format(SEGMENT_CLASSES[i]))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / n_classes
    K.print_tensor(total_loss, message=' total dice coef: ')
    return total_loss

############################
#
#   @fname: dice_coef_none
#   @param: y_true --> ground truth lables
#           y_pred --> predicted labels
#           epsilon --> added to avoid division by 0
#   @desc:  Implementation of the Sorensen-Dice coefficient 
#           specifically for class 0: No tumor 
#           Is a measurement of similarity between sets with
#           a small factor added to denominator to 
#           prevent divide-by 0 errors.
#
def dice_coef_none(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,0] * y_pred[:,:,:,0]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,0])) + K.sum(K.square(y_pred[:,:,:,0])) + epsilon)

############################
#
#   @fname: dice_coef_necrotic
#   @param: y_true --> ground truth lables
#           y_pred --> predicted labels
#           epsilon --> added to avoid division by 0
#   @desc:  Implementation of the Sorensen-Dice coefficient 
#           specifically for class 1: Necrotic 
#           Is a measurement of similarity between sets with
#           a small factor added to denominator to 
#           prevent divide-by 0 errors.
#
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,1])) + K.sum(K.square(y_pred[:,:,:,1])) + epsilon)

############################
#
#   @fname: dice_coef_edema
#   @param: y_true --> ground truth lables
#           y_pred --> predicted labels
#           epsilon --> added to avoid division by 0
#   @desc:  Implementation of the Sorensen-Dice coefficient 
#           specifically for class 2: Edema
#           Is a measurement of similarity between sets with
#           a small factor added to denominator to 
#           prevent divide-by 0 errors.
#
def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,2] * y_pred[:,:,:,2]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,2])) + K.sum(K.square(y_pred[:,:,:,2])) + epsilon)

############################
#
#   @fname: dice_coef_enhancing
#   @param: y_true --> ground truth lables
#           y_pred --> predicted labels
#           epsilon --> added to avoid division by 0
#   @desc:  Implementation of the Sorensen-Dice coefficient 
#           specifically for class 3: Enhancing
#           Is a measurement of similarity between sets with
#           a small factor added to denominator to 
#           prevent divide-by 0 errors.
#
def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,3] * y_pred[:,:,:,3]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,3])) + K.sum(K.square(y_pred[:,:,:,3])) + epsilon)

############################
#
#   @fname: precision
#   @param: y_true --> ground truth lables
#           y_pred --> predicted labels
#   @desc:  Measure of the precision of model prediction
#        
def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
############################
#
#   @fname: sensitivity
#   @param: y_true --> ground truth lables
#           y_pred --> predicted labels
#   @desc:  Measure of the sensitivity of model prediction
#           
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

############################
#
#   @fname: specificity
#   @param: y_true --> ground truth lables
#           y_pred --> predicted labels
#   @desc:  Measure of the specificity of model prediction
# 
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

############################
#
#   @fname: pathListIntoIds
#   @param: dirList   --> List of directories
#   @desc:  Converts list of directories into ids
#           so that they can be used to manipulate
#           the data more easily
#
def pathListIntoIds(dirList):
    x = []
    for i in range(0,len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    return x
    
