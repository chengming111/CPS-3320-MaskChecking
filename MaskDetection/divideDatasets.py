
# This is the code that handles the data set, 
# the data set that we handed in has been handled. 
# The processed data set is used in the main program.


## 2. Import the Keras library and divide the dataset

import keras
import os, shutil 

original_dataset_dir0 = '/Users/chengming/Desktop/datasets/mldata/mask1'
original_dataset_dir1 = '/Users/chengming/Desktop/datasets/mldata/nomask'

# Creat a new folder
base_dir = '/Users/chengming/Desktop/datasets/mldata/mask_small'
os.mkdir(base_dir)

# Creat the training, validation and testing diractory
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
# Directory of training pictures with masks
train_mask_dir = os.path.join(train_dir, 'mask')


# Directory of training pictures without masks
train_unmask_dir = os.path.join(train_dir, 'unmask')


# Diractory of validation pictures with masks
validation_mask_dir = os.path.join(validation_dir, 'mask')


# Diractory of validation pictures without masks
validation_unmask_dir = os.path.join(validation_dir, 'unmask')


# Diractory of testing pictures with masks
test_mask_dir = os.path.join(test_dir, 'mask')


# Diractory of testing pictures without masks
test_unmask_dir = os.path.join(test_dir, 'unmask')


# Copy 500 images of wearing masks to train_mask_dir
fnames = ['{}.jpg'.format(i) for i in range(1,500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir0, fname)
    dst = os.path.join(train_mask_dir, fname)
    shutil.copyfile(src, dst)

# Copy 500 images of wearing masks tovalidation_mask_dir
fnames = ['{}.jpg'.format(i) for i in range(1,500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir0, fname)
    dst = os.path.join(validation_mask_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy 500 images of wearing masks totest_mask_dir
fnames = ['{}.jpg'.format(i) for i in range(1,500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir0, fname)
    dst = os.path.join(test_mask_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy 1000 images of not wearing masks totrain_unmask_dir
fnames = ['{}.jpg'.format(i) for i in range(1,1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir1, fname)
    dst = os.path.join(train_unmask_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy 500 images of not wearing masks tovalidation_unmask_dir
fnames = ['{}.jpg'.format(i) for i in range(500, 1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir1, fname)
    dst = os.path.join(validation_unmask_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy 500 images of not wearing masks to test_unmask_dir
fnames = ['{}.jpg'.format(i) for i in range(1800, 2136)]
for fname in fnames:
    src = os.path.join(original_dataset_dir1, fname)
    dst = os.path.join(test_unmask_dir, fname)
    shutil.copyfile(src, dst)