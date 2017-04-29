import numpy as np
import pandas as pd
import h5py
import os
import caffe
import time

# Paths
CAFFE_HOME = "/mnt/c/caffe/"
DATA_HOME = "../data/"
FEATURES_HOME = '../features/'

# Model creation
# Using bvlc_reference_caffenet model for training
model = caffe.Net(CAFFE_HOME + 'models/bvlc_reference_caffenet/deploy.prototxt',
                  CAFFE_HOME + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                  caffe.TEST)
transformer = caffe.io.Transformer({'data': model.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.load(CAFFE_HOME + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))


def extract_features(image_paths):
    """
        This function is used to extract feature from the current batch of photos.
        Features are extracted using the pretrained bvlc_reference_caffenet
        Instead of returning 1000-dim vector from SoftMax layer, using fc7 as the final layer to get 4096-dim vector
    """
    train_size = len(image_paths)
    model.blobs['data'].reshape(train_size, 3, 227, 227)
    model.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data', caffe.io.load_image(x)), image_paths)
    out = model.forward()
    return model.blobs['fc7'].data


if not os.path.isfile(FEATURES_HOME + 'train_features.h5'):
    """
        If this file doesn't exist, create a new one and set up two columns: photoId, feature
    """
    file = h5py.File(FEATURES_HOME + 'train_features.h5', 'w')
    photoId = file.create_dataset('photoId', (0,), maxshape=(None,), dtype='|S54')
    feature = file.create_dataset('feature', (0, 4096), maxshape=(None, 4096))
    file.close()

# If this file exists, then track how many of the images are already done.
file = h5py.File(FEATURES_HOME + 'train_features.h5', 'r+')
already_extracted_images = len(file['photoId'])
file.close()

# Get training images and their business ids
train_data = pd.read_csv(DATA_HOME + 'train_photo_to_biz_ids.csv')
train_photo_paths = [os.path.join(DATA_HOME + 'train_photos/', str(photo_id) + '.jpg') for photo_id in
                     train_data['photo_id']]

# Each batch will have 500 images for feature extraction
train_size = len(train_photo_paths)
batch_size = 500
batch_number = already_extracted_images / batch_size + 1

print "Total images:", train_size
print "already_done_images: ", already_extracted_images

# Feature extraction of the train dataset
for image_count in range(already_extracted_images, train_size, batch_size):
    start_time = time.time()
    # Get the paths for images in the current batch
    image_paths = train_photo_paths[image_count: min(image_count + batch_size, train_size)]

    # Feature extraction for the current batch
    features = extract_features(image_paths)

    # Update the total count of images done so far
    total_done_images = image_count + features.shape[0]

    # Storing the features in h5 file
    file = h5py.File(FEATURES_HOME + 'train_features.h5', 'r+')
    file['photoId'].resize((total_done_images,))
    file['photoId'][image_count: total_done_images] = np.array(image_paths)
    file['feature'].resize((total_done_images, features.shape[1]))
    file['feature'][image_count: total_done_images, :] = features
    file.close()

    print "Batch No:", batch_number, "\tStart:", image_count, "\tEnd:", image_count + batch_size, "\tTime required:", time.time() - start_time, "sec", "\tCompleted:", float(
        image_count + batch_size) / float(train_size) * 100, "%"
    batch_number += 1
