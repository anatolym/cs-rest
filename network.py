# -*- coding: utf-8 -*-
"""
    CS - ColorSeason
    ~~~~~~
    ColorSeason application for image recognition (classification) by trained neural network model.
    :copyright: (c) 2016 by Anatoly Milkov <anatoly.milko@gmail.com>.
"""
import os
import sys
import numpy as np
import caffe

# caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
# sys.path.insert(0, caffe_root + 'python')


class Network(object):
    """docstring for ClassName"""

    def __init__(self):
        self.net = None
        self.transformer = None
        self.labels = None

    def load_model(self, root_path):
        caffe_root = os.path.join(root_path, 'caffe')
        caffe.set_mode_cpu()

        model_def = caffe_root + '/model/network.prototxt'
        model_weights = caffe_root + '/model/model.caffemodel'
        # model_def = caffe_root + '/model/ilsvrc12-deploy.prototxt'
        # model_weights = caffe_root + '/model/ilsvrc12-model.caffemodel'
        if os.path.isfile(model_def) and os.path.isfile(model_weights):
            print('Model and weights are found.')
        else:
            print('ERROR: Model and weights are not found.')
            exit()

        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)

        # load the mean ImageNet image (as distributed with Caffe) for
        # subtraction
        mu = np.load(
            caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        # average over pixels to obtain the mean (BGR) pixel values
        mu = mu.mean(1).mean(1)
        print('mean-subtracted values:', list(zip('BGR', mu)))

        # create transformer for the input called 'data'
        self.transformer = caffe.io.Transformer(
            {'data': self.net.blobs['data'].data.shape})

        # move image channels to outermost dimension
        self.transformer.set_transpose('data', (2, 0, 1))
        # subtract the dataset-mean value in each channel
        self.transformer.set_mean('data', mu)
        # rescale from [0, 1] to [0, 255]
        self.transformer.set_raw_scale('data', 255)
        # swap channels from RGB to BGR
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # set the size of the input (we can skip this if we're happy
        # with the default; we can also change it later, e.g., for different
        # batch sizes)
        self.net.blobs['data'].reshape(50,        # batch size
                                       3,         # 3-channel (BGR) images
                                       227, 227)  # image size is 227x227

        # load ImageNet labels
        labels_file = caffe_root + '/model/labels.txt'
        # labels_file = caffe_root + '/data/ilsvrc12/synset_words.txt'
        if not os.path.exists(labels_file):
            # !../data/ilsvrc12/get_ilsvrc_aux.sh
            print('ERROR: "%s" not found.' % labels_file)
            exit()

        # self.labels = np.loadtxt(labels_file, dtype=str, delimiter='\t')
        # Reading file by ourselves. np.loadtxt has bugs with sbite-to-str convertion.
        # See https://github.com/numpy/numpy/issues/2715
        self.labels = np.array(open(labels_file, 'r').read().splitlines())

    def test_image(self, image_path):
        image = caffe.io.load_image(image_path)
        transformed_image = self.transformer.preprocess('data', image)

        # copy the image data into the memory allocated for the net
        self.net.blobs['data'].data[...] = transformed_image

        # perform classification
        output = self.net.forward()

        # the output probability vector for the first image in the batch
        # output_prob = output['prob'][0]  # For ImageNet network ilsvrc12-deploy.prototxt
        output_prob = output['prob'][0]
        class_id = str(output_prob.argmax())
        class_probability = str(output_prob.max())
        # print('predicted class is:', class_id)

        class_label = self.labels[output_prob.argmax()]
        # print ('output label:', class_label)

        # sort top five predictions from softmax output
        # reverse sort and take five largest items
        top_inds = output_prob.argsort()[::-1][:5]

        # print('probabilities and    :')
        # print(list(zip(output_prob[top_inds], self.labels[top_inds])))

        return dict(
            class_id=class_id,
            class_probability=class_probability,
            class_label=class_label,
            top_inds=list(
                zip(output_prob[top_inds].tolist(), self.labels[top_inds].tolist()))
        )
