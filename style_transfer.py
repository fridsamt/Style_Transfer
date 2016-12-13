import tensorflow as tf
import numpy as np
import cv2
import ConfigParser
import os
import time

# parameters
configParser = ConfigParser.RawConfigParser()
configFilePath = 'config.cfg'
configParser.read(configFilePath)
#path parameters
content_dir = configParser.get('path', 'content_dir')
style_dir = configParser.get('path', 'style_dir')
content_image_file = configParser.get('path', 'content_images')
style_image_files = configParser.get('path', 'style_images')
style_image_files = [s.strip() for s in style_image_files.split(',')]

max_image_size = configParser.get('image_param', 'max_size')

model_mean = configParser.get('model_param', 'model_mean')
model_mean = np.array([float(x.strip()) for x in model_mean.split(',')]).reshape((1,1,1,3))

init_type = configParser.get('train_param', 'init_image')

def resize(image):
    h, w, _ = image.shape
    if h > max_image_size:
        w = (float(max_image_size) / float(h)) * w
        h = max_image_size
    else:
        w = max_image_size
        h = (float(max_image_size) / float(w)) * h
    image = cv2.resize(image, dsize=(int(w), int(h)), interpolation=cv2.INTER_AREA)
    return image

def main():
    content_image = get_content_image(content_image_file)
    _, h, w, _ = content_image.shape
    style_images = get_style_images(h,w,style_image_files)
    print 'Images loaded...'
    with tf.Graph().as_default():
        print'Begain style transfer...'
        init_image = get_init_image(init_type, content_image, style_images[0])
        tick = time.time()
        transfer(content_image, style_images, init_image)
        tock = time.time()
        print('Time used: {}'.format(tock - tick))
def transfer(content_image, style_image, init_image):
    pass

def get_style_images(h,w,style_image_files):
    images = []
    for file in style_image_files:
        path = os.path.join(style_dir, file)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = image.astype(np.float32)
        image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_AREA)
        # bgr to rgb
        image = image[..., ::-1]
        # add new axis
        image = image[np.newaxis, :, :, :]
        image -= model_mean
        images.append(image)
    return images

def get_content_image(content_images):
  path = os.path.join(content_dir, content_images)
  image = cv2.imread(path, cv2.IMREAD_COLOR)
  image = image.astype(np.float32)
  h, w, _ = image.shape
  if h > max_image_size or w > max_image_size:
      image = resize(image)
  # bgr to rgb
  image = image[...,::-1]
  # add new axis
  image = image[np.newaxis,:,:,:]
  image -= model_mean
  return image

def get_init_image(type, content_image, style_image):
    if type == 'content':
        return content_image
    elif type == 'style':
        return style_image
    elif type == 'noise':
        return content_image

#
# def build_vgg19(input_img):
#     if args.verbose: print('\nBUILDING VGG-19 NETWORK')
#     net = {}
#     _, h, w, d = input_img.shape
#
#     if args.verbose: print('loading model weights...')
#     vgg_rawnet = scipy.io.loadmat(args.model_weights)
#     vgg_layers = vgg_rawnet['layers'][0]
#     if args.verbose: print('constructing layers...')
#     net['input'] = tf.Variable(np.zeros((1, h, w, d), dtype=np.float32))
#
#     if args.verbose: print('LAYER GROUP 1')
#     net['conv1_1'] = conv_layer('conv1_1', net['input'], W=get_weights(vgg_layers, 0))
#     net['relu1_1'] = relu_layer('relu1_1', net['conv1_1'], b=get_bias(vgg_layers, 0))
#
#     net['conv1_2'] = conv_layer('conv1_2', net['relu1_1'], W=get_weights(vgg_layers, 2))
#     net['relu1_2'] = relu_layer('relu1_2', net['conv1_2'], b=get_bias(vgg_layers, 2))
#
#     net['pool1'] = pool_layer('pool1', net['relu1_2'])
#
#     if args.verbose: print('LAYER GROUP 2')
#     net['conv2_1'] = conv_layer('conv2_1', net['pool1'], W=get_weights(vgg_layers, 5))
#     net['relu2_1'] = relu_layer('relu2_1', net['conv2_1'], b=get_bias(vgg_layers, 5))
#
#     net['conv2_2'] = conv_layer('conv2_2', net['relu2_1'], W=get_weights(vgg_layers, 7))
#     net['relu2_2'] = relu_layer('relu2_2', net['conv2_2'], b=get_bias(vgg_layers, 7))
#
#     net['pool2'] = pool_layer('pool2', net['relu2_2'])
#
#     if args.verbose: print('LAYER GROUP 3')
#     net['conv3_1'] = conv_layer('conv3_1', net['pool2'], W=get_weights(vgg_layers, 10))
#     net['relu3_1'] = relu_layer('relu3_1', net['conv3_1'], b=get_bias(vgg_layers, 10))
#
#     net['conv3_2'] = conv_layer('conv3_2', net['relu3_1'], W=get_weights(vgg_layers, 12))
#     net['relu3_2'] = relu_layer('relu3_2', net['conv3_2'], b=get_bias(vgg_layers, 12))
#
#     net['conv3_3'] = conv_layer('conv3_3', net['relu3_2'], W=get_weights(vgg_layers, 14))
#     net['relu3_3'] = relu_layer('relu3_3', net['conv3_3'], b=get_bias(vgg_layers, 14))
#
#     net['conv3_4'] = conv_layer('conv3_4', net['relu3_3'], W=get_weights(vgg_layers, 16))
#     net['relu3_4'] = relu_layer('relu3_4', net['conv3_4'], b=get_bias(vgg_layers, 16))
#
#     net['pool3'] = pool_layer('pool3', net['relu3_4'])
#
#     if args.verbose: print('LAYER GROUP 4')
#     net['conv4_1'] = conv_layer('conv4_1', net['pool3'], W=get_weights(vgg_layers, 19))
#     net['relu4_1'] = relu_layer('relu4_1', net['conv4_1'], b=get_bias(vgg_layers, 19))
#
#     net['conv4_2'] = conv_layer('conv4_2', net['relu4_1'], W=get_weights(vgg_layers, 21))
#     net['relu4_2'] = relu_layer('relu4_2', net['conv4_2'], b=get_bias(vgg_layers, 21))
#
#     net['conv4_3'] = conv_layer('conv4_3', net['relu4_2'], W=get_weights(vgg_layers, 23))
#     net['relu4_3'] = relu_layer('relu4_3', net['conv4_3'], b=get_bias(vgg_layers, 23))
#
#     net['conv4_4'] = conv_layer('conv4_4', net['relu4_3'], W=get_weights(vgg_layers, 25))
#     net['relu4_4'] = relu_layer('relu4_4', net['conv4_4'], b=get_bias(vgg_layers, 25))
#
#     net['pool4'] = pool_layer('pool4', net['relu4_4'])
#
#     if args.verbose: print('LAYER GROUP 5')
#     net['conv5_1'] = conv_layer('conv5_1', net['pool4'], W=get_weights(vgg_layers, 28))
#     net['relu5_1'] = relu_layer('relu5_1', net['conv5_1'], b=get_bias(vgg_layers, 28))
#
#     net['conv5_2'] = conv_layer('conv5_2', net['relu5_1'], W=get_weights(vgg_layers, 30))
#     net['relu5_2'] = relu_layer('relu5_2', net['conv5_2'], b=get_bias(vgg_layers, 30))
#
#     net['conv5_3'] = conv_layer('conv5_3', net['relu5_2'], W=get_weights(vgg_layers, 32))
#     net['relu5_3'] = relu_layer('relu5_3', net['conv5_3'], b=get_bias(vgg_layers, 32))
#
#     net['conv5_4'] = conv_layer('conv5_4', net['relu5_3'], W=get_weights(vgg_layers, 34))
#     net['relu5_4'] = relu_layer('relu5_4', net['conv5_4'], b=get_bias(vgg_layers, 34))
#
#     net['pool5'] = pool_layer('pool5', net['relu5_4'])
#
#     return net
if __name__ == "__main__":
    main()