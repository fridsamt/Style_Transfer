import ConfigParser
import os
import time
import cv2
import numpy as np
import scipy.io
import tensorflow as tf

# parameters
configParser = ConfigParser.RawConfigParser()
configFilePath = 'config.cfg'
configParser.read(configFilePath)
# path parameters
output_dir = configParser.get('path', 'output_dir')
content_dir = configParser.get('path', 'content_dir')
style_dir = configParser.get('path', 'style_dir')
content_image_file = configParser.get('path', 'content_images')
style_image_files = configParser.get('path', 'style_images')
style_image_files = [s.strip() for s in style_image_files.split(',')]
model_path = configParser.get('path', 'model_path')
max_image_size = configParser.get('image_param', 'max_size')
model_mean = configParser.get('model_param', 'model_mean')
model_mean = np.array([float(x.strip()) for x in model_mean.split(',')]).reshape((1, 1, 1, 3))
init_type = configParser.get('train_param', 'init_image')
device = configParser.get('train_param', 'device')
pooling_type = configParser.get('train_param', 'pooling_type')
style_image_weights = configParser.get('train_param', 'style_image_weights')
style_image_weights = [float(x.strip()) for x in style_image_weights.split(',')]
max_iterations = float(configParser.get('train_param', 'max_iterations'))
style_weight = float(configParser.get('train_param', 'style_weight'))
content_weight = float(configParser.get('train_param', 'content_weight'))
style_layers = configParser.get('train_param', 'style_layers')
style_layers = [s.strip() for s in style_layers.split(',')]
style_layer_weights = configParser.get('train_param', 'style_layer_weights')
style_layer_weights = [float(x.strip()) for x in style_layer_weights.split(',')]
content_layers = configParser.get('train_param', 'content_layers')
content_layers = [s.strip() for s in content_layers.split(',')]
content_layer_weights = configParser.get('train_param', 'content_layer_weights')
content_layer_weights = [float(x.strip()) for x in content_layer_weights.split(',')]
noise_ratio = 1

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
    style_images = get_style_images(h, w, style_image_files)
    print 'Images loaded...'
    with tf.Graph().as_default():
        print'Begain style transfer...'
        init_image = get_init_image(init_type, content_image, style_images[0])
        tick = time.time()
        transfer(content_image, style_images, init_image)
        tock = time.time()
        print('Time used: {}'.format(tock - tick))


def transfer(content_image, style_image, init_image):
    with tf.device(device), tf.Session() as sess:
        outputs = forward_input(init_image)
        # style loss
        style_loss = sum_style_losses(sess, outputs, style_image)
        # content loss
        content_loss = sum_content_losses(sess, outputs, content_image)
        # loss weights
        a = content_weight
        b = style_weight

        # total loss
        L_total = a * content_loss + b * style_loss
        print 'Initialization done. Begain training...'
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            L_total, method='L-BFGS-B',
            options={'maxiter': max_iterations,
                     'disp': 10})
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        sess.run(outputs['input'].assign(init_image))
        optimizer.minimize(sess)
        output_image = sess.run(outputs['input'])
        image_path = os.path.join(output_dir, 'output.png')
        output_image = restore_image(output_image, model_mean)
        cv2.imwrite(image_path, output_image)


def restore_image(image, mean):
    image += mean
    # shape (1, h, w, d) to (h, w, d)
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    # rgb to bgr
    image = image[..., ::-1]
    return image


def sum_style_losses(sess, outputs, style_images):
    loss = 0.
    for image, image_weights in zip(style_images, style_image_weights):
        sess.run(outputs['input'].assign(image))
        style_loss = 0.
        for layer, weight in zip(style_layers, style_layer_weights):
            a = sess.run(outputs[layer])
            x = outputs[layer]
            a = tf.convert_to_tensor(a)
            style_loss += style_layer_loss(a, x) * weight
        style_loss /= float(len(style_layers))
        loss += (style_loss * image_weights)
    loss /= float(len(style_images))
    return loss


def get_style_images(h, w, style_image_files):
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
    image = image[..., ::-1]
    # add new axis
    image = image[np.newaxis, :, :, :]
    image -= model_mean
    return image


def get_init_image(type, content_image, style_image):
    if type == 'content':
        return content_image
    elif type == 'style':
        return style_image
    elif type == 'noise':
        return get_noise_image(noise_ratio, content_image)

def get_noise_image(noise_ratio, content_image):
  np.random.seed(0)
  noise_image = np.random.uniform(-20., 20., content_image.shape).astype(np.float32)
  image = noise_ratio * noise_image + (1.-noise_ratio) * content_image
  return image

def sum_content_losses(sess, outputs, content_image):
    sess.run(outputs['input'].assign(content_image))
    content_loss = 0.
    for layer, weight in zip(content_layers, content_layer_weights):
        p = sess.run(outputs[layer])
        x = outputs[layer]
        p = tf.convert_to_tensor(p)
        content_loss += content_layer_loss(p, x) * weight
    content_loss /= float(len(content_layers))
    return content_loss


def get_layer_weights(layers, i):
    weights = layers[i][0][0][2][0][0]
    W = tf.constant(weights)
    return W


def get_layer_bias(layers, i):
    bias = layers[i][0][0][2][0][1]
    b = tf.constant(np.reshape(bias, (bias.size)))
    return b


def forward_input(input_image):
    _, h, w, d = input_image.shape
    outputs = {}
    print 'Intializing variables...'
    model = scipy.io.loadmat(model_path)
    layers = model['layers'][0]

    outputs['input'] = tf.Variable(np.zeros((1, h, w, d), dtype=np.float32))
    outputs['conv1_1'] = tf.nn.conv2d(outputs['input'], get_layer_weights(layers, 0), strides=[1, 1, 1, 1],
                                      padding='SAME')
    outputs['relu1_1'] = tf.nn.relu(outputs['conv1_1'] + get_layer_bias(layers, 0))
    outputs['conv1_2'] = tf.nn.conv2d(outputs['relu1_1'], get_layer_weights(layers, 2), strides=[1, 1, 1, 1],
                                      padding='SAME')
    outputs['relu1_2'] = tf.nn.relu(outputs['conv1_2'] + get_layer_bias(layers, 2))
    outputs['pool1'] = pooling(outputs['relu1_2'])

    outputs['conv2_1'] = tf.nn.conv2d(outputs['pool1'], get_layer_weights(layers, 5), strides=[1, 1, 1, 1],
                                      padding='SAME')
    outputs['relu2_1'] = tf.nn.relu(outputs['conv2_1'] + get_layer_bias(layers, 5))
    outputs['conv2_2'] = tf.nn.conv2d(outputs['relu2_1'], get_layer_weights(layers, 7), strides=[1, 1, 1, 1],
                                      padding='SAME')
    outputs['relu2_2'] = tf.nn.relu(outputs['conv2_2'] + get_layer_bias(layers, 7))
    outputs['pool2'] = pooling(outputs['relu2_2'])

    outputs['conv3_1'] = tf.nn.conv2d(outputs['pool2'], get_layer_weights(layers, 10), strides=[1, 1, 1, 1],
                                      padding='SAME')
    outputs['relu3_1'] = tf.nn.relu(outputs['conv3_1'] + get_layer_bias(layers, 10))
    outputs['conv3_2'] = tf.nn.conv2d(outputs['relu3_1'], get_layer_weights(layers, 12), strides=[1, 1, 1, 1],
                                      padding='SAME')
    outputs['relu3_2'] = tf.nn.relu(outputs['conv3_2'] + get_layer_bias(layers, 12))

    outputs['conv3_3'] = tf.nn.conv2d(outputs['relu3_2'], get_layer_weights(layers, 14), strides=[1, 1, 1, 1],
                                      padding='SAME')
    outputs['relu3_3'] = tf.nn.relu(outputs['conv3_3'] + get_layer_bias(layers, 14))
    outputs['conv3_4'] = tf.nn.conv2d(outputs['relu3_3'], get_layer_weights(layers, 16), strides=[1, 1, 1, 1],
                                      padding='SAME')
    outputs['relu3_4'] = tf.nn.relu(outputs['conv3_4'] + get_layer_bias(layers, 16))
    outputs['pool3'] = pooling(outputs['relu3_4'])

    outputs['conv4_1'] = tf.nn.conv2d(outputs['pool3'], get_layer_weights(layers, 19), strides=[1, 1, 1, 1],
                                      padding='SAME')
    outputs['relu4_1'] = tf.nn.relu(outputs['conv4_1'] + get_layer_bias(layers, 19))
    outputs['conv4_2'] = tf.nn.conv2d(outputs['relu4_1'], get_layer_weights(layers, 21), strides=[1, 1, 1, 1],
                                      padding='SAME')
    outputs['relu4_2'] = tf.nn.relu(outputs['conv4_2'] + get_layer_bias(layers, 21))

    outputs['conv4_3'] = tf.nn.conv2d(outputs['relu4_2'], get_layer_weights(layers, 23), strides=[1, 1, 1, 1],
                                      padding='SAME')
    outputs['relu4_3'] = tf.nn.relu(outputs['conv4_3'] + get_layer_bias(layers, 23))
    outputs['conv4_4'] = tf.nn.conv2d(outputs['relu4_3'], get_layer_weights(layers, 25), strides=[1, 1, 1, 1],
                                      padding='SAME')
    outputs['relu4_4'] = tf.nn.relu(outputs['conv4_4'] + get_layer_bias(layers, 25))
    outputs['pool4'] = pooling(outputs['relu4_4'])

    outputs['conv5_1'] = tf.nn.conv2d(outputs['pool4'], get_layer_weights(layers, 28), strides=[1, 1, 1, 1],
                                      padding='SAME')
    outputs['relu5_1'] = tf.nn.relu(outputs['conv5_1'] + get_layer_bias(layers, 28))
    outputs['conv5_2'] = tf.nn.conv2d(outputs['relu5_1'], get_layer_weights(layers, 30), strides=[1, 1, 1, 1],
                                      padding='SAME')
    outputs['relu5_2'] = tf.nn.relu(outputs['conv5_2'] + get_layer_bias(layers, 30))

    outputs['conv5_3'] = tf.nn.conv2d(outputs['relu5_2'], get_layer_weights(layers, 32), strides=[1, 1, 1, 1],
                                      padding='SAME')
    outputs['relu5_3'] = tf.nn.relu(outputs['conv5_3'] + get_layer_bias(layers, 32))
    outputs['conv5_4'] = tf.nn.conv2d(outputs['relu5_3'], get_layer_weights(layers, 34), strides=[1, 1, 1, 1],
                                      padding='SAME')
    outputs['relu5_4'] = tf.nn.relu(outputs['conv5_4'] + get_layer_bias(layers, 34))
    outputs['pool5'] = pooling(outputs['relu5_4'])

    return outputs


def style_layer_loss(a, x):
    _, h, w, d = a.get_shape()
    M = h.value * w.value
    N = d.value
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss


def content_layer_loss(p, x):
    _, h, w, d = p.get_shape()
    M = h.value * w.value
    N = d.value
    K = 1. / (2. * N ** 0.5 * M ** 0.5)
    loss = K * tf.reduce_sum(tf.pow((x - p), 2))
    return loss


def gram_matrix(x, area, depth):
    F = tf.reshape(x, (area, depth))
    G = tf.matmul(tf.transpose(F), F)
    return G


def pooling(layer_input):
    if pooling_type == 'avg':
        pool = tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
    elif pooling_type == 'max':
        pool = tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
    return pool


if __name__ == "__main__":
    main()
