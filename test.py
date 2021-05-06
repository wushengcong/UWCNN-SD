from __future__ import print_function
from utils import *
from model import UWCNN
import pprint
from glob import glob
import cv2
import os
import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_integer("epoch", 30, "Number of epoch [30]")
flags.DEFINE_string("checkpoint_dir", "UWCNN-SD", "Name of checkpoint directory [checkpoint]")

FLAGS = flags.FLAGS
pp = pprint.PrettyPrinter()


def main():
    output_path = "Outputs/"
    data_hf = glob('./DCT/HF/*.*')
    data_lf = glob('./DCT/LF/*.*')

    with tf.Session() as sess:
        model = UWCNN(sess)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        checkpoint = "checkpoint/" + FLAGS.checkpoint_dir + "/UWCNN.model-" + str(FLAGS.epoch-1)
        saver = tf.train.import_meta_graph(checkpoint + ".meta")

        saver.restore(sess, checkpoint)

        for i in range(len(data_hf)):
            img_hf = data_hf[i]
            img_lf = data_lf[i]

            test_image_name, _ = os.path.splitext(os.path.basename(img_hf))
            hf_img = cv_norm(cv2.imread(img_hf))
            lf_img = cv_norm(cv2.imread(img_lf))
            shape_x, shape_y, shape_z = hf_img.shape

            hf_images = tf.placeholder(tf.float32, [1, shape_x, shape_y, shape_z], name='hf_image')
            lf_images = tf.placeholder(tf.float32, [1, shape_x, shape_y, shape_z], name='lf_image')
            output = model.CNNnet(hf_images, lf_images, reuse=True, name='CNNnet')

            hf_img = np.array(hf_img).reshape(1, shape_x, shape_y, shape_z)
            lf_img = np.array(lf_img).reshape(1, shape_x, shape_y, shape_z)

            output_img= sess.run(output, feed_dict={hf_images: hf_img, lf_images: lf_img})
            output_img = output_img.squeeze()
            output_img = cv_inv_norm(output_img)

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            output_name = output_path + "/" + str(test_image_name) + ".png"
            cv2.imwrite(output_name, output_img)
            print('image:{:d}'.format(i+1))
    tf.reset_default_graph()


if __name__ == '__main__':
    main()








