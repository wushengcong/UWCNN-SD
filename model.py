from utils import *
import tensorflow as tf


class UWCNN(object):
    def __init__(self,
                 sess,
                 image_size=256,
                 label_size=256,
                 batch_size=16,
                 c_dim=3,
                 datasets=None,
                 checkpoint_dir=None,
                 sample_dir=None):

        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size

        self.c_dim = c_dim
        self.datasets_dir = datasets
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.build_model()

    def build_model(self):
        with tf.name_scope('HF_input'):
            self.images_hf = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.c_dim],
                                            name='images_hf')

        with tf.name_scope('LF_input'):
            self.images_lf = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.c_dim],
                                            name='images_lf')

        with tf.name_scope('output'):
            self.output = self.CNNnet(self.images_hf, self.images_lf, reuse=False, name='CNNnet')

        self.saver = tf.train.Saver(max_to_keep=50)

    def CNNnet(self, hf_image, lf_image, gf_dim=32, reuse=False, name='CNNnet'):

        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            lf_c1 = relu(conv2d(input_=lf_image, output_dim=gf_dim, kernel_size=3, stride=1, name='lf_c1'))
            lf_c2 = relu(conv2d(input_=lf_c1, output_dim=gf_dim, kernel_size=3, stride=1, name='lf_c2'))
            lf_concat1 = tf.concat([lf_c1, lf_c2], axis=3, name="lf_Concat1")
            lf_c3 = relu(conv2d(input_=lf_concat1, output_dim=gf_dim, kernel_size=3, stride=1, name='lf_c3'))
            lf_concat2 = tf.concat([lf_c1, lf_c2, lf_c3], axis=3, name="lf_Concat2")
            lf_c4 = relu(conv2d(input_=lf_concat2, output_dim=gf_dim, kernel_size=3, stride=1, name='lf_c4'))
            lf_concat3 = tf.concat([lf_c1, lf_c2, lf_c3, lf_c4], axis=3, name="lf_Concat3")
            lf_c5 = conv2d(input_=lf_concat3, output_dim=3, kernel_size=1, stride=1, name='lf_c5')
            lf_output = lf_c5 * lf_image - lf_c5 + 1

            hf_c1 = lrelu(conv2d(input_=hf_image, output_dim=gf_dim, kernel_size=3, stride=1, name='hf_c1'))
            hf_c2 = lrelu(conv2d(input_=hf_c1, output_dim=gf_dim, kernel_size=3, stride=1, name='hf_c2'))
            hf_c3 = lrelu(conv2d(input_=hf_c2, output_dim=gf_dim, kernel_size=3, stride=1, name='hf_c3'))
            hf_c4 = lrelu(conv2d(input_=hf_c3, output_dim=gf_dim, kernel_size=3, stride=1, name='hf_c4'))
            hf_c5 = lrelu(conv2d(input_=hf_c4, output_dim=gf_dim, kernel_size=3, stride=1, name='hf_c5'))
            hf_c6 = lrelu(conv2d(input_=hf_c5, output_dim=gf_dim, kernel_size=3, stride=1, name='hf_c6'))
            hf_c7 = conv2d(input_=hf_c6, output_dim=3, kernel_size=1, stride=1, name='hf_c7')
            hf_output = hf_c7 + hf_image
            mid_output = hf_output + lf_output

            c1 = lrelu(batch_norm(conv2d(input_=mid_output, output_dim=gf_dim, kernel_size=3, stride=1, name='re_c1'), name='re_c1_bn'))
            c2 = lrelu(batch_norm(conv2d(input_=c1, output_dim=gf_dim, kernel_size=3, stride=1, name='re_c2'), name='re_c2_bn'))
            c3 = lrelu(batch_norm(conv2d(input_=c2, output_dim=gf_dim, kernel_size=3, stride=1, name='re_c3'), name='re_c3_bn'))
            c4 = lrelu(batch_norm(conv2d(input_=c3, output_dim=gf_dim, kernel_size=3, stride=1, name='re_c4'), name='re_c4_bn'))
            c5 = lrelu(batch_norm(conv2d(input_=c4, output_dim=gf_dim, kernel_size=3, stride=1, name='re_c5'), name='re_c5_bn'))
            c6 = lrelu(batch_norm(conv2d(input_=c5, output_dim=gf_dim, kernel_size=3, stride=1, name='re_c6'), name='re_c6_bn'))
            c7 = conv2d(input_=c6, output_dim=3, kernel_size=1, stride=1, name='re_c7')
            output = c7

            return output
