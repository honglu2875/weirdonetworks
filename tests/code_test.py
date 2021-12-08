import ast
import numpy as np
import json
import cv2
from nets.utils import *
import tensorflow as tf
import time

def main():
    grads = (tf.convert_to_tensor([1.,2.,3.]), tf.convert_to_tensor([1.,1.,1.,1.]))
    rgrads = (tf.convert_to_tensor([3.,4.,5.]), tf.convert_to_tensor([1.,0.,1.,0.]))
    projected_gradient = []
    for g, rg in zip(grads, rgrads):
        d = tf.math.reduce_sum(tf.math.multiply(rg,g)) / tf.math.reduce_sum(tf.math.multiply(g,g))
        projected_gradient.append(rg-tf.math.multiply(d, g))
        print(tf.math.multiply(projected_gradient[-1],g))

    print(projected_gradient)

    flattened_g = tf.concat( [tf.reshape(x, [-1]) for x in grads], axis=0 )
    flattened_rg = tf.concat( [tf.reshape(x, [-1]) for x in rgrads], axis=0 )
    d = tf.math.reduce_sum(tf.math.multiply(flattened_rg,flattened_g)) / tf.math.reduce_sum(tf.math.multiply(flattened_g,flattened_g))
    projected_gradient = tuple([rgrads[i]-tf.math.multiply(d,grads[i]) for i in range(len(rgrads))])

    print(projected_gradient)

if __name__ == '__main__':
    main()
