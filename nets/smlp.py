from .utils import *
import sonnet as snt
import tensorflow as tf

class SMLP(snt.Module):

    def __init__(self, l1=30, l2=30, D=3, ISO_ONLY=True):

        super(SMLP, self).__init__()
        self.flatten = snt.Flatten()
        self.net1 = snt.Linear(l1, name="net1")
        self.net2 = snt.Linear(l2, name="net2")
        self.logits = snt.Linear(10, name="logits")
        self._mat = tf.convert_to_tensor(gen_matrix(28,28,D=D, ISO_ONLY=ISO_ONLY), dtype=tf.float32)


    def __call__(self, images, is_training=False):

        output = tf.expand_dims(self.flatten(images), axis=-1) # Make it into a vector of shape (*, 784, 1)
        output = tf.repeat(tf.expand_dims(output, axis=1), repeats=self._mat.shape[0], axis=1) # Repeat to match the number of transformations, resulting in shape (*, number of trans, 784, 1)
        output = tf.matmul(self._mat, output) # Apply the matrices. The matrices will be applied on the last three coordinates by convention

        output = self.flatten(output)
        output = tf.nn.relu(self.net1(output))
        output = tf.nn.relu(self.net2(output))


        output = self.logits(self.flatten(output))

        #return {"logits": output, "softmax": tf.nn.softmax(output)}
        return output # Only return logits!
