from .utils import *
import sonnet as snt
import tensorflow as tf

class SMLP(snt.Module):

    def __init__(self, l1=5, l2=10):

        super(SMLP, self).__init__()
        self.flatten = snt.Flatten()
        self.net1 = snt.Linear(l1, name="net1")
        self.net2 = snt.Linear(l2, name="net2")
        self.logits = snt.Linear(10, name="logits")
        self._mat = tf.convert_to_tensor(gen_matrix(28,28,D=3), dtype=tf.float32)

    def __call__(self, images, is_training=False):

        output = tf.expand_dims(self.flatten(images), axis=-1)
        out = None
        for m in range(self._mat.shape[0]):
            mat = tf.repeat(tf.expand_dims(self._mat[m], axis=0), repeats=output.shape[0], axis=0)
            res = self.flatten(tf.matmul(mat, output))

            res = tf.nn.relu(self.net1(res))
            res = tf.nn.relu(self.net2(res))

            if out is None:
                out = tf.expand_dims(res, axis=0)
            else:
                out = tf.concat([out, tf.expand_dims(res, axis=0)], axis=0)


        output = tf.transpose(out, perm=[1,0,2])
        output = self.logits(self.flatten(output))

        return {"logits": output, "softmax": tf.nn.softmax(output)}
