import util
import sonnet as snt

class AMLP(snt.Module):

  def __init__(self):
    super(MLP_transformed, self).__init__()
    self.flatten = snt.Flatten()
    self.net1 = snt.Linear(50, name="net1")
    self.net2 = snt.Linear(100, name="net2")
    self.logits = snt.Linear(10, name="logits")
    #self._mat = [tf.convert_to_tensor(m, dtype=float32) for m in generate_matrix(28,28)[0]]
    self._mat = tf.convert_to_tensor(generate_matrix(28,28)[0], dtype=tf.float32)

  def __call__(self, images, is_training=False):
    '''
    output = tf.expand_dims(tf.expand_dims(self.flatten(images), axis=-1), axis=1)
    output = tf.repeat(output, repeats=self._mat.shape[0], axis=1) #repeat the number of transforms
    mat = tf.repeats(tf.expand_dims(self._mat, axis=0), repeats=output.shape[0], axis=0) #repeat the batch size

    output = tf.matmul(mat, output) #(batch, number of trans, 784, 1)
    assert len(output.shape)==4 and output.shape[-1]==1
    output = tf.reshape(output, output.shape[:3])
    '''

    output = tf.expand_dims(self.flatten(images), axis=-1)
    out = None
    for m in range(self._mat.shape[0]):
        mat = tf.repeat(tf.expand_dims(self._mat[m], axis=0), repeats=output.shape[0], axis=0)
        #res = tf.reshape(tf.matmul(mat, output), output.shape[:3])
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
