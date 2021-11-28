import tensorflow as tf
import sonnet as snt

class quadMLP(snt.Module):
  def __init__(self):
    super(quadMLP_big, self).__init__()
    self.flatten = snt.Flatten()
    self.hidden1 = snt.Linear(100, name="hidden1")
    self.logits = snt.Linear(10, name="logits")

  def __call__(self, images, is_training=False):
    if len(images.shape) == 3:
        small = tf.expand_dims(images, axis=-1)
    elif len(images.shape) == 4:
        small = images
    else:
        raise Exception("Unrecognized input shape {}".format(str(images.shape)))

    output = self.flatten(images)
    output = tf.reshape(output, [-1,n,1]) * tf.reshape(output, [-1,1,n]) # turn into quadratic function


    output = tf.nn.relu(self.hidden1(output))
    output = self.logits(output)

    #return {"logits": output, "softmax": tf.nn.softmax(output)}
    return output # Only return logits!
