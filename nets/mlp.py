import sonnet as snt

class MLP(snt.Module):

  def __init__(self):
    super(MLP, self).__init__()
    self.flatten = snt.Flatten()
    self.hidden1 = snt.Linear(100, name="hidden1")
    self.hidden2 = snt.Linear(100, name="hidden2")
    self.logits = snt.Linear(10, name="logits")

  def __call__(self, images, is_training=False):
    output = self.flatten(images)
    output = tf.nn.gelu(self.hidden1(output))
    output = tf.nn.gelu(self.hidden2(output))
    output = self.logits(output)
    return {"logits": output, "softmax": tf.nn.softmax(output)}

mlp = MLP()
