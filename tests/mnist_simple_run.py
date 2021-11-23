import tensorflow as tf
import sonnet as snt
import tensorflow_datasets as tfds # tensorflow datasets
from tqdm import tqdm
import random

from nets import MLP
from nets.utils import rand_trans, step


def process_batch(images, labels):
    images = tf.squeeze(images, axis=[-1])
    images = tf.cast(images, dtype=tf.float32)
    images /= 255.
    images = tf.clip_by_value(images, 0., 1.)
    return images, labels

def progress_bar(generator, size, batch_size=1):
  return tqdm(
      generator,
      unit='images',
      unit_scale=batch_size,
      total=size)

def mnist(split, batch_size=100): #batch is deactivated to fit the unpack and transformation
    dataset, ds_info = tfds.load('mnist:3.*.*', split=split, as_supervised=True,
                               with_info=True)
    dataset = dataset.map(process_batch)
    #dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()
    return dataset, ds_info

def unpack_dataset(dataset):
    x = []
    y = []
    for elem in dataset.as_numpy_iterator():
        assert len(elem) == 2
        x.append(elem[0])
        y.append(elem[1])

    return (x, y)





def main():

    print("TensorFlow version {}".format(tf.__version__))
    print("Sonnet version {}".format(snt.__version__))


    mnist_train, mnist_train_info = mnist('train')
    mnist_test, mnist_test_info = mnist('test')

    x_train, y_train = unpack_dataset(mnist_train)
    x_train_trans = []
    y_train_trans = []
    NUM = 10000
    for i in range(NUM):
        ind = random.randrange(0,len(x_train),1)
        x_train_trans.append(rand_trans(x_train[ind]))
        y_train_trans.append(y_train[ind])

    opt = snt.optimizers.Adam(learning_rate=0.001)

    num_epochs = 1
    #model = CNN(use_batch_norm=False)
    model = MLP()
    #mlp = MLP()
    #model = quadMLP()

    BATCH_SIZE = 100
    num_images = len(x_train_trans)
    #num_images = 60000



    grad_norm = []
    loss_all = []
    tr = [(x_train_trans[i*BATCH_SIZE:(i+1)*BATCH_SIZE],y_train_trans[i*BATCH_SIZE:(i+1)*BATCH_SIZE]) for i in range(int(num_images/BATCH_SIZE))]


    #for images, labels in progress_bar(mnist_train.repeat(num_epochs)):
    for images, labels in progress_bar(tr * num_epochs, (num_images // BATCH_SIZE) * num_epochs, batch_size=BATCH_SIZE):
        loss, n = step(tf.reshape(tf.convert_to_tensor(images), [-1,28,28,1]), tf.convert_to_tensor(labels), model, opt)
        grad_norm.append(n)
        loss_all.append(loss)

    print("\n\nFinal loss: {}".format(loss.numpy()))

    #mnist_shuffled = mnist_train.shuffle(10000).repeat()





if __name__ == "__main__":
    main()
