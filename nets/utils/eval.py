import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def show_pic(img, NROWS=4, NCOLS=4):
    # img: an array of images of shape (*, *, 3)

    fig = plt.gcf()
    fig.set_size_inches(NCOLS*4, NROWS*4)

    assert len(np.array(img).shape) == 4

    if np.array(img).shape[3] == 1:
        GREYSCALE = True
    else:
        GREYSCALE = False

    for i, im in enumerate(img):
        if i>=NROWS*NCOLS: break
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(NROWS, NCOLS, i + 1)
        sp.axis('Off') # Don't show axes (or gridlines)

        if GREYSCALE:
            plt.imshow(np.reshape(im, np.array(im).shape[:2]), cmap='gray')
        else:
            plt.imshow(im)


    plt.show()

def ev_rotation_stability(model, test, MAX_IMG=100, D=10, NEED_RESCALING=False):
    # model: the model to be evaluated
    # test: a generator consisting of tuples (x, y) where x is an array of data and y is the ground truth.
    # MAX_IMG: (optional) the maximal number of images to be evaluated. Existed for performance reason.
    # D: (optional) number of samples on rotations

    # returns the variance, the difference, the whole prediction table, the target value

    pred = []
    batch_size = len(test[0][0])

    # Store the first MAX_IMG pictures in the generator
    data = []
    y_target = []
    img_count = 0
    for x, y in test:
        for i in range(len(x)):
            sc = 255. if NEED_RESCALING else 1.
            data.append(x[i]*sc) # evenly sample rotations
            y_target.append(y[i])

            img_count += 1
            if img_count >= MAX_IMG: break
        if img_count >= MAX_IMG: break

    # Apply transform and evaluate
    for i in range(D):
        d = []
        for im in data:
            d.append(ImageDataGenerator().apply_transform(im, {'theta':i/D*360}))

        pred.append( np.reshape(model(tf.convert_to_tensor(d)), -1) )

    pred = np.array(pred)
    var = np.var(pred, axis=0)
    dif = np.average(abs(pred[:-2,:] - pred[1:-1,:]), axis=0)

    return var, dif, pred, y_target
