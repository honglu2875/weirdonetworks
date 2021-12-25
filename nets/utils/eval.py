import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def ev_rotation_stability(model, test, MAX_IMG=100, D=10):
    # model: the model to be evaluated
    # test: tuple (x, y). x is an array of data and y is the ground truth (True/False)
    # MAX_IMG: (optional) the maximal number of images to be evaluated (False: unrestricted). Existed for performance reason.
    # D: (optional) number of samples on rotations



    for i in min(len(test[0]), MAX_IMG):
        img = test[i]
        data = []
        for j in range(D):
            #data.append(rotate(s[0][0], i/D*360))
            data.append(ImageDataGenerator().apply_transform(img, {'theta':j/D*360}))

        pred=model(tf.convert_to_tensor(data))

        print(np.array(pred))
        print(np.var(pred))
