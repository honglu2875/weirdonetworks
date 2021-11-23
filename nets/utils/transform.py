import cv2
import numpy as np
import random

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def scale_image(image, scale): #scale from 0 to 1, 0->scale to half, 1->original size
    result = cv2.resize(image, ((int)(28*(scale/2+1/2)), (int)(28*(scale/2+1/2))), interpolation=cv2.INTER_NEAREST)
    return result

def rand_trans(image):
    image = np.array(image)
    if image.shape!=(28,28):
        raise Exception("image must be of size (28,28)")

    num = random.randrange(0, 999999999999+1, 1)
    angle = 360*((num%1000)/1000); num = (int)(num/1000)
    scale = (num%1000)/1000; num = (int)(num/1000)
    left = (num%1000)/1000; num = (int)(num/1000)
    up = (num%1000)/1000; num = (int)(num/1000)

    img = scale_image(rotate_image(image, angle), scale) # randomly scale and rotate

    scaled_size = img.shape[0]
    left = (int)((28-scaled_size)/2*left)
    up = (int)((28-scaled_size)/2*up)

    result = cv2.copyMakeBorder(img, up, 28-scaled_size-up, left, 28-scaled_size-left, cv2.BORDER_CONSTANT) # randomly translate
    assert(result.shape==(28,28)) # In case I'm being stupid

    return result

def generate_matrix(width: int, height: int, D=1):
    coord_mat = [] # Matrices for the affine transformations
    mat = [] # Matrices for flattened matrix
    SIZE = max(width, height)

    r = np.arange(1/2, 2, 5/2/D)
    for a in r:
        for b in r:
            for c in r:
                for d in r:
                    for e in r:
                        for f in r:
                            coord_mat.append(np.array([[a,b],[c,d],[e,f]]))
                            m = np.zeros((width*height, width*height))

                            for x in range(width):
                                for y in range(height):
                                    x1=int(a*x+c*y+e)
                                    y1=int(b*x+d*y+f)
                                    if x1>=0 and x1<width and y1>=0 and y1<height:
                                        m[x1+y1*width, x+y*width] = 1

                            mat.append(m)

    return (mat, coord_mat)
