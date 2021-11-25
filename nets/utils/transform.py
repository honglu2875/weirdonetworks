import cv2
import numpy as np
from random import random, randrange

def check_image(image):
    # Check some critical properties of the image input
    assert len(image.shape)==2


def rotate_image(image, angle: float):
    # image: np.array of dimension 2
    # angle: float, indicating the angle

    check_image(image)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    M = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image.astype(np.uint8), M, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def scale_image(image, scale: float, ORIG_SIZE=True):
    # image: np.array of dimension 2
    # scale: float, from 0 to 1

    check_image(image)
    assert scale>0 and scale<=1

    def p(size):
        return max((int)(size*scale),1)

    sizex, sizey = image.shape
    result = cv2.resize(image.astype(np.uint8), (p(sizex), p(sizey)), interpolation=cv2.INTER_NEAREST)
    if ORIG_SIZE:
        result = cv2.copyMakeBorder(result, 0, sizey, 0, sizex, cv2.BORDER_CONSTANT) # x and y are flipped in cv2
    return result

def sheer_image(image, sheer: float):
    # image: np.array of dimension 2
    # sheer: float, (recommended between -1 and 1, for image clarity)

    check_image(image)

    M = np.array( [[1,sheer,0],[0,1,0]] ).astype(np.float32)
    result = cv2.warpAffine(image.astype(np.uint8), M, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def translate_image(image, left: float, top: float):
    # Translate an image based on the porportion from the left and the top
    # image: np.array of dimension 2
    # left: float, between -1 and 1, indicating the proportion to translate off the left border
    # top: float, between -1 and 1, indicating the proportion to translate off the upside border

    check_image(image)
    assert left>-1 and left<=1 and top>-1 and top<=1

    sizex, sizey = image.shape

    left_p = int(sizex*left)
    top_p = int(sizey*top)
    M = np.array( [[1,0,left_p], [0,1,top_p]] ).astype(np.float32)
    result = cv2.warpAffine(image.astype(np.uint8), M, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def trans(image, angle: float, scale: float, left: float, top: float, sheer: float=.0):
    # The full image transformation function. Combine all the above IN A SPECIFIC ORDER.
    # image: np.array of dimension 2
    # angle: float
    # scale: float, between 0 and 1
    # left: float, between 0 and 1, indicating the proportion to translate off the left border
    # top: float, between 0 and 1, indicating the proportion to translate off the upside border
    # sheer: float, (recommended between -1 and 1, for image clarity)

    result = translate_image(
        scale_image(
        sheer_image(
        rotate_image(
        image.astype(np.uint8), angle),
        sheer),
        scale),
        left, top)
    return result


#### might remove:
def rand_trans(image):
    # Randomly perform a scaling and a rotation
    # image: np.array of dimension 2, fixed size of (28,28)

    check_image(image)

    sizex, sizey = image.shape

    # Generate random numbers
    angle, scale, left, up, sheer = randrange(0,360), random()*0.5+0.5, randrange(), random(), random()*0.5

    img = scale_image(rotate_image(image, angle), scale) # randomly scale and rotate

    scaled_size = img.shape[0]
    left = (int)((28-scaled_size)/2*left)
    up = (int)((28-scaled_size)/2*up)

    result = cv2.copyMakeBorder(img, up, 28-scaled_size-up, left, 28-scaled_size-left, cv2.BORDER_CONSTANT) # randomly translate
    assert(result.shape==(28,28)) # In case I'm being stupid

    return result




def rand_affine(image):
    # Randomly perform an affine transformation
    # (remark: often the image gets crushed or goes out of the bound...)
    # image: np.array of dimension 2

    check_image(image)

    def p(i):
        return i*2-1

    sizex, sizey = image.shape
    a, b, c, d = p(random()), p(random()), p(random()), p(random())
    M = np.float32([[a, b, randrange(-sizex//2, sizex//2)], [c, d, randrange(-sizey//2, sizey//2)]])

    return cv2.warpAffine(image.astype(np.uint8), M, image.shape[1::-1], flags=cv2.INTER_LINEAR), M

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
