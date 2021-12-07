import cv2
import numpy as np
from random import random, randrange
import math

def check_image(image):
    # Check some critical properties of the image input
    assert len(image.shape)==2



def rotate_image(image, angle: float, BORDER_VAL=0):
    # image: np.array of dimension 2
    # angle: float, indicating the angle

    check_image(image)

    ox, oy = tuple((np.array(image.shape) - 1) / 2)
    sizex, sizey = image.shape

    res = np.zeros(image.shape) + BORDER_VAL # even faster than creating an array and then set values
    angle_r = angle/360*2*math.pi

    cos = math.cos(-angle_r)
    sin = math.sin(-angle_r)
    for x in range(sizex):
        for y in range(sizey):
            qx = math.floor(ox + cos * (x - ox) - sin * (y - oy))
            qy = math.floor(oy + sin * (x - ox) + cos * (y - oy))
            if qx>=0 and qx<sizex and qy>=0 and qy<sizey:
                res[x][y] = image[qx][qy]
    # OpenCV only supports uint8 which is really stupid.... Have to write my own but the end result is slightly different. The interpolation mode issue???
    #M = cv2.getRotationMatrix2D((oy,ox), angle, 1.0)
    #res = cv2.warpAffine(image.astype(np.uint8), M, image.shape[1::-1], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=BORDER_VAL)
    return res

def scale_image(image, scale: float, ORIG_SIZE=True, BORDER_VAL=0):
    # image: np.array of dimension 2
    # scale: float, positive

    check_image(image)
    assert scale>0

    def p(size):
        return max(math.floor(size*scale),1)

    sizex, sizey = image.shape
    if ORIG_SIZE:
        shp = (sizex, sizey)
    else:
        shp = (p(sizex), p(sizey))
    res = np.zeros(shp) + BORDER_VAL
    for x in range(res.shape[0]):
        for y in range(res.shape[1]):
            qx = math.floor(x/scale)
            qy = math.floor(y/scale)
            if qx>=0 and qx<sizex and qy>=0 and qy<sizey:
                res[x][y] = image[qx][qy]
    #result = cv2.resize(image.astype(np.uint8), (p(sizex), p(sizey)), interpolation=cv2.INTER_NEAREST)
    #result = cv2.copyMakeBorder(result, 0, sizey-y, 0, sizex-x, borderType=cv2.BORDER_CONSTANT, value=BORDER_VAL) # x and y are flipped in cv2
    return res

def sheer_image(image, sheer: float, BORDER_VAL=0):
    # image: np.array of dimension 2
    # sheer: float, (recommended between -1 and 1, for image clarity)

    check_image(image)

    sizex, sizey = image.shape
    res = np.zeros(image.shape) + BORDER_VAL
    for x in range(sizex):
        for y in range(sizey):
            qx = math.floor(x-y*sheer)
            qy = math.floor(y)
            if qx>=0 and qx<sizex and qy>=0 and qy<sizey:
                res[x][y] = image[qx][qy]

    #M = np.array( [[1,sheer,0],[0,1,0]] ).astype(np.float32)
    #result = cv2.warpAffine(image.astype(np.uint8), M, image.shape[1::-1], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=BORDER_VAL)
    return res

def translate_image(image, left: float, top: float, BORDER_VAL=0):
    # Translate an image based on the porportion from the left and the top
    # image: np.array of dimension 2
    # left: float, between -1 and 1, indicating the proportion to translate off the left border
    # top: float, between -1 and 1, indicating the proportion to translate off the upside border

    check_image(image)
    assert left>-1 and left<=1 and top>-1 and top<=1

    sizex, sizey = image.shape

    left_p = int(sizex*left)
    top_p = int(sizey*top)

    res = np.zeros(image.shape) + BORDER_VAL
    for x in range(sizex):
        for y in range(sizey):
            qx = int(x-top_p)
            qy = int(y-left_p)
            if qx>=0 and qx<sizex and qy>=0 and qy<sizey:
                res[x][y] = image[qx][qy]
    #M = np.array( [[1,0,left_p], [0,1,top_p]] ).astype(np.float32)
    #result = cv2.warpAffine(image.astype(np.uint8), M, image.shape[1::-1], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=BORDER_VAL)
    return res

def trans(image, angle: float, scale: float, left: float, top: float, sheer: float=.0, BORDER_VAL=0):
    # The full image transformation function. Combine all the above IN A SPECIFIC ORDER.
    # image: np.array of dimension 2
    # angle: float, in degrees
    # scale: float, between 0 and 1
    # left: float, between 0 and 1, indicating the proportion to translate off the left border
    # top: float, between 0 and 1, indicating the proportion to translate off the upside border
    # sheer: float, (recommended between -1 and 1, for image clarity)
    # returns the resulting picture and a pixel map (a same-sized matrix whose value indicates which pixel it comes from)

    result = translate_image(
        scale_image(
        sheer_image(
        rotate_image(
        image, angle, BORDER_VAL=BORDER_VAL),
        sheer, BORDER_VAL=BORDER_VAL),
        scale, BORDER_VAL=BORDER_VAL),
        left, top, BORDER_VAL=BORDER_VAL)

    return result

def random_trans(image, BORDER_VAL=0, ISO_ONLY=True):
    # Randomly transform using trans(...)
    # image: np.array of dimension 2

    angle = random()*360
    if ISO_ONLY:
        scale, sheer, left, top = 1, 1, 0, 0
    else:
        scale = random()*0.5+0.5
        sheer = random()*0.5
        left, top = random()*0.5-0.25, random()*0.5-0.25

    return trans(image, angle, scale, left, top, sheer=sheer)

def gen_orbit(image, D: int, BORDER_VAL=-1, ISO_ONLY=True):
    # Sample a bunch of transformations and generate matrices corresponding to the transformations between flattened coordinates of images.
    # image: np.array of dimension 2
    # D: int, a constant indicating how many angles, scales, etc. will be sampled for each parameter

    check_image(image)

    x, y = image.shape

    angles = np.arange(0, 360, 360 / D)
    if ISO_ONLY:
        scales = [1]
        lefts = [0]
        tops = [0]
        sheers = [0]
    else:
        scales = np.arange(1., 0.5, -0.5 / D)
        lefts = np.arange(0., 0.5, 0.5 / D)
        tops = np.arange(0., 0.5, 0.5 / D)
        sheers = np.arange(0., 1., 1. / D)


    index = [(a,sc,sh,l,t) for a in angles for sc in scales for sh in sheers for l in lefts for t in tops]

    results = []
    for angle, scale, sheer, left, top in index:
        results.append(trans(image, angle, scale, left, top, sheer=sheer, BORDER_VAL=BORDER_VAL))
    return results

def gen_matrix(x: int, y:int, D: int=4, ISO_ONLY=True):
    # A transformation on an image can be realized as a (permutation) linear transformation (flatten the image and give the pixels coordinates).
    # This function generates such matrices.
    # x,y: integers, indicating the size of the image
    # D: integer, same as gen_orbit

    M_init = np.arange(x*y).reshape((x,y))
    M = gen_orbit(M_init, D, BORDER_VAL=-1, ISO_ONLY=ISO_ONLY)

    res = []
    for m in M:
        T = np.zeros((x*y, x*y))
        for i in range(x*y):
            e = m.astype(int)[i//y][i%y]
            if e!=-1:
                T[e][i] = 1
        res.append(T)

    return res



########## OBSOLETE ###########
def enc(n):
    # Encode a number (<2^24) into 3 uint8
    # return a tuple of 3 np.uint8
    return np.uint8(n-(n>>8<<8)), np.uint8((n>>8)-(n<<16)), np.uint8(n<<16)

def dec(t):
    # Combine a tuple of 3 np.uint8 into a long integer
    assert len(t)==3

    return int(t[0]+(t[1]<<8)+(t[2]<<16))

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
