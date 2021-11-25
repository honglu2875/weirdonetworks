import ast
import numpy as np
import json
import cv2
from nets.utils import *

def main():
    with open("tests/test_img", "r") as f:
        pics = list(np.array( ast.literal_eval(json.load(f)) ))
    for pic in pics[:1]:
        print(pic.shape)
        #p = translate_image(pic, 0.5, 0)
        #p = sheer_image(pic, 1)
        #p = scale_image(pic, 0.5)
        #p = rotate_image(pic, 80)
        p = trans(pic, 50, 0.8, 0.2, -0.2, 0.2)
        cv2.imshow("image", p)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
