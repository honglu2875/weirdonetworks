import ast
import numpy as np
import json
import cv2
from nets.utils import *

def main():
    x, y = 3, 3
    M = gen_matrix(x,y,D=2)
    print(M)

    '''
    with open("tests/test_img", "r") as f:
        pics = list(np.array( ast.literal_eval(json.load(f)) ))

    N = gen_orbit(pics[0], 2, BORDER_VAL=128)
    for p in N:
        cv2.imshow("image", p.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    '''
if __name__ == '__main__':
    main()
