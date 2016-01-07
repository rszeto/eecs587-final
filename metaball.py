# An example implementation of the algorithm described at
# http://www.niksula.cs.hut.fi/~hkankaan/Homepages/metaballs.html
#
# The code contains some references to the document above, in form
# ### Formula (x)
# to make clear where each of the formulas is implemented (and what
# it looks like in Python)
#
# Since Python doesn't have an in-built vector type, I used complex
# numbers for coordinates (x is the real part, y is the imaginary part)
#
# Made by Hannu Kankaanp. Use for whatever you wish.

from __future__ import division

import math
import numpy as np
import cv2
import sys
import os

class Ball:
    """Single metaball."""
    def __init__(self, pos, size):
        self.pos = pos
        self.size = size

def main(argv):
    # Get directory name from args, or set default if not specified
    saveDir = ''
    if len(argv) > 1:
        saveDir = argv[1]
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
    # Seed rng
    np.random.seed(123)
    
    imSize = [150, 200]
    
    threshold = 0.0004
    goo = 2.2
    r = .018
    numBalls = 40
    numImages = 5
    visualize = False

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    for loopVar in range(numImages):
        if visualize:
            print 'Drawing...'
        # Init image to display
        image = np.zeros((imSize[0], imSize[1], 3), np.uint8)
        # Create random balls
        balls = []
        for i in range(numBalls):
            row = np.random.randint(0, imSize[0])
            col = np.random.randint(0, imSize[1])
            balls.append(Ball((row,col), size=r))
        # Draw the balls
        simpleDrawBalls(image, balls, goo, threshold)
        fileName = 'bahmetaballs%02d.png' % loopVar
        cv2.imwrite(os.path.join(saveDir, fileName), image)
        if visualize:
            # Draw centers
            for i in range(len(balls)):
                ball = balls[i]
                cv2.circle(image, (ball.pos[1], ball.pos[0]), 2, (0,255,0), -1)
            cv2.imshow('image', image)
            print 'Press key to continue'
            k = cv2.waitKey(0)
            if k==27 or k==-1:
                cv2.destroyAllWindows()
                return

def metaballTerm(ball, point, goo):
    denom = pow(np.linalg.norm(np.array(ball.pos)-np.array(point)), goo)
    if denom == 0:
        return 10000
    return ball.size/denom

def simpleDrawBalls(image, balls, goo, threshold):
    imSize = image.shape
    for row in range(imSize[0]):
        for col in range(imSize[1]):
            sum = 0
            for i in range(len(balls)):
                ball = balls[i]
                sum += metaballTerm(ball, (row, col), goo)
                if sum > threshold:
                    image[row, col] = [255, 255, 255]
                    break

if __name__ == '__main__':
    main(sys.argv)
