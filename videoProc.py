#! /usr/bin/env python3

import cv2 as cv
import numpy as np
import argparse as ap
import os


def vidMedian(videoFile):
    if os.path.isfile(videoFile):

        cap = cv.VideoCapture(videoFile)
        length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height =int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        myNpArray = np.empty(shape=(height,width,length))
        thisFrIx = 0 
       
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                myNpArray[:,:,thisFrIx] = grey
                thisFrIx += 1
            else:
                break    

        newImage = np.median(myNpArray, axis=2)
        cv.imwrite("median.jpg", newImage)
        cap.release()
        cv.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('videoFile', help='Path to the video to be processed')
    args = parser.parse_args()
    
    vidMedian(args.videoFile)
