#! /usr/bin/env python3

import opencv as cv
import numpy as np
import argparse as ap
import os

def iter_frames(video): 
    ret, frame = video.read()
    while ret:
        yield frame
        ret, frame = video.read()
    

def vidMedian(videoFile):
    if os.path.isfile(videoFile):

        video = cv.VideoCapture(videoFile)
        length = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        height =int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
        myNpArray = np.empty(shape=(height,width,length))
       
        for thisFrIx, frame in enumerate(iter_frames(video)):
            grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            myNpArray[:,:,thisFrIx] = grey

        newImage = np.median(myNpArray, axis=2)
        cv.imwrite("median.jpg", newImage)
        video.release()
        cv.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('videoFile', help='Path to the video to be processed')
    args = parser.parse_args()
    
    vidMedian(args.videoFile)
