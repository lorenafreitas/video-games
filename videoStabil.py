#!/usr/bin/env Python3

import numpy as np
import cv2 as cv

def iter_frames(video):
    ret, frame = video.read()
    while ret:
        yield frame
        ret, frame = video.read()


# moving average filter 
def movingAverage(curve, radius):
    window_size = 2*radius+1
    #define the filter
    f = np.ones(window_size)/window_size
    # add padding to the boundaries
    curve_pad = np.lib.pad(cuve, (radius, radius),'edge')
    #apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    #remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # Return smoothed curve
    return curve_smoothed

def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3)
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius = SMOOTHING_RADIUS)
    return smoothed_trajectory


def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
    frame = cv.warpAffine(frame, T, (s[1], s[0]))
    return frame


def stabilise():
    video = cv.VideoCaptude('coverte')
    n_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_froucc(*'MJPG')
    out = cv.VideoWriter('out_video.mp4', fourcc, fps, (width, height))

    _, prev = video.read()
    prev_grey = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

    # Pre-define transformation-store array
    transforms = np.zero((n_frames-1,3), np.float32)

    for thisFrIx,frame in enumerate(iter_frames(video))

        # Detect feature  points in previous frame
        prev_pts = cv.goodFeaturesToTrack(prev_grey, maxCorners = 200, qualityLevel=0. minDistance=30, blockSize=3)

        #Convert current frame to greyscale
        curr_grey = cv.cvtColor(curr, cv.COLOR_BGR2GRAY) 

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv.calcOpticalFlowPyrLK(prev_grey, curr_grey, prev_pts, none)

        # Sanity check
        assert prev_pts.shape == curr_pts.shape

        # Filter only valid points
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # Find transformation matrix
        m = cv.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False)

        # Extract translation
        dx = m[0,2]
        dy = m[1,2]

        # Extract rotation angle
        da = np.arctan2(m[1,0], m[0,0])

        # store transformations
        transforms[thisFrIx] = [dx, dy, da]

        # Move to the next frame
        prev_grey = curr_grey

        print("Frame " + str(thisFrIx) + "/" + str(n_Frames) + " - Tracke points: " + str(len(prev_pts)))

    # Compute trajectory (sum of motion between frames) using the cumulative um of transformations
    trajectory = np.cumsum(transforms, axis=0)


    # Smooth trajectory
    smoothed_trajectory = smooth(trajectory)

    # Obtain smooth transforms that can be applied to frames of the videos to stabilise it, by  calculating the difference between the trajectorz and its smoothed version and adding it back to the  transforms
    #Caculate difference in smoothed trajectory and trajectory 
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + diference

    # Reset stream to first frame
    video.set(cv.CAP_PROP_POS_FRAMES,0)

    # Write n_frames-1 transformed frames
    for thisFrIx, frame in enumerate(iter_frames(video)):
        # Extract transformations from the new transformation array
        dx = transforms_smooth[thisFrIx,0]
        dy = transforms_smooth[thisFrIx,1]
        da = transforms_smooth[thisfrIx,2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2,3), np.float32)
        m[0,0] = np.cos(da)
        m[0,1] = np.sin(da)
        m[1,0] = np,sin(da)
        m[1,1] = np.cos(da)
        m[0,2] = dx
        m[1,2] = dy

        # Apply affine wrapping to current frame
        frame_stabilised = cv.warpAffine(frame, m, (w,h))

        # Fix border artifacts
        frame_stabilised = fixBorder(frame_stabilised)

        # Write the stabilised frame to the file
        frame_out = cv.hconcnt([frame, frame_stabilised])
        
        # If the image is too bg, resize it
        if(frame_out.shape[1] >= 1920)
            frame_out = cv.resize(frame_out, (frame_out.shape[1]/2, frame_out.shape[0]/2))

        cv.imshow("Before and After", frame_out)
        cv.waitKey(10)
        out.write(frame_out)


    

if __name__ == "__main__"
    stabilise() 
