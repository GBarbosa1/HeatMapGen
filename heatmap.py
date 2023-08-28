import numpy as np
import cv2
import copy
import time

def main():
    feed = cv2.VideoCapture('videoplayback3.mp4')
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    num_frames = 350
    controll = 1
    for i in range(0, num_frames):

        if (controll == 1):
            sucess, npFrame = feed.read()
            firstFrame = copy.deepcopy(npFrame)
            gray = cv2.cvtColor(npFrame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            heatmap = np.zeros((height, width), np.uint8)
            controll = 0
        else:
            ret, npFrame = feed.read()  
            gray = cv2.cvtColor(npFrame, cv2.COLOR_BGR2GRAY)  
            fgmask = fgbg.apply(gray)  
            limiter = 30
            motionVal = 4
            ret, treshold = cv2.threshold(fgmask, limiter, motionVal, cv2.THRESH_BINARY)
            heatmap = cv2.add(heatmap, treshold)
            cv2.imshow('heatBuild', heatmap)

        cv2.imshow('frame', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    heatMapImage = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    result_overlay = cv2.addWeighted(firstFrame, 0.7, heatMapImage, 0.7, 0)
    
    cv2.imwrite('out.jpg', result_overlay)
    feed.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    main()
