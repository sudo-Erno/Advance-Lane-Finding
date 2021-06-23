import cv2
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Lane:
    
    # TODO: Implement a method to select which side of the video will be where the lane will be detected (left or right)
    # TODO: Add parameter for the coords in the warp image

    def __init__(self, videoPath, side='left', threshMin = 190, threshMax = 255, nwindows = 9, minpixels = 50, margin = 50): 
        self.cap = cv2.VideoCapture(videoPath)
        self.side = side

        self.threshMin = threshMin
        self.threshMax = threshMax
        self.nwindows = nwindows
        self.minpixels = minpixels
        self.margin = margin

    def convertToGray(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def convertToHLS(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    def applyInRange(self, frame, MinThresh = 190, MaxThresh = 255):
        return cv2.inRange(frame, MinThresh, MaxThresh)
    
    def gaussianBlurFrame(self, frame, kernelSize = 3):
        return cv2.GaussianBlur(frame, (kernelSize, kernelSize), 0)
    
    def warpFrame(self, frame, src, dst):
        img_size = (frame.shape[1], frame.shape[0])

        M = cv2.getPerspectiveTransform(src, dst)

        return cv2.warpPerspective(frame, M, img_size, flags = cv2.INTER_LINEAR)
    
    def slidingWindow(self, img):
        """
        Slides a window through the image and applies the histogram method for line detection.
        The image must be gray and warped.
        """
        # 720 x 1280
        # y --> 720 (0)
        # x --> 1280 (1)

        sizeY, sizeX = img.shape

        outputImg = np.dstack((img, img, img))

        # Compute histogram for the bottom half of the image along the x-axis
        hist = np.sum(img[sizeY//2:,:], axis=0)

        # Height of each window
        window_height = np.int(sizeY // self.nwindows)

        # Check indexes != 0
        nonzeroInx = np.nonzero(img)
        nonzeroInxY = np.array(nonzeroInx[0])
        nonzeroInxX = np.array(nonzeroInx[1])

        # Split the image in two and set the centers
        leftXCenter = np.argmax(hist[:sizeX // 2])
        rightXCenter = np.argmax(hist[sizeX // 2:])  + sizeX // 2

        # Set the x-center of the boxes, which will be corrected over time
        leftXCurrent = leftXCenter
        rightXCurrent = rightXCenter
        
        # Lists to save indexes of pixel inside the rectangle
        leftSidePixels = []
        rightSidePixels = []

        for window in range(self.nwindows):
            # Make the boxes
            # Calculate the Y coords
            yLow = sizeY - (1 + window) * window_height
            yHigh = sizeY - window * window_height
            
            # Calculate the X coords for the left and right side
            xLowLeft = leftXCurrent - self.margin
            xHighLeft = leftXCurrent + self.margin
            xLowRight = rightXCurrent - self.margin
            xHighRight = rightXCurrent + self.margin

            # Draw rectangle for the left lane
            cv2.rectangle(outputImg, (xLowLeft, yLow), (xHighLeft, yHigh), (0, 255, 0), 3)
            
            # Draw rectangle for the right lane
            cv2.rectangle(outputImg, (xLowRight, yLow), (xHighRight, yHigh), (0, 255, 0), 3)

            # Check if pixels's values != 0 are inside the window (rectanle)

            # Check if the indexes are in the boxes and their values != 0
            leftSidePixelsInsideBox = ((nonzeroInxX >= xLowLeft) & (nonzeroInxX <= xHighLeft) & (nonzeroInxY >= yLow) & (nonzeroInxY <= yHigh)).nonzero()[0]
            rightSidePixelsInsideBox = ((nonzeroInxX >= xLowRight) & (nonzeroInxX <=xHighRight) & (nonzeroInxY >= yLow) & (nonzeroInxY <= yHigh)).nonzero()[0]

            leftSidePixels.append(leftSidePixelsInsideBox)
            rightSidePixels.append(rightSidePixelsInsideBox)

            if len(leftSidePixelsInsideBox) > self.minpixels:
                leftXCurrent = np.int(np.mean(nonzeroInxX[leftSidePixelsInsideBox]))

            if len(rightSidePixelsInsideBox) > self.minpixels:
                rightXCurrent = np.int(np.mean(nonzeroInxX[rightSidePixelsInsideBox]))

        try:
            leftSidePixels = np.concatenate(leftSidePixels)
            rightSidePixels = np.concatenate(rightSidePixels)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        leftLaneY = nonzeroInxY[leftSidePixels]
        leftLaneX = nonzeroInxX[leftSidePixels]
        rightLaneY = nonzeroInxY[rightSidePixels]
        rightLaneX = nonzeroInxX[rightSidePixels]

        return leftLaneY, leftLaneX, rightLaneY, rightLaneX, outputImg
    
    def fitPolynomial(self, leftLaneY, leftLaneX, rightLaneY, rightLaneX, warpedImg):
        
        outputImg = np.dstack((warpedImg, warpedImg, warpedImg)) * 255

        # Get the coefficients (A, B, C)
        leftFit = np.polyfit(leftLaneX, leftLaneY, 2)
        rightFit = np.polyfit(rightLaneX, rightLaneY, 2)

        # Generate x values. These will be the y for plotting
        ploty = np.linspace(0, outputImg.shape[0]-1, outputImg.shape[0])

        try:
            leftFitX = ploty*leftFit[0]**2 + ploty*leftFit[1] + leftFit[2]
            rightFitX = ploty*rightFit[0]**2 + ploty*rightFit[1] + leftFit[2]
        
        except TypeError:
            # In case there is no C
            leftFitX = ploty*leftFit[0]**2 + ploty*leftFit[1]
            rightFitX = ploty*rightFit[0]**2 + ploty*rightFit[1]

        # VISUALIZATION #

        # outputImg[leftLaneY, leftLaneX] = [255, 0, 0]
        # outputImg[rightLaneY, rightLaneX] = [0, 0, 255]

        return outputImg

    def checkStop(self):
        
        k = cv2.waitKey(30) & 0xff
        
        if k == 27:
            return False
            
        return True

    def detectLane(self):
        
        while self.checkStop():

            _, frame = self.cap.read()

            hls = self.convertToHLS(frame)

            sChannel = hls[:,:,2]

            blurImage = self.gaussianBlurFrame(sChannel, 5)

            mask = self.applyInRange(blurImage, self.threshMin, self.threshMax)

            hlsResult = cv2.bitwise_and(frame, frame, mask = mask)

            # Perspective Transform
            sizeY, sizeX = (hlsResult.shape[0], hlsResult.shape[1])

            src = np.float32([[200, sizeY], [575, 450], [775, 450], [1100, sizeY]])
            dst = np.float32([[200, sizeY], [200, 0], [750, 0], [750, sizeY]])

            warp = self.warpFrame(hlsResult, src, dst)

            grayWarp = self.convertToGray(warp)

            leftLaneY, leftLaneX, rightLaneY, rightLaneX, warpedImg = self.slidingWindow(grayWarp)

            res = self.fitPolynomial(leftLaneY, leftLaneX, rightLaneY, rightLaneX, warpedImg)

            cv2.imshow('Frame', res)
        
        cv2.destroyAllWindows()
        self.cap.release()

left_lane = Lane(videoPath = './videos/project_video.mp4', side = 'left', threshMin = 170, threshMax = 255, nwindows = 15)
left_lane.detectLane()