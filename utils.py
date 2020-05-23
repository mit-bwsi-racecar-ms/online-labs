import cv2
import numpy as np
from matplotlib import pyplot as plt

#############################
#### Parameters
#############################

video_port = 0

#############################
#### Show Image
#############################

def show_inline(img):
    '''Displays an image inline.'''
    if (len(img.shape) == 3):
        b, g, r = cv2.split(img)
        rgb_img = cv2.merge([r,g,b])
        plt.imshow(rgb_img)
        plt.xticks([]), plt.yticks([])
        plt.show()
    else:
        arr = np.asarray(img)
        plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
        plt.xticks([]), plt.yticks([])
        plt.show()
    
def close_windows():
    '''Close popup window via 'ESC' key.'''
    while cv2.waitKey(200) & 0xFF != 27:
        pass
    cv2.destroyAllWindows()
    cv2.waitKey(1)

#############################
#### Show Video
#############################
    
def video(function):
    '''Displays video with modifiable frames.'''
    video = cv2.VideoCapture(video_port)
    while cv2.waitKey(200) & 0xFF != 27:
        frame = video.read()[1]
        if frame is not None:
            #frame = cv2.resize(frame, (640,480))  # Uncomment this line if issues arise
            frame = cv2.resize(frame, (640,480))
            function(frame)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
def shrink(img, scale):
    '''Returns a shrinked image by the scale specified.'''
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

#############################
#### Display Coordinates
#############################
    
def coordinates():
    '''Displays coordinates of the mouse as it moves.'''
    window = 'Coordinates'
    def show_coords(x, y):
        img = np.zeros((500, 500, 3), np.uint8)
        cv2.putText(img, 'x: ' + str(x), (20, 480), 0, 0.75, (255, 255, 255), 2)
        cv2.putText(img, 'y: ' + str(y), (120, 480), 0, 0.75, (255, 255, 255), 2)
        cv2.putText(img, 'Point: ' + str((x, y)), (220, 480), 0, 0.75, (255, 255, 255), 2)
        cv2.imshow(window, img)
    def callback(event, x, y, flags, params):
        if event == cv2.EVENT_MOUSEMOVE:
            show_coords(x, y)
    show_coords(0, 0)
    cv2.setMouseCallback(window, callback)
    while cv2.waitKey(200) & 0xFF != 27:
        pass
    cv2.destroyAllWindows()
    cv2.waitKey(1)

#############################
#### Contours
#############################

def findEdgedImage(image):
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    
def findGreatestContour(contours):
    cnt = [-1, -1]
    for i in range(0, len(contours)):
        if (cv2.contourArea(contours[i]) >= cnt[1]):
            cnt  = [i, cv2.contourArea(contours[i])]
    return contours[cnt[0]]

def detectDrawings(img, color_range):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, color_range[0], color_range[1])
    contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if contours is []:
        return img
    cnt = findGreatestContour(contours)
    
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    '''
    for contour in contours:
        pt, r = cv2.minEnclosingCircle(cnt)
    
    box = cv2.boxPoints(cv2.minAreaRect(cnt))
    box = np.int0(box)
    if abs(cv2.contourArea(box) - cv2.contourArea(cnt)) > 10000:
        print(cv2.contourArea(box), cv2.contourArea(cnt))
    else:
        cv2.drawContours(img,[box],0,(0,0,255),2)
    '''
    return img, mask 


#############################
#### Painter
#############################

def find_center(contour):
    '''Returns the center coordinates of the smallest circle around contour'''
    x, y = cv2.minEnclosingCircle(contour)[0]
    return (int(x), int(y))


def find_radius(contour):
    '''Returns the radius of the smallest circle around contour'''
    radius = cv2.minEnclosingCircle(contour)[1]
    return int(radius)
  
#############################
#### HSV
#############################

def hsv_select(filename):
    '''Thresholds image via HSV Trackbar values.'''
    img = cv2.imread(filename)
    h, w, ch = img.shape
    if w > 700:
        img = cv2.resize(img, (700, int(h*(700.0/w))))  # resize to width of 700
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    window_name = 'HSV Select'
    cv2.imshow(window_name, img) 
    hsv_min = np.array([0, 0, 0])
    hsv_max = np.array([179, 255, 255])  
    def callback(value):
        hsv_min[0] = cv2.getTrackbarPos('H_min', window_name)
        hsv_max[0] = cv2.getTrackbarPos('H_max', window_name)
        hsv_min[1] = cv2.getTrackbarPos('S_min', window_name)
        hsv_max[1] = cv2.getTrackbarPos('S_max', window_name)
        hsv_min[2] = cv2.getTrackbarPos('V_min', window_name)
        hsv_max[2] = cv2.getTrackbarPos('V_max', window_name)
        mask = cv2.inRange(img_hsv, hsv_min, hsv_max)
        img_masked = cv2.bitwise_and(img, img, mask=mask)
        cv2.putText(img_masked, 'HSV Lower: {}'.format(tuple(hsv_min)), (10, 35), 0, 0.75, (255, 255, 255), 2)
        cv2.putText(img_masked, 'HSV Upper: {}'.format(tuple(hsv_max)), (10, 70), 0, 0.75, (255, 255, 255), 2)
        cv2.imshow(window_name, img_masked)    
    # make the trackbar used for HSV masking 
    cv2.createTrackbar('H_min', window_name, 0, 179, callback) 
    cv2.createTrackbar('H_max', window_name, 0, 179, callback)   
    cv2.createTrackbar('S_min', window_name, 0, 255, callback)   
    cv2.createTrackbar('S_max', window_name, 0, 255, callback)  
    cv2.createTrackbar('V_min', window_name, 0, 255, callback)  
    cv2.createTrackbar('V_max', window_name, 0, 255, callback)  
    # set initial trackbar values
    cv2.setTrackbarPos('H_min', window_name, 0)
    cv2.setTrackbarPos('H_max', window_name, 179)
    cv2.setTrackbarPos('S_min', window_name, 0)
    cv2.setTrackbarPos('S_max', window_name, 255)
    cv2.setTrackbarPos('V_min', window_name, 0)
    cv2.setTrackbarPos('V_max', window_name, 255)
    # call callback to show HSV bounds before trackbar is moved
    callback(0)
    # wait for 'ESC' destroy windows
    while cv2.waitKey(200) & 0xFF != 27:
        pass
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    
def hsv_select_live():
    '''Thresholds live video via HSV Trackbar values.'''
    window_name = 'HSV Select Live' 
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    video = cv2.VideoCapture(video_port)
    hsv_min = np.array([0,0,0])
    hsv_max = np.array([179,255,255])
    def update():
        frame = video.read()[1]
        if frame is not None:
            #frame = cv2.flip(cv2.resize(video.read()[1], (640, 480)), 1)
            frame = cv2.resize(frame, (640,480))
            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_min[0] = cv2.getTrackbarPos('H_min', window_name)
            hsv_max[0] = cv2.getTrackbarPos('H_max', window_name)
            hsv_min[1] = cv2.getTrackbarPos('S_min', window_name)
            hsv_max[1] = cv2.getTrackbarPos('S_max', window_name)
            hsv_min[2] = cv2.getTrackbarPos('V_min', window_name)
            hsv_max[2] = cv2.getTrackbarPos('V_max', window_name)
            mask = cv2.inRange(img_hsv, hsv_min, hsv_max)
            img_masked = cv2.bitwise_and(frame, frame, mask = mask)
            h,w,ch = frame.shape
            cv2.putText(img_masked, 'HSV Lower: {}'.format(hsv_min), (10, 35), 0, 0.75, (255, 255, 255), 2)
            cv2.putText(img_masked, 'HSV Upper: {}'.format(hsv_max), (10, 70), 0, 0.75, (255, 255, 255), 2)
            cv2.imshow(window_name, img_masked)  
    def callback(value):
        update()  
    # make the trackbar used for HSV masking    
    cv2.createTrackbar('H_min', window_name, 0, 179, callback) 
    cv2.createTrackbar('H_max', window_name, 0, 179, callback)   
    cv2.createTrackbar('S_min', window_name, 0, 255, callback)   
    cv2.createTrackbar('S_max', window_name, 0, 255, callback)  
    cv2.createTrackbar('V_min', window_name, 0, 255, callback)  
    cv2.createTrackbar('V_max', window_name, 0, 255, callback) 
    # set initial trackbar values
    cv2.setTrackbarPos('H_min', window_name, 0)
    cv2.setTrackbarPos('H_max', window_name, 179)
    cv2.setTrackbarPos('S_min', window_name, 0)
    cv2.setTrackbarPos('S_max', window_name, 255)
    cv2.setTrackbarPos('V_min', window_name, 0)
    cv2.setTrackbarPos('V_max', window_name, 255)
    # wait for 'ESC' destroy windows
    while cv2.waitKey(200) & 0xFF != 27:
        update()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
#############################
#### Feature Detection
#############################

def find_object(img, img_q, total_matches, MIN_MATCH_COUNT, kp_img, kp_frame, m, good_matches, color, query_columns):
    '''
    Draws an outline around a detected objects given matches and keypoints.
    
    If enough matches are found, extract the locations of matched keypoints in both images. 
    The matched keypoints are passed to find the 3x3 perpective transformation matrix.
    Use transformation matrix to transform the corners of img to corresponding points in trainImage. 
    Draw matches.
    '''
    if total_matches > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp_img[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_frame[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) 
        matchesMask = mask.ravel().tolist()

        h,w,ch = img_q.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        
        dst[:,:,0] += query_columns
        img = cv2.polylines(img,[np.int32(dst)], True, color ,3, cv2.LINE_AA)
        dst = None
    #else:
        #print ("Not enough matches are found - %d/%d" % (total_matches, MIN_MATCH_COUNT))
        

def draw_matches(img, frame, total_keypoints, matches, kp_img, kp_frame):
    '''FLANN based Matcher:
    Reference https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    '''
    # Number of successful matches
    total_matches = 0
    
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # store all the good matches as per Lowe's ratio test.
    good_matches = []
    i = 0
    m = []
    for x in list(enumerate(matches)):
        if len(x[1]) == 2:
            if x[1][0].distance < 0.7*x[1][1].distance:
                matchesMask[i]=[1,0]
                total_matches += 1
                good_matches.append(x[1][0])
                m.append(x[1][0])
        i += 1
     
    # Getting percentages and putting it on screen
    match_percent = round(total_matches/total_keypoints, 3)
    
    cv2.putText(frame,'Match:{}'.format(match_percent),(0, 100), 1, 4,(255,255,255), 1, cv2.LINE_AA)
    
    # params for drawMatchesKnn
    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
    
    img_matches = cv2.drawMatchesKnn(img, kp_img, frame, kp_frame, matches, None, **draw_params)
    
    return img_matches, total_matches, m, good_matches

#############################
#### Hough Lines
#############################

def get_hough_lines(img):
    '''Returns an array with rho,theta points (Polar Coordinates) for all the lines found in the image'''
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    lines = cv2.HoughLines(edges,rho = 1, theta = 1*np.pi/180, threshold = 200)
    return lines

def draw_hough_lines(img, lines):
    '''Returns the img with hough lines drawn on it.'''
    if (lines is not None):
        for x in range(0, len(lines)):
            for rho, theta in lines[x]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b)) # X coordinate of point 1
                y1 = int(y0 + 1000*(a))  # Y coordinate of point 1
                x2 = int(x0 - 1000*(-b)) # X coordinate of point 2
                y2 = int(y0 - 1000*(a))  # Y coordinate of point 2
        
                # Using the the 2 points defined above, call cv2.line() to draw the line on the image
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

#############################
#### Checkerboards
#############################                

def draw_checkerboard_center(img):
    # convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    # locate checkerboard
    ret, corn = cv2.findChessboardCorners(gray, (7,7), None) 
    x = 0
    y = 0
    # if the checkerboard is found ...
    if ret == True:  
        # draw checkerboard on img
        img = cv2.drawChessboardCorners(img, (7,7), corn, ret) 
        x = corn[24,0 ,0]
        y = corn[24,0,1]
        # draw a circle arround the center of the checkerboard
        cv2.circle(img,(x,y),30,(0,0,255), 3) 
    # return the annotated img and the x coordinate of the center of the checkerboard
    return img, x, y  


                
