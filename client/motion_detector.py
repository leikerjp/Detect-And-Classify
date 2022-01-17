import requests
import numpy
import cv2
import time

from MotionTracker import *
from settings import *

# Setup capture device
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1])
cam.set(cv2.CAP_PROP_FPS, camera_framerate)

# Setup motion detector
background_subtractor = cv2.createBackgroundSubtractorMOG2()
background_subtractor.setBackgroundRatio(background_ratio)
background_subtractor.setNMixtures(background_nmixtures)
background_subtractor.setHistory(background_history)

blob_params = cv2.SimpleBlobDetector_Params()
blob_params.minThreshold = blob_minThreshold
blob_params.maxThreshold = blob_maxThreshold
blob_params.thresholdStep = blob_thresholdStep
blob_params.minDistBetweenBlobs = blob_minDistBetweenBlobs
blob_params.filterByArea = blob_filterByArea
blob_params.minArea = blob_minArea
blob_params.maxArea = blob_maxArea
blob_params.filterByColor = blob_filterByColor
blob_params.filterByCircularity = blob_filterByCircularity
blob_params.filterByConvexity = blob_filterByConvexity
blob_params.filterByInertia = blob_filterByInertia
blob_detector = cv.SimpleBlobDetector_create(blob_params)

open_element = cv2.getStructuringElement(cv2.MORPH_RECT, morph_open_size)
close_element = cv2.getStructuringElement(cv2.MORPH_RECT, morph_close_size)
mTracker = MotionTracker(height=image_resolution[1],
                         width=image_resolution[0],
                         fps=camera_framerate,
                         background_subtractor=background_subtractor,
                         blob_detector=blob_detector,
                         open_element=open_element,
                         close_element=close_element)

# Setup times so that we've already exceeded the elapsed time and are guaranteed to send when we get a detection
time_to_wait_in_seconds = 5
previous_time_in_seconds = 0
current_time_in_seconds = time.time()
while(1):

    # Read a frame and resize (size was arbitrarily picked)
    ret, frame = cam.read()
    if camera_resolution != image_resolution:
        frame = cv2.resize(frame, image_resolution)
    if convert_to_gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Run motion detector
    # Add a border so that blob detector doesn't fail when object is near edge
    frameBord = cv2.resize(frame, (image_resolution[0]-2*border_width,image_resolution[1]-2*border_width))
    frameBord = cv2.copyMakeBorder(frameBord, top=5, bottom=5, left=5, right=5, borderType=cv2.BORDER_CONSTANT,value=[255, 255, 255])

    # Track and sort returned/active tracks
    tracks = mTracker.detect_and_track(frameBord)
    tracks.sort(key=lambda track: track.id)
    keypoints = [track.centroid for track in tracks]
    motion_detected = len(keypoints) > 0

    # Calculate elapsed time since last sent message, we only want to send if time has been
    # great than 5 seconds.
    time_elapsed = current_time_in_seconds - previous_time_in_seconds > time_to_wait_in_seconds

    # Send any detected motion to server
    if motion_detected and time_elapsed:

        if draw_detected_keypoints:
            frame = cv.drawKeypoints(frame, keypoints, outImage=np.array([]), color=(0, 0, 255), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Serialize the image in preparation to send over REST API
        imencoded = cv2.imencode(".jpg", frame)[1]
        data_encode = numpy.array(imencoded).tobytes()
        file = dict(file=('image.jpg', data_encode, 'image/jpeg', {'Expires': '0'}))

        # Send to server
        response = requests.post('http://192.168.0.122:5000/log_image', files=file, data=camera_metadata)

        # If we get a response it's to update our camera's id because
        # this is the first time we sent a message
        if response.content:
            respDict = response.json()
            camera_metadata['id'] = respDict['id']

            with open("camera_id.txt", 'w') as file:
                # Using writelines to overwrite the file, but with just the one line
                # (need to be sure we don't write to the file, but OVERWRITE the file)
                file.writelines([str(camera_metadata['id'])])

        # Update previous time (because we just did a send)
        previous_time_in_seconds = current_time_in_seconds

    # The last thing in the loop is to update the current time for the elapsed time calculation
    current_time_in_seconds = time.time()


# When everything done, release the capture (though we really won't get here)
cam.release()
cv2.destroyAllWindows()
