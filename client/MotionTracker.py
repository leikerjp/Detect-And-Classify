'''
  Author:  Jordan Leiker
  Last Date Modified: 07/24/2021

  Description:
  This is a MotionDetect class. This class and the algorithm are taken directly from a publicly available Matlab motion tracking example
  and ported to OpenCV-Python. The algorithm and Matlab code can be found here:
  https://www.mathworks.com/help/vision/ug/motion-based-multiple-object-tracking.html
'''
import cv2 as cv
import numpy as np
import copy


class Track(object):
    __lastId = 1

    def __init__(self, centroid, kalman_filter):
        self.id = Track.__lastId
        self.centroid = centroid
        self.kalman_filter = kalman_filter
        self.age = 1
        self.total_visible_count = 1
        self.consecutive_invisible_count = 0
        self.flag = True

        Track.__lastId += 1


class MotionTracker(object):
    '''
    The MotionTracker class detects motion and tracks the motion using Kalman Filters. To track, the pipeline is:
        --- Motion Detection ---
        (1) performs background subtraction
        (2) does morphological noise reduction
        (3) does blob detection
        --- Motion Tracking ---
        (4) Path assignment
        (5) Kalman Filtering
    '''
    invisible_for_too_long = 90;
    age_threshold = 30
    visibility_threshold = 0.6

    def __init__(self, height, width, fps, background_subtractor, blob_detector, open_element, close_element):
        self.image_resolution = (height, width) # row/col == height/width
        self.framerate = fps
        self.track_norm_cutoff = (self.image_resolution[0] * self.image_resolution[1]) / 1000.0
        self.background_subtractor = background_subtractor
        self.blob_detector = blob_detector
        self.open_element = open_element
        self.close_element = close_element
        self.detected_keypoints = []
        self.tracks = []

    def _detect(self, frame):
        '''
        Perform motion detection on a given input frame. The output is the masked image used in blob detection as well
        as the centroids of the detected blobs
        :param frame:
        :return:
        '''
        # back ground subtract
        foreground_mask = self.background_subtractor.apply(frame)

        # clean up noise
        foreground_mask = cv.morphologyEx(foreground_mask, cv.MORPH_OPEN, self.open_element)
        foreground_mask = cv.morphologyEx(foreground_mask, cv.MORPH_CLOSE, self.close_element)

        # threshold out shadows, invert for blob detect
        ret, foreground_mask = cv.threshold(foreground_mask, 1, 255, cv.THRESH_BINARY_INV);
        self.fg = foreground_mask
        # blob detect
        self.detected_keypoints = self.blob_detector.detect(foreground_mask)

    def _predictNewLocationsOfTracks(self):
        '''
        For every detected track use Kalman to predict next position
        :param None:
        :return None:
        '''
        # print(len(self.tracks))
        for track in self.tracks:
            # stpst = track.kalman_filter.statePost
            # print("prepredict" + str((stpst[0, 0], stpst[1, 0])))
            prediction = track.kalman_filter.predict()
            track.centroid.pt = (prediction[0,0], prediction[1,0])
            # print("center" + str(track.centroid.pt))

    def _createNewTrack(self, centroid):
        '''
        Create a new track based on a given centroid. The new track will initialize a Kalman filter as well as track
        visibility parameters.
        Note: Kalman filter tracks using acceleration, velocity, and initial position, i.e. x(t) = 1/2 a(t)^2 + v(t) + x0
        :param centroid:
        :return Track:
        '''
        dt = 1.0 / self.framerate

        kalman_filter = cv.KalmanFilter(6, 2, 0)
        kalman_filter.statePost = np.array([[centroid.pt[0], centroid.pt[1], 0, 0, 0, 0]]).T
        kalman_filter.transitionMatrix = np.array([[1, 0, dt, 0, dt * dt, 0],
                                                   [0, 1, 0, dt, 0, dt * dt],
                                                   [0, 0, 1, 0, dt, 0],
                                                   [0, 0, 0, 1, 0, dt],
                                                   [0, 0, 0, 0, 1, 0],
                                                   [0, 0, 0, 0, 0, 1]])
        kalman_filter.measurementMatrix = 1.0 * np.eye(2,6)
        kalman_filter.processNoiseCov = 1e-5 * np.eye(6,6)
        kalman_filter.measurementNoiseCov = 1e-1 * np.eye(2,2)
        kalman_filter.errorCovPost = 1.0 * np.eye(6,6)

        return Track(centroid, kalman_filter)


    def _deleteLostTracks(self):
        '''
        Manage track list by creating a new list, adding valid tracks to the new list, then overriding the old list
        once all tracks are parsed. "Invalid" tracks are tracks that have been invisibile for too long
        :param None:
        :return None:
        '''

        cleaned_tracks = list.copy(self.tracks)
        for track in self.tracks:
            visibility = track.total_visible_count / track.age
            if (track.consecutive_invisible_count >= MotionTracker.invisible_for_too_long) \
                    or (track.age < MotionTracker.age_threshold and visibility < MotionTracker.visibility_threshold):
                cleaned_tracks.remove(track)

        self.tracks = list.copy(cleaned_tracks)


    def _assignDetectionsToTracks(self):
        '''
        Assign all detected centroids to tracks. A centroid is assigned to a track if the centroid is within a
        cutoff distance to the predicted location. If the centroid is assigned the Kalman filter is corrected with
        the centroid location. If the centroid is not assigned to any track a new track is created.
        :param None:
        :return None:
        '''
        # Set all track flags to false initially so we can keep track of which tracks got a new detection
        for track in self.tracks:
                track.flag = False

        # Cycle through all detected centroids and attempt to assign it to a track based on L2 distance
        for detected_keypoint in self.detected_keypoints:
            assigned = False
            for track in self.tracks:
                current_prediction = track.kalman_filter.statePost
                current_prediction = (current_prediction[0,0], current_prediction[1,0])
                if (np.linalg.norm(np.subtract(current_prediction, detected_keypoint.pt)) < self.track_norm_cutoff) \
                        and (track.flag == False):
                    # stpst = track.kalman_filter.statePost
                    # print("precorrect" + str((stpst[0, 0], stpst[1, 0])))
                    track.kalman_filter.correct(detected_keypoint.pt)
                    # stpst = track.kalman_filter.statePost
                    # print("postcorrect" + str((stpst[0, 0], stpst[1, 0])))
                    track.centroid = detected_keypoint
                    track.age += 1
                    track.total_visible_count += 1
                    track.consecutive_invisible_count = 0
                    track.flag = True
                    assigned = True
                    break

            if not assigned:
                self.tracks.append(self._createNewTrack(detected_keypoint))

        # Manage tracks that haven't had an assignment
        for track in self.tracks:
            if not track.flag:
                track.age += 1
                track.consecutive_invisible_count += 1

    def detect_and_track(self, frame):
        '''
        Iterate one frame through the motion detector and tracker. A list of all current tracked centroids is
        returned.
        :param frame:
        :return (list of tracked centroids):
        '''
        # Run motion detector on frame
        self._detect(frame)

        # Create centroid predictions of for each track
        self._predictNewLocationsOfTracks()

        # Assign detections to tracks / create new tracks as needed
        self._assignDetectionsToTracks()

        # Clean up old tracks
        self._deleteLostTracks()

        #return [track.centroid for track in self.tracks]
        return self.tracks






