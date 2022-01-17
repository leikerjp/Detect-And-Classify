'''
    Camera Settings

    Raspberry Pi 3 with RPI Camera v2

    ***INREDIBLY IMPORTANT NOTE***
    The GPU Memory was bumped up in /boot/config.txt
'''
with open("camera_id.txt", 'r') as file:
    id = file.readlines()
    id = id[0]

camera_metadata = dict(id=id, name="rpi3-cam-0")
camera_resolution = (2592,1952) # 320,240 - 240p
camera_framerate = 10

'''
    Image Settings
    
    Describes the images we want to send over the REST API to the server 
    which will be logged in our database.
'''
image_resolution = camera_resolution
convert_to_gray = False
draw_detected_keypoints = True

'''
    Motion Detector Settings
    
    These settings will need to be manually tuned depending on what objects we are trying to 
    detect and the camera settings themselves.
'''
background_ratio = 0.7  # set to match Matlab
background_nmixtures = 3  # set to match Matlab
background_history = 1000  # 500 is default

blob_minThreshold = 0
blob_maxThreshold = 254
blob_thresholdStep = 253
blob_minDistBetweenBlobs = 50
blob_filterByArea = True
blob_minArea = (camera_resolution[0] * camera_resolution[1]) / 50 # Manually tuned
blob_maxArea = (camera_resolution[0] * camera_resolution[1]) / 1.5  # Manually tuned
blob_filterByColor = False
blob_filterByCircularity = False
blob_filterByConvexity = False
blob_filterByInertia = False

morph_open_size = (5,5)
morph_close_size = (8,8)

border_width = 5