#!/usr/bin/env python3

import cv2, sys, time

import read

import logging
logging.basicConfig(level = logging.INFO, # Messages on terminal
                    format = '%(asctime)s %(message)s',
                    datefmt = '%H:%M:%S')


def main(paths, day=2, camera=3, initial_frame=0, num_frames=36000, limbo=True):
    
    final_frame = initial_frame+num_frames-1
    if num_frames < 1:
        logging.info('Number of frames needs to be higher than 0\n')
        sys.exit()
    if final_frame > 35999:
        if initial_frame > 35999:
            logging.info('Invalid initial frame (0-35999)\n')
            sys.exit()
        else:
            final_frame = 35999
            num_frames = final_frame+1-initial_frame

    data_path = paths['data_path']
    videos_path = paths['videos_path']

    # Video to load
    video_file = '30min_day'+str(day)+'_cam'+str(camera)+'_20fps_960x540.MP4'
    video_name = 'Day '+str(day)+' Camera '+str(camera)
        
    cap = cv2.VideoCapture(videos_path+video_file) # Capture object to read video
    if not cap.isOpened(): # Exit if video not opened
        logging.info('Could not open the video\n')
        sys.exit()
    cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame) # Set the first frame to read
    logging.info(f'Selected video: {video_name}\n')
    
    # Read annotations
    logging.info('Getting annotations ...\n')
    det_full, cam_full, lost_full = read.read_annotations(path=data_path, day=day, initial_frame=initial_frame, num_frames=num_frames)

    frame_index = initial_frame
    
    ti = time.time() # Start timer
    blue = (72, 49, 40)
    gray = (187, 187, 187)
    white = (201, 238, 233)
    
    # Process video
    while cap.isOpened() and (frame_index <= final_frame) and (frame_index <= 35999):
        ret, frame = cap.read() # Read new frame
        if not ret: # Exit if reading failure
            logging.info('Unable to read the video file\n')
            break
        
        # Print frame number on image
        cv2.putText(frame, 'Frame:', (25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, blue, 2)
        cv2.putText(frame, str(frame_index), (25,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, blue, 2)
        
        # Read bounding boxes from the frame
        annotations, centers, num_part = read_detections(det=det_full, cam=cam_full, lost=lost_full, camera=camera, frame_index=frame_index-initial_frame)

        # Print number of people annotated
        cv2.putText(frame, 'People:', (875,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, blue, 2)
        cv2.putText(frame, str(num_part), (900,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, blue, 2)

        # Draw bounding boxes
        for box in annotations.values():
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[2]), int(box[3]))
            cv2.rectangle(frame, p1, p2, white, 1 , 1)

        # Draw centers
        for i, target in centers.items():
            center = (int(target[0]), int(target[1]))
            cv2.circle(frame, center, 3, white, -1, 1)
            center = (int(target[0])+5, int(target[1])-5)
            cv2.putText(frame, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, gray, 2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'): # Pause on P button
            while True:
                key2 = cv2.waitKey(1) & 0xFF
                cv2.imshow(video_name, frame)
                if key2 == ord('p'): # Continue on P button
                    break
        cv2.imshow(video_name, frame)
        
        if key == ord('q'): # Quit on Q button
            break
        
        cv2.waitKey(0)
        
        frame_index += 1

    tf = time.time() # End timer
    t_tot = tf-ti

    # Releasing objects
    cap.release()
    cv2.destroyAllWindows()

    logging.info(f'Elapsed time: {t_tot} seconds\n')
    logging.info(f'{frame_index-initial_frame} frames processed ({initial_frame}-{frame_index-1})\n')


def read_detections(det, cam, lost, camera, frame_index):
    
    det = det[frame_index]
    cam = cam[frame_index]
    lost = lost[frame_index]
    
    if limbo:
        l = (82, 72.5)
    else:
        l = (0,0)

    detections = {}
    centers = {}
    num_part = 0

    for i, num_cam in enumerate(cam):
        if (num_cam == camera) and (lost[i] == 0): # If the person is annotated on the right camera and is not lost
            detection = (det[1+i*7], det[2+i*7], det[3+i*7], det[4+i*7]) # Coordinates of bboxes.
            center = get_center(box=detection)
            if good_box(box=detection, center=center, limbo=l): # Check if valid box
                detections[i] = detection
                centers[i] = center
                num_part += 1
        
    return detections, centers, num_part

def good_box(box, center, limbo):
    limbo = limbo
    if (box != (0,0,0,0)) and (limbo[0] <= center[0] < 960-limbo[0]) and (limbo[1] <= center[1] < 540-limbo[1]): # If box inside permitted region
        approved = True
    else:
        approved = False 
    return approved

def get_center(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
        
    xc = (x1+x2)/2
    yc = (y1+y2)/2
    center = (xc, yc)
    return center


if __name__ == '__main__':
    limbo = True
    try:
        day = int(sys.argv[1])
        camera = int(sys.argv[2])
        initial_frame = int(sys.argv[3])
        num_frames = int(sys.argv[4])
    except:
        print('Parameters not given correctly\n')
        print('Usage:\n\tpython3 annotations.py day camera initial_frame num_frames\n')
        print('Example:\n\tpython3 annotations.py 2 3 700 300\n')
        sys.exit()

    paths = read.read_paths()

    main(paths, day, camera, initial_frame, num_frames, limbo)