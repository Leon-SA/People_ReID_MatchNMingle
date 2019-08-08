#!/usr/bin/env python3

import cv2, sys, time, csv, random, numpy as np, os

from mht import MHT # MHT class
import read

import logging
logging.basicConfig(level = logging.INFO, # Messages on terminal
                    format = '%(asctime)s %(message)s',
                    datefmt = '%H:%M:%S')


def main(paths, day=2, camera=3, initial_frame=0, num_frames=36000, N_pruning=0):

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
    output_path = paths['output_path']
    if not os.path.isdir(output_path):
        logging.info('Output directory does not exist\n')
        sys.exit()
    
    # Video to load
    video_file = '30min_day'+str(day)+'_cam'+str(camera)+'_20fps_960x540.MP4'
    video_name = 'Day '+str(day)+' Camera '+str(camera)

    cap = cv2.VideoCapture(videos_path+video_file) # Capture object to read video
    if not cap.isOpened(): # Exit if video not opened
        logging.info('Could not open the video\n')
        sys.exit()
    cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame) # Set the first frame to read
    ret, frame = cap.read() # Read first frame
    if not ret: # Exit if reading failure
        logging.info('Unable to read the video file\n')
        sys.exit()
    logging.info(f'Selected video: {video_name}\n')
    
    # Read annotations
    logging.info('Getting annotations ...\n')
    det_full, cam_full, lost_full = read.read_annotations(path=data_path, day=day, initial_frame=initial_frame, num_frames=num_frames)

    # Read annotations from first frame
    frame_index = initial_frame
    annotations0, num_part0 = read_detections(det=det_full, cam=cam_full, lost=lost_full, camera=camera, frame_index=frame_index-initial_frame)
    logging.info(f'{num_part0} participants annotated on the frame {frame_index}\n')
    
    # Object MultiTracker (Primary Trackers)
    multi_tracker = cv2.MultiTracker_create()
    boxes, targets_tracked = change_det_boxes(init_boxes=annotations0) # From (x1,y1,x2,y2) to (x1,y1,width,height)
    for box in boxes: # Initialization of primary trackers for each target
        multi_tracker.add(cv2.TrackerKCF_create(), frame, box)
        multi_tracker.add(cv2.TrackerMedianFlow_create(), frame, box)
        multi_tracker.add(cv2.TrackerMIL_create(), frame, box)
        
    # MHT
    tracking_params = {'N_pruning': N_pruning, # Index for pruning
                'distance_threshold': 100, # Distance threshold for hypothesis formation
                'distance_threshold2': 75, # Distance threshold for the updating of primary trackers
                'MIL_weight': 0.2, # MIL Tracker weight on the track scoring
                'MF_weight': 0.35, # MF Tracker weight on the track scoring
                'KCF_weight': 0.45, # KCF Tracker weight on the track scoring
                'color_score_threshold': 0.20, # Bhattacharyya distance threshold score between color histograms for Re-ID 
                'color_score_weight': 0.75, # Color histograms weight on the lost tracks scoring
                'lost_time_threshold': 25, # Time of loss threshold for Re-ID
                'lost_time_weight': 0.25, # Time of loss weight on the lost tracks scoring
                'color_hist_bins': 4} # Number of bins per histogram
    mht = MHT(tracking_params) # Object MHT initialized
    logging.info('Running MHT ...\n')
    ti = time.time() # Start timer
    logging.info(f'Frame: {frame_index} ...')
    mht.init(frame=frame, detections=annotations0) # Run MHT on first frame.

    ids = [] # ID's of targets tracked at each frame
    fps_acc = 0 # Processing speed
    
    trk_path = output_path+'tracker/'
    if not os.path.isdir(trk_path):
        os.mkdir(trk_path)

    res_file = trk_path+'Results_day'+str(day)+'_cam'+str(camera)+'_'+str(tracking_params['N_pruning'])
    speed_file = trk_path+'Speed_day'+str(day)+'_cam'+str(camera)+'_'+str(tracking_params['N_pruning'])
    runtime_file = trk_path+'Time_day'+str(day)+'_cam'+str(camera)+'_'+str(tracking_params['N_pruning'])

    frame_index += 1
    #########################################
    frame_print = set(np.arange(initial_frame-1, final_frame, 100)) # To print frame every 100 frames
    frame_save = set(np.arange(initial_frame-1, final_frame, 1000)) # To save results every 1000 frames

    # Process video and track objects
    while cap.isOpened() and (frame_index <= final_frame) and (frame_index <= 35999):
        ret, frame = cap.read() # Read new frame
        if not ret: # Exit if reading failure
            break

        if frame_index in frame_print:
            logging.info(f'Frame: {frame_index} ...')

        timer = cv2.getTickCount() # Start timer to get FPS
        # Read annotations
        annotations, num_part = read_detections(det=det_full, cam=cam_full, lost=lost_full, camera=camera, frame_index=frame_index-initial_frame)
        
        if num_part < num_part0:
            logging.info(f'Number of annotations changed from {num_part0} to {num_part} on frame {frame_index}\n')
            num_part0 = num_part
        if num_part > num_part0:
            logging.info(f'Number of annotations changed from {num_part0} to {num_part} on frame {frame_index}\n')
            num_part0 = num_part

        # Update primary trackers for the current frame
        ret, multitracker_results = multi_tracker.update(frame)
        # Turn results from primary trackers into a dictionary for MHT
        trackers_results = change_track_boxes(init_boxes=multitracker_results, indexes=targets_tracked)
        
        # Run MHT with annotations (detections) and tracker results
        solution_coord, track_ids, new_tracks = mht.run(frame=frame, detections=annotations, trackers_results=trackers_results)
        
        # Update primary trackers when they're lost or there are new targets
        if len(new_tracks) != 0:
            for key in new_tracks.keys():
                if key in ids: # If an existing tracker needs to be updated
                    i = targets_tracked.index(key)
                    targets_tracked[i] = random.randint(50,500) # Change ID
                targets_tracked.append(key) # Append corresponding ID

                # Append new trackers for the target
                box = new_tracks[key]
                new_box = (box[0], box[1], box[2]-box[0], box[3]-box[1])
                multi_tracker.add(cv2.TrackerKCF_create(), frame, new_box)
                multi_tracker.add(cv2.TrackerMedianFlow_create(), frame, new_box)
                multi_tracker.add(cv2.TrackerMIL_create(), frame, new_box)

        ids = track_ids # Update track ID's

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer) # Compute frames per second (FPS) of the processing
        
        fps_acc += fps

        if frame_index in frame_save: # Save results
            tf = time.time() # End timer
            t_tot = tf-ti        
            fps_mean = fps_acc/(frame_index-initial_frame+1) # Average FPS
            logging.info(f'Elapsed time: {tf-ti} seconds\n')
            logging.info(f'{frame_index-initial_frame+1} frames processed ({initial_frame}-{frame_index})\n')

            # Save results to a .CSV file
            #results_file = res_file+'_'+str(initial_frame)+'-'+str(frame_index)+'.csv'
            #write_csv(file_name=results_file, solution_coordinates=solution_coord)

            fps_file = speed_file+'_'+str(initial_frame)+'-'+str(frame_index)+'.csv'
            np.savetxt(fps_file, [fps_mean], delimiter=',')
            time_file = runtime_file+'_'+str(initial_frame)+'-'+str(frame_index)+'.csv'
            np.savetxt(time_file, [t_tot], delimiter=',')

        frame_index += 1

    tf = time.time() # End timer
    t_tot = tf-ti

    fps_mean = fps_acc/(frame_index-initial_frame) # Average FPS
    
    logging.info('MHT completed\n')
    logging.info(f'Elapsed time: {t_tot} seconds\n')
    logging.info(f'{frame_index-initial_frame} frames processed ({initial_frame}-{frame_index-1})\n')
    
    # Save results to a .CSV file
    results_file = res_file+'_'+str(initial_frame)+'-'+str(frame_index-1)+'.csv'
    write_csv(file_name=results_file, solution_coordinates=solution_coord)

    fps_file = speed_file+'_'+str(initial_frame)+'-'+str(frame_index-1)+'.csv'
    np.savetxt(fps_file, [fps_mean], delimiter=',')
    time_file = runtime_file+'_'+str(initial_frame)+'-'+str(frame_index-1)+'.csv'
    np.savetxt(time_file, [t_tot], delimiter=',')


def read_detections(det, cam, lost, camera, frame_index):
    
    det = det[frame_index]
    cam = cam[frame_index]
    lost = lost[frame_index]

    limbo = (82, 72.5)
    #limbo = (0,0)

    detections = {}
    num_part = 0
    for i, num_cam in enumerate(cam):
        if (num_cam == camera) and (lost[i] == 0): # If the person is annotated on the right camera and is not lost
            detection = (det[1+i*7], det[2+i*7], det[3+i*7], det[4+i*7]) # Coordinates of bboxes.
            center = get_center(box=detection)
            if good_box(box=detection, center=center, limbo=limbo): # Check if valid box
                detections[num_part] = detection
                num_part += 1
        
    return detections, num_part

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

def change_det_boxes(init_boxes):
    boxes = []
    targets_tracked = []
    for i, box in init_boxes.items():
        boxes.append((box[0], box[1], box[2]-box[0], box[3]-box[1]))
        targets_tracked.append(i)
    return boxes, targets_tracked

def change_track_boxes(init_boxes, indexes):
    b = []
    for box in init_boxes:
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]+box[0]
        y2 = box[3]+box[1]
        b.append((x1, y1, x2, y2))
    boxes = {}
    for i in indexes:
        boxes[i] = []
        boxes[i].append(b.pop(0))
        boxes[i].append(b.pop(0))
        boxes[i].append(b.pop(0))
    return boxes

def write_csv(file_name, solution_coordinates):
    logging.info('Writing CSV ...\n')
    csv_rows = []
    people = len(solution_coordinates)
    frames = len(solution_coordinates[0])

    for frame in range(frames):
        line = []
        for track in range(people):
            coordinate = solution_coordinates[track][frame]
            if coordinate is None: # If target not present in the frame, coordinates will be (-1,-1,-1,-1)
                x1 = y1 = x2 = y2 = -1
            else:
                x1, y1, x2, y2 = [str(x) for x in coordinate]
            line = line + [x1, y1, x2, y2]
        csv_rows.append(line)
    
    with open(file_name, 'w') as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        writer.writerows(csv_rows)

    logging.info(f'CSV saved to: {file_name}\n')


if __name__ == '__main__':
    
    try:
        day = int(sys.argv[1])
        camera = int(sys.argv[2])
        initial_frame = int(sys.argv[3])
        num_frames = int(sys.argv[4])
        N_pruning = int(sys.argv[5])
    except:
        print('Parameters not given correctly\n')
        print('Usage:\n\tpython3 tracker.py day camera initial_frame num_frames N_pruning\n')
        print('Example:\n\tpython3 tracker.py 2 3 700 300 0\n')
        sys.exit()

    paths = read.read_paths()

    main(paths, day, camera, initial_frame, num_frames, N_pruning)