#!/usr/bin/env python3

import cv2, sys, time, csv, numpy as np, os

import read

from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment

import logging
logging.basicConfig(level = logging.INFO, # Messages on terminal
                    format = '%(asctime)s %(message)s',
                    datefmt = '%H:%M:%S')


class Track():
    def __init__(self, initial_box, missing_frames):
        self.coordinates = [None]*missing_frames + [initial_box]

    def update(self, box):
        self.coordinates.append(box)
    
    def get_coordinates(self):
        return self.coordinates


def main(paths, day=2, camera=3, initial_frame=0, num_frames=36000, iou_th=10):
    
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
    output_path = paths['output_path']
    if not os.path.isdir(output_path):
        logging.info('Output directory does not exist\n')
        sys.exit()
        
    video_name = 'Day '+str(day)+' Camera '+str(camera)
    logging.info(f'Selected video: {video_name}\n')
    
    frame_index = initial_frame

    fps_acc = 0 # Processing speed

    hung_path = output_path+'hungarian/'
    if not os.path.isdir(hung_path):
        os.mkdir(hung_path)

    # Read annotations
    logging.info('Getting annotations ...\n')
    det_full, cam_full, lost_full = read.read_annotations(path=data_path, day=day, initial_frame=initial_frame, num_frames=num_frames)

    res_file = hung_path+'Results_day'+str(day)+'_cam'+str(camera)+'_'+str(iou_th)
    speed_file = hung_path+'Speed_day'+str(day)+'_cam'+str(camera)+'_'+str(iou_th)
    runtime_file = hung_path+'Time_day'+str(day)+'_cam'+str(camera)+'_'+str(iou_th)

    logging.info('Running Hungarian algorithm ...\n')
    ti = time.time() # Start timer

    tracks = [] # List of lists
    #########################################
    frame_print = set(np.arange(initial_frame-1, final_frame, 100)) # To print frame every 100 frames
    frame_save = set(np.arange(initial_frame-1, final_frame, 1000)) # To save results every 1000 frames
    
    while (frame_index <= final_frame) and (frame_index <= 35999):
        if frame_index in frame_print:
            logging.info(f'Frame: {frame_index} ...')
        
        timer = cv2.getTickCount() # Start timer to get FPS
        # Read annotations
        annotations = read_detections(det=det_full, cam=cam_full, lost=lost_full, camera=camera, frame_index=frame_index-initial_frame) # List of np arrays

        track_boxes, ids = [], []
        for i, track in enumerate(tracks):
            trk_coord = track.get_coordinates()
            box = trk_coord[len(trk_coord)-1]
            if box is not None:
                track_boxes.append(box)
                ids.append(i)

        matches, unmatched_detections, unmatched_tracks = assign_detections(tracks=track_boxes, detections=annotations, iou_th=iou_th)
        #matches2, unmatched_detections2, unmatched_tracks2 = assign_detections2(tracks=track_boxes, detections=annotations, iou_th=iou_th)

        if len(unmatched_detections) > 0:
            for det_id in unmatched_detections:
                box = annotations[det_id]
                tracks.append(Track(initial_box=box, missing_frames=frame_index-initial_frame))
        
        if len(unmatched_tracks) > 0:
            for trk_id in unmatched_tracks:
                real_id = ids[trk_id]
                tracks[real_id].update(None)

        if matches.size > 0:
            for trk_id, det_id in matches:
                real_id = ids[trk_id]
                tracks[real_id].update(annotations[det_id])

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer) # Compute frames per second (FPS) of the processing

        fps_acc += fps
        
        if frame_index in frame_save: # Save results
            tf = time.time() # End timer
            t_tot = tf-ti        
            fps_mean = fps_acc/(frame_index-initial_frame+1) # Average FPS
            logging.info(f'Elapsed time: {tf-ti} seconds\n')
            logging.info(f'{frame_index-initial_frame+1} frames processed ({initial_frame}-{frame_index})\n')
            
            #solution_coord = get_sol_coord(tracks=tracks, num_frames=frame_index-initial_frame+1)

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
    
    logging.info('Hungarian algorithm completed\n')
    logging.info(f'Elapsed time: {tf-ti} seconds\n')
    logging.info(f'{frame_index-initial_frame} frames processed ({initial_frame}-{frame_index-1})\n')
    
    solution_coord = get_sol_coord(tracks=tracks, num_frames=num_frames)

    # Save results to a .CSV file
    results_file = res_file+'_'+str(initial_frame)+'-'+str(frame_index-1)+'.csv'
    write_csv(file_name=results_file, solution_coordinates=solution_coord)

    fps_file = speed_file+'_'+str(initial_frame)+'-'+str(frame_index-1)+'.csv'
    np.savetxt(fps_file, [fps_mean], delimiter=',')
    time_file = runtime_file+'_'+str(initial_frame)+'-'+str(frame_index-1)+'.csv'
    np.savetxt(time_file, [t_tot], delimiter=',')


def assign_detections(tracks, detections, iou_th):
    iou_mat= np.zeros((len(tracks),len(detections)),dtype=np.float32)

    for t, trk in enumerate(tracks):
        for d, det in enumerate(detections):
            iou_mat[t, d] = iou(trk, det)

    matched_idx = linear_assignment(-iou_mat)

    unmatched_tracks, unmatched_detections = [], []
    for t, trk in enumerate(tracks):
        if (t not in matched_idx[:,0]):
            unmatched_tracks.append(t)

    for d, det in enumerate(detections):
        if (d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []
    
    for m in matched_idx:
        if (iou_mat[m[0], m[1]] < iou_th/100):
            unmatched_tracks.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if (len(matches) == 0):
        matches = np.empty((0,2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_tracks)

def assign_detections2(tracks, detections, iou_th):
    iou_mat= np.zeros((len(tracks),len(detections)),dtype=np.float32)

    for t, trk in enumerate(tracks):
        for d, det in enumerate(detections):
            iou_mat[t, d] = iou(trk, det)

    tracks_matched, det_matched = linear_sum_assignment(-iou_mat)

    unmatched_tracks, unmatched_detections = [], []
    for t, trk in enumerate(tracks):
        if (t not in tracks_matched):
            unmatched_tracks.append(t)

    for d, det in enumerate(detections):
        if (d not in det_matched):
            unmatched_detections.append(d)

    matches = []
    
    for track in tracks_matched:
        for det in det_matched:
            if (iou_mat[track, det]) < (iou_th/100):
                unmatched_tracks.append(track)
                unmatched_detections.append(det)
            else:
                m = np.array([track, det])
                matches.append(m.reshape(1,2))
    
    if (len(matches) == 0):
        matches = np.empty((0,2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_tracks)

def iou(a, b):
    w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    
    s_a = (a[2] - a[0])*(a[3] - a[1])
    s_b = (b[2] - b[0])*(b[3] - b[1])

    return float(s_intsec)/(s_a + s_b -s_intsec)

def get_sol_coord(tracks, num_frames):
    solution_coord = []

    for track in tracks:
        trk_coord = track.get_coordinates()
        if len(trk_coord) != num_frames:
            missing = num_frames-len(trk_coord)
            trk_coord = trk_coord + [None]*missing
        solution_coord.append(trk_coord)
    
    return solution_coord

def read_detections(det, cam, lost, camera, frame_index):
    
    det = det[frame_index]
    cam = cam[frame_index]
    lost = lost[frame_index]
    
    limbo = (82, 72.5)
    #limbo = (0,0)

    detections = []
    #num_part = 0

    for i, num_cam in enumerate(cam):
        if (num_cam == camera) and (lost[i] == 0): # If the person is annotated on the right camera and is not lost
            detection = [det[1+i*7], det[2+i*7], det[3+i*7], det[4+i*7]] # Coordinates of bboxes.
            center = get_center(box=detection)
            if good_box(box=detection, center=center, limbo=limbo): # Check if valid box
                detections.append( np.array(detection) )
                #num_part += 1
        
    return detections#, num_part

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
        iou_th = int(sys.argv[5])
    except:
        print('Parameters not given correctly\n')
        print('Usage:\n\tpython3 hungarian.py day camera initial_frame num_frames iou_th\n')
        print('Example:\n\tpython3 hungarian.py 2 3 700 300 25\n')
        sys.exit()

    paths = read.read_paths()

    main(paths, day, camera, initial_frame, num_frames, iou_th)