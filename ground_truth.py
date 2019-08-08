#!/usr/bin/env python3

import numpy as np, sys, csv, os

import read

import logging
logging.basicConfig(level = logging.INFO, # Messages on terminal
                    format = '%(asctime)s %(message)s',
                    datefmt = '%H:%M:%S')


def ground_truth(paths, day=2, camera=3, limbo=True):

    data_path = paths['data_path']
    output_path = paths['output_path']
    if not os.path.isdir(output_path):
        logging.info('Output directory does not exist\n')
        sys.exit()
    
    gt_path = output_path+'ground_truth/'
    if not os.path.isdir(gt_path):
        os.mkdir(gt_path)
    
    video_name = 'Day '+str(day)+' Camera '+str(camera)
    logging.info(f'Selected video: {video_name}\n')

    initial_frame = 0
    num_frames = 36000

    logging.info('Getting ground truth bounding boxes ...\n')

    # Read annotations
    logging.info('Getting annotations ...\n')
    det_full, cam_full, lost_full = read.read_annotations(path=data_path, day=day, initial_frame=initial_frame, num_frames=num_frames)

    gtruth = {}
    start_frame = {}
    frame_print = set(np.arange(-1, num_frames-1, 500)) # To print frame every 500 frames
    for frame_index in range(num_frames):
        if frame_index in frame_print:
            logging.info(f'Frame: {frame_index} ...')
        
        annotations = read_detections(det=det_full, cam=cam_full, lost=lost_full, camera=camera, frame_index=frame_index, limbo=limbo)
        for i, annot in annotations.items():
            if i in gtruth.keys():
                gtruth[i].append(annot)
            else:
                gtruth[i] = [annot]
                start_frame[i] = frame_index
        for j in gtruth.keys():
            if j not in annotations.keys():
                gtruth[j].append(None)
    
    for k, frame in start_frame.items():
        if frame != initial_frame:
            missing = frame
            gtruth[k] = [None]*missing + gtruth[k]
    
    if limbo:
        gtruth_file = gt_path+'GroundTruth_day'+str(day)+'_cam'+str(camera)+'_limbo'+'.csv'
    else:
        gtruth_file = gt_path+'GroundTruth_day'+str(day)+'_cam'+str(camera)+'.csv'
    
    write_csv(file_name=gtruth_file, gtruth=gtruth)


def read_detections(det, cam, lost, camera, frame_index, limbo):
    
    det = det[frame_index]
    cam = cam[frame_index]
    lost = lost[frame_index]
    
    if limbo:
        l = (82, 72.5)
    else:
        l = (0,0)

    detections = {}
    
    for i, num_cam in enumerate(cam):
        if (num_cam == camera) and (lost[i] == 0): # If the person is annotated on the right camera and is not lost
            detection = [det[1+i*7], det[2+i*7], det[3+i*7], det[4+i*7]] # Coordinates of bboxes.
            center = get_center(box=detection)
            if good_box(box=detection, center=center, limbo=l): # Check if valid box
                detections[i] = np.array(detection)
        
    return detections

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

def write_csv(file_name, gtruth):
    logging.info('Writing CSV ...\n')
    csv_rows = []

    frames = len(list(gtruth.values())[0])
    for frame_index in range(frames):
        line = []
        for person in sorted(gtruth.keys()):
            box = gtruth[person][frame_index]
            if box is None:
                x1 = y1 = x2 = y2 = -1
            else:
                x1, y1, x2, y2 = [x for x in box]
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
        l = sys.argv[3]
    except:
        print('Parameters not given correctly\n')
        print('Usage:\n\tpython3 ground_truth.py day camera limbo\n')
        print('Example:\n\tpython3 ground_truth.py 2 3 true\n')
        sys.exit()

    if (l == 'true') or (l == 'True'):
        limbo = True
    elif (l == 'false') or (l == 'False'):
        limbo = False
    else:
        print('Parameter "limbo" should be "true" or "false"\n')
        print('Usage:\n\tpython3 ground_truth.py day camera limbo\n')
        print('Example:\n\tpython3 ground_truth.py 2 3 true\n')
        sys.exit()

    paths = read.read_paths()

    ground_truth(paths, day, camera, limbo)