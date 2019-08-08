#!/usr/bin/env python3

import sys, csv, os

import read
from ground_truth import ground_truth

import logging
logging.basicConfig(level = logging.INFO, # Messages on terminal
                    format = '%(asctime)s %(message)s',
                    datefmt = '%H:%M:%S')


def participants_history(paths, day=2, camera=3, initial_frame=0, num_frames=36000, limbo=True):

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

    output_path = paths['output_path']
    if not os.path.isdir(output_path):
        logging.info('Output directory does not exist\n')
        sys.exit()
    
    gt_path = output_path+'ground_truth/'
    if not os.path.isdir(gt_path):
        os.mkdir(gt_path)

    video_name = 'Day '+str(day)+' Camera '+str(camera)
    logging.info(f'Selected video: {video_name}\n')

    if limbo:
        l = '_limbo.csv'
    else:
        l = '.csv'

    gt_name = gt_path+'GroundTruth_day'+str(day)+'_cam'+str(camera)+l
    
    if not os.path.isfile(gt_name):
        ground_truth(paths=paths, day=day, camera=camera, limbo=limbo)

    # Read ground truth
    logging.info('Getting ground truth ...\n')
    gtruth_full = read.read_ground_truth(filename=gt_name, initial_frame=initial_frame, num_frames=num_frames)

    people = int(len(gtruth_full[0])/4)
    present = {}
    history = {}

    for frame in range(num_frames):
        if frame > 35999:
            break
        
        det = gtruth_full[frame]
        for p in range(people):
            box = (det[p*4], det[1+p*4], det[2+p*4], det[3+p*4])
            if box != (-1,-1,-1,-1):
                if p not in present.keys():
                    present[p] = frame
            else:
                if p in present.keys():
                    h = (present[p], frame-1)
                    if p not in history.keys():
                        history[p] = [h]
                    else:
                        history[p].append(h)
                    del present[p]
    
    for p, frame in present.items():
        h = (frame, final_frame)
        if p not in history.keys():
            history[p] = [h]
        else:
            history[p].append(h)

    ph_path = output_path+'participants_history/'
    if not os.path.isdir(ph_path):
        os.mkdir(ph_path)

    file_name = ph_path+'ParticipantsHistory_day'+str(day)+'_cam'+str(camera)+'_'+str(initial_frame)+'-'+str(final_frame)+l
    write_csv(file_name=file_name, history=history)


def write_csv(file_name, history):
    logging.info('Writing CSV ...\n')
    csv_rows = []

    for p, hist in history.items():
        for pair in hist:
            line = [p, pair[0], pair[1]]
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
        l = sys.argv[5]
    except:
        print('Parameters not given correctly\n')
        print('Usage:\n\tpython3 participants_history.py day camera initial_frame num_frames limbo\n')
        print('Example:\n\tpython3 participants_history.py 2 3 700 300 true\n')
        sys.exit()

    if (l == 'true') or (l == 'True'):
        limbo = True
    elif (l == 'false') or (l == 'False'):
        limbo = False
    else:
        print('Parameter "limbo" should be "true" or "false"\n')
        print('Usage:\n\tpython3 participants_history.py day camera initial_frame num_frames limbo\n')
        print('Example:\n\tpython3 participants_history.py 2 3 700 300 true\n')
        sys.exit()

    paths = read.read_paths()

    participants_history(paths, day, camera, initial_frame, num_frames, limbo)