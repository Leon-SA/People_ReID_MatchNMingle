#!/usr/bin/env python3

import sys, numpy as np, os

import read
from participants_history import participants_history

import logging
logging.basicConfig(level = logging.INFO, # Messages on terminal
                    format = '%(asctime)s %(message)s',
                    datefmt = '%H:%M:%S')


def reids(paths, day=2, camera=3, initial_frame=0, num_frames=36000, limbo=True):

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
    
    ph_path = output_path+'participants_history/'
    if not os.path.isdir(ph_path):
        os.mkdir(ph_path)

    if limbo:
        l = '_limbo.csv'
    else:
        l = '.csv'

    ph_name = ph_path+'ParticipantsHistory_day'+str(day)+'_cam'+str(camera)+'_'+str(initial_frame)+'-'+str(final_frame)+l
    
    if not os.path.isfile(ph_name):
        participants_history(paths=paths, day=day, camera=camera, initial_frame=initial_frame, num_frames=num_frames, limbo=limbo)

    logging.info('Getting participants history ...\n')
    try:
        hist = np.loadtxt(open(ph_name, 'rb'), delimiter=',')
    except:
        logging.info('Participants history file not found\n')
        sys.exit()

    lines = hist.shape[0]
    was = {}
    re_ids = []
    for line in range(lines):
        h = hist[line]
        person = h[0]
        if person not in was.keys():
            was[person] = h[2]
        else:
            re_ids.append((person, h[1], was[person]))
            was[person] = h[2]
    
    #print(len(re_ids))
    #print(re_ids)
    return re_ids


if __name__ == '__main__':
    
    try:
        day = int(sys.argv[1])
        camera = int(sys.argv[2])
        initial_frame = int(sys.argv[3])
        num_frames = int(sys.argv[4])
        l = sys.argv[5]
    except:
        print('Parameters not given correctly\n')
        print('Usage:\n\tpython3 reids.py day camera initial_frame num_frames limbo\n')
        print('Example:\n\tpython3 reids.py 2 3 700 300 true\n')
        sys.exit()

    if (l == 'true') or (l == 'True'):
        limbo = True
    elif (l == 'false') or (l == 'False'):
        limbo = False
    else:
        print('Parameter "limbo" should be "true" or "false"\n')
        print('Usage:\n\tpython3 reids.py day camera initial_frame num_frames limbo\n')
        print('Example:\n\tpython3 reids.py 2 3 700 300 true\n')
        sys.exit()

    paths = read.read_paths()

    reids(paths, day, camera, initial_frame, num_frames, limbo)