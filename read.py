#!/usr/bin/env python3

import os, sys, numpy as np

import logging
logging.basicConfig(level = logging.INFO, # Messages on terminal
                    format = '%(asctime)s %(message)s',
                    datefmt = '%H:%M:%S')

def read_paths():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    paths_file = os.path.join(dir_path, "paths.txt")
    keys = ["data_path", "videos_path", "output_path"]
    paths = {}
    with open(paths_file) as f:
        for line in f:
            line_data = line.split("#")[0].split('=')
            if len(line_data) == 2:
                key, val = [s.strip() for s in line_data]
                if key in keys:
                    try:
                        val = str(val)
                    except ValueError:
                        raise AssertionError(f"Incorrect value type in paths.txt: {line}")

                    keys.remove(key)
                    paths[key] = val
            else:
                raise AssertionError(f"Error in paths.txt formatting: {line}")

    if keys:
        raise AssertionError("Parameters not found in paths.txt: " + ", ".join(keys))

    return paths

def read_annotations(path, day, initial_frame, num_frames):
    try:
        det = np.loadtxt(open(path+'DATA.csv', 'rb'), delimiter=',', skiprows=initial_frame, max_rows=num_frames)
        cam = np.loadtxt(open(path+'CAMERA.csv', 'rb'), delimiter=',', skiprows=initial_frame, max_rows=num_frames)
        lost = np.loadtxt(open(path+'LOST.csv', 'rb'), delimiter=',', skiprows=initial_frame, max_rows=num_frames)
    except:
        logging.info('Manual annotations not found\n')
        sys.exit()
    
    # Split data
    if day == 1:
        det = det[:, 0:32*7]
        cam = cam[:, 0:32]
        lost = lost[:, 0:32]
    elif day == 2:
        det = det[:, 32*7:62*7]
        cam = cam[:, 32:62]
        lost = lost[:, 32:62]
    elif day == 3:
        det = det[:, 62*7:92*7]
        cam = cam[:, 62:92]
        lost = lost[:, 62:92]
    else:
        logging.info('Invalid video file\n')
        sys.exit()
    
    return det, cam, lost

def read_ground_truth(filename, initial_frame, num_frames):
    try:
        det = np.loadtxt(open(filename, 'rb'), delimiter=',', skiprows=initial_frame, max_rows=num_frames)
    except:
        logging.info('Ground truth file not found\n')
        sys.exit()
    
    return det
