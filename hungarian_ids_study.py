#!/usr/bin/env python3

import sys, numpy as np, os, csv

import read
from reids import reids
from ground_truth import ground_truth

import logging
logging.basicConfig(level = logging.INFO, # Messages on terminal
                    format = '%(asctime)s %(message)s',
                    datefmt = '%H:%M:%S')

def main(paths, day=2, camera=3, initial_frame=0, num_frames=36000, iou_th=25):
    
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
    
    hung_path = output_path+'hungarian/'
    if not os.path.isdir(hung_path):
        logging.info('Directory with results does not exist\n')
        sys.exit()

    gt_path = output_path+'ground_truth/'
    if not os.path.isdir(gt_path):
        ground_truth(paths=paths, day=day, camera=camera, limbo=True)

    video_name = 'Day '+str(day)+' Camera '+str(camera)
    logging.info(f'Selected video: {video_name}\n')

    # Read Re-ID cases
    logging.info('Getting re-identification cases ...\n')
    re_ids = reids(paths=paths, day=day, camera=camera, initial_frame=initial_frame, num_frames=num_frames, limbo=True)

    # Read ground truth
    logging.info('Getting ground truth ...\n')
    gt_name = gt_path+'GroundTruth_day'+str(day)+'_cam'+str(camera)+'_limbo.csv'
    gtruth_full = read.read_ground_truth(filename=gt_name, initial_frame=initial_frame, num_frames=num_frames)

    # Read results
    r_name = hung_path+'Results_day'+str(day)+'_cam'+str(camera)+'_'+str(iou_th)+'_'+str(initial_frame)+'-'+str(final_frame)+'.csv'
    logging.info('Getting Hungarian algorithm results ...\n')
    try:
        results_full = np.loadtxt(open(r_name, 'rb'), delimiter=',')
    except:
        logging.info('Hungarian algorithm results file not found\n')
        sys.exit()
    '''
    get_id_changes(gtruth, results)
    get_reid_evaluation(re_ids, gtruth, results)
    '''
    info = {}

    id_changes = get_id_changes(gtruth_full, results_full)
    success, fail = get_reid_evaluation(re_ids, gtruth_full, results_full)
    info[final_frame] = [id_changes, fail, success]

    sub_frames = sorted(range(-1, num_frames, 1000), reverse=True)

    for f in sub_frames:
        if f > 0:
            gtruth = gtruth_full[0:f+1,:]
            results = results_full[0:f+1,:]

            re_ids = reids(paths=paths, day=day, camera=camera, initial_frame=initial_frame, num_frames=f+1, limbo=True)            
            success, fail = get_reid_evaluation(re_ids, gtruth, results)

            gtruth = clean_array(array=gtruth)
            results = clean_array(array=results)
            id_changes = get_id_changes(gtruth, results)

            info[f] = [id_changes, fail, success]
    
    id_name = hung_path+'IDInfo_day'+str(day)+'_cam'+str(camera)+'_'+str(iou_th)+'_'+str(initial_frame)+'-'+str(final_frame)+'.csv'
    write_csv(file_name=id_name, info=info)

def write_csv(file_name, info):
    logging.info('Writing output CSV ...\n')
    csv_rows = []
    
    line = ['final_frame','id_changes','reid_cases','successful_reids','failed_reids']
    csv_rows.append(line)

    for frame in sorted(info.keys()):
        i = info[frame]
        line = [frame, i[0], i[1]+i[2], i[2], i[1]]
        csv_rows.append(line)
    
    with open(file_name, 'w') as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        writer.writerows(csv_rows)

    logging.info(f'CSV saved to: {file_name}\n')

def get_id_changes(gtruth, results):
    
    frames = int(gtruth.shape[0])
    if frames != int(results.shape[0]):
        logging.info('Number of frames do not match ...\n')
        sys.exit()

    peop_gt = int(gtruth.shape[1]/4)
    peop_res = int(results.shape[1]/4)

    ids = {}
    for p in range(peop_gt):
        ids[p] = [None]

    for f in range(frames):
        det_gt = gtruth[f]
        for g in range(peop_gt):
            box_gt = (det_gt[4*g], det_gt[4*g+1], det_gt[4*g+2], det_gt[4*g+3])
            if box_gt != (-1,-1,-1,-1):
                det_res = results[f]
                for r in range(peop_res):
                    box_res = (det_res[4*r], det_res[4*r+1], det_res[4*r+2], det_res[4*r+3])
                    if box_res == box_gt:
                        ids[g].append(r)
                        break
            else:
                ids[g].append(ids[g][-1])
    
    full_count = 0
    for line in ids.values():
        count = 0
        for i, j in enumerate(line):
            if j is not None:
                first_index = i
                first_id = j
                break
        for j in line[first_index:]:
            if j != first_id:
                count += 1
                first_id = j
        #print(count)
        full_count += count
    
    return full_count
    #print('full: ', full_count)

def get_reid_evaluation(re_ids, gtruth, results):
    success = 0
    fails = 0
    for case in re_ids:
        i = int(case[0])
        in_frame = int(case[1])
        out_frame = int(case[2])
        det = gtruth[in_frame]
        box_in = (det[4*i], det[4*i+1], det[4*i+2], det[4*i+3])
        det = gtruth[out_frame]
        box_out = (det[4*i], det[4*i+1], det[4*i+2], det[4*i+3])
        det = results[in_frame]
        people = int(len(det)/4)
        for k in range(people):
            box = (det[4*k], det[4*k+1], det[4*k+2], det[4*k+3])
            if box == box_in:
                j = k
                break
        det = results[out_frame]
        box_out_results = (det[4*j], det[4*j+1], det[4*j+2], det[4*j+3])
        if box_out_results == box_out:
            success += 1
        else:
            fails += 1

    return success, fails

def clean_array(array):
    people = int(array.shape[1]/4)
    frames = int(array.shape[0])
    valid = np.zeros([people])
    for f in range(frames):
        d = array[f]
        for p in range(people):
            box = (d[4*p], d[4*p+1], d[4*p+2], d[4*p+3])
            if box != (-1,-1,-1,-1):
                valid[p] += 1

    erase = []
    for person in range(people):
        if valid[person] == 0:
            erase.append(person)
    
    new_array = np.zeros([frames,1])
    for p in sorted(range(people)):
        if p not in erase:
            new_array = np.hstack((new_array, array[:,4*p:4*p+4]))
    new_array = new_array[:,1:]
    return new_array

'''
def get_id_changes(gtruth, results):
    
    frames = int(gtruth.shape[0])
    if frames != int(results.shape[0]):
        logging.info('Number of frames do not match ...\n')
        sys.exit()

    peop_gt = int(gtruth.shape[1]/4)
    peop_res = int(results.shape[1]/4)

    ids = {}
    for p in range(peop_gt):
        ids[p] = [None]

    for f in range(frames):
        det_gt = gtruth[f]
        for g in range(peop_gt):
            box_gt = (det_gt[4*g], det_gt[4*g+1], det_gt[4*g+2], det_gt[4*g+3])
            if box_gt != (-1,-1,-1,-1):
                det_res = results[f]
                for r in range(peop_res):
                    box_res = (det_res[4*r], det_res[4*r+1], det_res[4*r+2], det_res[4*r+3])
                    if box_res == box_gt:
                        ids[g].append(r)
                        break
            else:
                ids[g].append(ids[g][-1])
    
    full_count = 0
    for line in ids.values():
        count = 0
        for i, j in enumerate(line):
            if j is not None:
                first_index = i
                first_id = j
                break
        for j in line[first_index:]:
            if j != first_id:
                count += 1
                first_id = j
        print(count)
        full_count += count
    
    print('full: ', full_count)

def get_reid_evaluation(re_ids, gtruth, results):
    success = 0
    fails = 0
    for case in re_ids:
        i = int(case[0])
        in_frame = int(case[1])
        out_frame = int(case[2])
        det = gtruth[in_frame]
        box_in = (det[4*i], det[4*i+1], det[4*i+2], det[4*i+3])
        det = gtruth[out_frame]
        box_out = (det[4*i], det[4*i+1], det[4*i+2], det[4*i+3])
        det = results[in_frame]
        people = int(len(det)/4)
        for k in range(people):
            box = (det[4*k], det[4*k+1], det[4*k+2], det[4*k+3])
            if box == box_in:
                j = k
                break
        det = results[out_frame]
        box_out_results = (det[4*j], det[4*j+1], det[4*j+2], det[4*j+3])
        if box_out_results == box_out:
            success += 1
        else:
            fails += 1

    print(success, fails)

'''
if __name__ == '__main__':
    
    try:
        day = int(sys.argv[1])
        camera = int(sys.argv[2])
        initial_frame = int(sys.argv[3])
        num_frames = int(sys.argv[4])
        iou_th = int(sys.argv[5])
    except:
        print('Parameters not given correctly\n')
        print('Usage:\n\tpython3 hungarian_ids_study.py day camera initial_frame num_frames iou_th\n')
        print('Example:\n\tpython3 hungarian_ids_study.py 2 3 700 300 25\n')
        sys.exit()

    paths = read.read_paths()

    main(paths, day, camera, initial_frame, num_frames, iou_th)