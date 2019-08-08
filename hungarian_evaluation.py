#!/usr/bin/env python3

import numpy as np, sys, csv, os, time

import clear_mot
import read
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

    video_name = 'Day '+str(day)+' Camera '+str(camera)
    logging.info(f'Selected video: {video_name}\n')

    filename = hung_path+'Evaluation_day'+str(day)+'_cam'+str(camera)+'_'+str(iou_th)

    ti = time.time() # Start timer
    results, erase = get_results(path=hung_path, day=day, camera=camera, iou_th=iou_th, initial_frame=initial_frame, num_frames=num_frames)
    ground_truth = get_ground_truth(paths=paths, day=day, camera=camera, initial_frame=initial_frame, num_frames=num_frames)
    
    meas = results
    gtruth = ground_truth
    for e_frame in erase:
        del results[e_frame]
        del gtruth[e_frame]

    eval_info = {}
    # Evaluation of the tracker with the dictionaries
    logging.info('Evaluating Hungarian algorithm results ...\n')
    clear = clear_mot.ClearMetrics(gtruth, meas, 10) # Distance threshold for a measurement to be considered a true positive
    clear.match_sequence() # Perform the evaluation
    evaluation = [clear.get_mota(),
                clear.get_motp(),
                clear.get_fn_count(),
                clear.get_fp_count(),
                clear.get_mismatches_count(),
                clear.get_object_count(),
                clear.get_matches_count()]

    logging.info('Evaluation finished\n')
    logging.info(f'Results:\nMOTA: {evaluation[0]}, MOTP: {evaluation[1]}, FN: {evaluation[2]}, FP: {evaluation[3]}\nMismatches: {evaluation[4]}, Objects: {evaluation[5]}, Matches: {evaluation[6]}\n')
    
    # Save results to a .CSV file
    #eval_name = filename+'_'+str(initial_frame)+'-'+str(final_frame)+'.csv'
    #write_csv(file_name=eval_name, results=evaluation)
    eval_info[final_frame] = evaluation

    #########################################
    sub_frames = sorted(range(-1, num_frames, 1000), reverse=True)
    for frame in sub_frames:
        if (frame > 0):
            #results, e = get_results(path=output_path, day=day, camera=camera, iou_th=iou_th, initial_frame=initial_frame, num_frames=frame+1)
            meas = results
            gtruth = ground_truth

            for e_frame in sorted(gtruth.keys(), reverse=True):
                if (e_frame > frame):
                    del gtruth[e_frame]
                    del meas[e_frame]
                else:
                    break

            for e_frame in erase:
                er = e_frame-initial_frame
                if er < max(gtruth.keys()):
                    del meas[er]
                    del gtruth[er]
            
            meas = clean_dictionary(dic=meas)
            gtruth = clean_dictionary(dic=gtruth)

            logging.info('Evaluating Hungarian algorithm results ...\n')
            clear = clear_mot.ClearMetrics(gtruth, meas, 10) # Distance threshold for a measurement to be considered a true positive
            clear.match_sequence() # Perform the evaluation
            tf = time.time() # End timer
            evaluation = [clear.get_mota(),
                        clear.get_motp(),
                        clear.get_fn_count(),
                        clear.get_fp_count(),
                        clear.get_mismatches_count(),
                        clear.get_object_count(),
                        clear.get_matches_count()]

            logging.info('Evaluation finished\n')
            logging.info(f'Results:\nMOTA: {evaluation[0]}, MOTP: {evaluation[1]}, FN: {evaluation[2]}, FP: {evaluation[3]}\nMismatches: {evaluation[4]}, Objects: {evaluation[5]}, Matches: {evaluation[6]}\n')
    
            # Save results to a .CSV file
            #eval_name = filename+'_'+str(initial_frame)+'-'+str(initial_frame+frame)+'.csv'
            #write_csv(file_name=eval_name, results=evaluation)
            eval_info[frame] = evaluation

    tf = time.time() # End timer
    t_tot = tf-ti
    time_file = hung_path+'EvalTime_day'+str(day)+'_cam'+str(camera)+'_'+str(iou_th)+'_'+str(initial_frame)+'-'+str(final_frame)+'.csv'
    np.savetxt(time_file, [t_tot], delimiter=',')
    eval_name = filename+'_'+str(initial_frame)+'-'+str(final_frame)+'.csv'
    write_csv(file_name=eval_name, info=eval_info)


def get_ground_truth(paths, day, camera, initial_frame, num_frames):
    gt_name = paths['output_path']+'ground_truth/GroundTruth_day'+str(day)+'_cam'+str(camera)+'_limbo.csv'
    if not os.path.isfile(gt_name):
        ground_truth(paths=paths, day=day, camera=camera, limbo=True)

    # Read ground truth
    logging.info('Getting ground truth ...\n')
    det_full = read.read_ground_truth(filename=gt_name, initial_frame=initial_frame, num_frames=num_frames)

    frame_print = set(np.arange(-1, num_frames-1, 100)) # To print frame every 100 frames
    gtruth = {}
    for frame in range(num_frames):
        if frame in frame_print:
            logging.info(f'Frame: {initial_frame+frame} ...')

        gtruth[frame] = []
        det = det_full[frame]
                
        people = int(len(det)/4)
        for i in range(people):
            if det[4*i] != -1:
                detection = [det[4*i], det[4*i+1], det[4*i+2], det[4*i+3]]
                gtruth[frame].append(np.array(detection))
            else:
               gtruth[frame].append(None)

    gtruth = clean_dictionary(dic=gtruth)

    return gtruth

def clean_dictionary(dic):
    people = len(dic[0])
    valid = np.zeros([people])
    for lis in dic.values():
        for person in range(people):
            if lis[person] is not None:
                valid[person] += 1

    erase = []
    for person in range(people):
        if valid[person] == 0:
            erase.append(person)
    for e_person in sorted(erase, reverse=True):
        for lis in dic.values():
            del lis[e_person]

    return dic

def get_results(path, day, camera, iou_th, initial_frame, num_frames):
    trk_name = path+'Results_day'+str(day)+'_cam'+str(camera)+'_'+str(iou_th)+'_'+str(initial_frame)+'-'+str(num_frames+initial_frame-1)+'.csv'
    if not os.path.isfile(trk_name):
        logging.info('Results file does not exist\n')
        sys.exit()
    
    # Read results
    logging.info('Getting Hungarian algorithm results ...\n')
    try:
        det_full = np.loadtxt(open(trk_name, 'rb'), delimiter=',')
    except:
        logging.info('Hungarian algorithm results file not found\n')
        sys.exit()

    frame_print = set(np.arange(-1, num_frames-1, 100)) # To print frame every 100 frames
    results = {}
    erase = []
    for frame in range(num_frames):
        if frame in frame_print:
            logging.info(f'Frame: {initial_frame+frame} ...')

        results[frame] = []
        det = det_full[frame]
                
        people = int(len(det)/4)
        erase_frame = True
        for i in range(people):
            if det[4*i] != -1:
                detection = [det[4*i], det[4*i+1], det[4*i+2], det[4*i+3]]
                results[frame].append(np.array(detection))
                erase_frame = False
            else:
               results[frame].append(None)
        if erase_frame:
            erase.append(frame)
    return results, erase

def write_csv(file_name, info):
    logging.info('Writing output CSV ...\n')
    csv_rows = []
    
    line = ['final_frame', 'MOTA', 'MOTP', 'FN', 'FP', 'Mismatches', 'Objects', 'Matches']
    csv_rows.append(line)

    for frame in sorted(info.keys()):
        i = info[frame]
        line = [frame, i[0], i[1], i[2], i[3], i[4], i[5], i[6]]
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
        print('Usage:\n\tpython3 hungarian_evaluation.py day camera initial_frame num_frames iou_th\n')
        print('Example:\n\tpython3 hungarian_evaluation.py 2 3 700 300 25\n')
        sys.exit()

    paths = read.read_paths()

    main(paths, day, camera, initial_frame, num_frames, iou_th)