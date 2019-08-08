#!/usr/bin/env python3

import sys, csv, numpy as np, os

import read
from reids import reids

import logging
logging.basicConfig(level = logging.INFO, # Messages on terminal
                    format = '%(asctime)s %(message)s',
                    datefmt = '%H:%M:%S')


def main(paths):

    output_path = paths['output_path']
    if not os.path.isdir(output_path):
        logging.info('Output directory does not exist\n')
        sys.exit()
    
    logging.info('Obtaining information about re-identification cases ...\n')

    vi_path = output_path+'videos_info/'
    if not os.path.isdir(vi_path):
        os.mkdir(vi_path)

    num_reids, people_reids = {}, {}

    for day in {1,2,3}:
        for camera in {1,2,3}:
            # Limbo
            re_ids = reids(paths, day, camera, 0, 36000, True)
            num_reids[str(day)+'_'+str(camera)+'_limbo'] = len(re_ids)
            people = set()
            for case in re_ids:
                person = case[0]
                if person not in people:
                    people.add(person)
            people_reids[str(day)+'_'+str(camera)+'_limbo'] = people
            # No limbo
            re_ids = reids(paths, day, camera, 0, 36000, False)
            num_reids[str(day)+'_'+str(camera)] = len(re_ids)
            people = set()
            for case in re_ids:
                person = case[0]
                if person not in people:
                    people.add(person)
            people_reids[str(day)+'_'+str(camera)] = people
    
    write_csv(path=vi_path, num=num_reids, people=people_reids)


def write_csv(path, num, people):
    logging.info('Writing CSVs ...\n')
    head = ['day', 'camera', 'num_reids', 'reid_people', 'num_reid_people']
    rows_limbo = []
    rows = []
    rows_limbo.append(head)
    rows.append(head)
    name_limbo = path+'VideosInfo_limbo.csv'
    name = path+'VideosInfo.csv'

    for day in np.arange(1,4):
        for camera in np.arange(1,4):
            line = [day, camera, num[str(day)+'_'+str(camera)+'_limbo'], str(people[str(day)+'_'+str(camera)+'_limbo']), len(people[str(day)+'_'+str(camera)+'_limbo'])]
            rows_limbo.append(line)

            line = [day, camera, num[str(day)+'_'+str(camera)], str(people[str(day)+'_'+str(camera)]), len(people[str(day)+'_'+str(camera)])]
            rows.append(line)

    with open(name_limbo, 'w') as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        writer.writerows(rows_limbo)
    
    with open(name, 'w') as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        writer.writerows(rows)

    logging.info('Files saved\n')
    

if __name__ == '__main__':
    
    paths = read.read_paths()

    main(paths)