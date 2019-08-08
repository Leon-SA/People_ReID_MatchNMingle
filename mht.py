#!/usr/bin/env python3

import cv2, numpy as np

from copy import deepcopy

from weighted_graph import WeightedGraph # MWIS algorithm codes
from hypothesis import Track # Class for each hypothesis

import logging
logging.basicConfig(level = logging.INFO, # Messages on terminal
                    format = '%(asctime)s %(message)s',
                    datefmt = '%H:%M:%S')

class MHT:
    def __init__(self, params):
        # Load parameters
        self.N = params['N_pruning']
        self.d_th = params['distance_threshold']
        self.d_th2 = params['distance_threshold2']
        self.trackers_weights = [params['KCF_weight'], params['MF_weight'], params['MIL_weight']]
        self.color_score_th = params['color_score_threshold']
        self.color_score_weight = params['color_score_weight']
        self.lost_time_th = params['lost_time_threshold']
        self.lost_time_weight = params['lost_time_weight']
        self.bins = params['color_hist_bins']
        
        self.track_detections = [] # Track detections over all the frames
        self.tracks = [] # Corresponding objects Track
        self.coordinates = [] # Coordinates for all frame detections
        self.frame_index = 0
        self.traject_count = 0 # Used to set an ID to each object Track
        
    def init(self, frame, detections):
        # Initialization of tracks in the first frame
        self.coordinates.append({})
        for index, detection in detections.items():
            detection_id = str(index)
            self.coordinates[self.frame_index][detection_id] = detection
            box_hist = self.get_color_histogram(frame=frame, box=detection) # Color histogram of th bbox that contains the target
            self.tracks.append(Track(init_track_id=self.traject_count, init_detection=detection, init_hist=box_hist))
            self.traject_count += 1
            self.track_detections.append([''] * self.frame_index + [detection_id])
        self.frame_index += 1

    def run(self, frame, detections, trackers_results):
        self.coordinates.append({})
        track_count = len(self.tracks)

        for index, detection in detections.items():
            detection_id = str(index)
            self.coordinates[self.frame_index][detection_id] = detection
            box_hist = self.get_color_histogram(frame=frame, box=detection) # Color histogram
            
            # Update existing branches
            for i in range(track_count):
                # Copy the track hypothesis
                track_tree = self.tracks[i]
                continued_branch = deepcopy(track_tree)
                if continued_branch.is_lost(): # If hypothesis is a lost track
                    hist_stack = continued_branch.get_hist_stack()
                    lost_time = continued_branch.get_lost_time()
                    candidate, score = self.get_matching_score(new_hist=box_hist, track_stack=hist_stack, lost_time=lost_time) # Compare color histograms between the current target and the lost one
                    if candidate: # It is candidate if color histograms are similar enough
                        # Create new hypothesis (copy + new detection)
                        continued_branch.update(detection=detection, hist=box_hist, score=score, trackers_lost=False)
                        self.tracks.append(continued_branch)
                        self.track_detections.append(self.track_detections[i] + [detection_id])
                else: # Regular (not lost) track
                    track_id = continued_branch.get_track_id()
                    inside, score, trackers_lost = self.get_trackers_score(detection=detection, tracker_results=trackers_results[track_id]) # Get track score based on distances
                    if inside: # Create new hypothesis only if, at least, one of the primary trackers are inside the gating area
                        continued_branch.update(detection=detection, hist=box_hist, score=score, trackers_lost=trackers_lost)
                        self.tracks.append(continued_branch)
                        self.track_detections.append(self.track_detections[i] + [detection_id])
            
            # Create new branch from the detection (new target possibility)
            self.tracks.append(Track(init_track_id=self.traject_count, init_detection=detection, init_hist=box_hist))
            self.traject_count += 1
            self.track_detections.append([''] * self.frame_index + [detection_id])
        
        # Update the track with a dummy detection (lost target possibility)
        for j in range(track_count):
            self.tracks[j].update(detection=None, hist=None, score=None, trackers_lost=None)
            self.track_detections[j].append('')
            
        
        prune_index = max(0, self.frame_index-self.N) # Index for N-scan pruning
        conflicting_tracks = self.get_conflicting_tracks(self.track_detections) # Conflicting tracks: share an observation at any time
        solution_ids = self.get_global_hypothesis(self.tracks, conflicting_tracks) # MWIS
        non_solution_ids = list(set(range(len(self.tracks))) - set(solution_ids))
        prune_ids = set()
        solution_coordinates = [] # List of coordinates for each track
        for solution_id in solution_ids:
            detections = self.track_detections[solution_id]
            track_coordinates = []
            for i in range(len(detections)):
                if detections[i] == '':
                    track_coordinates.append(None)
                else:
                    track_coordinates.append(self.coordinates[i][detections[i]])
            solution_coordinates.append(track_coordinates) # Get the coordinates (bboxes) of the solution
            
            # Identify subtrees that diverge from the solution_trees at frame k-N
            if self.N > 0:
                d_id = self.track_detections[solution_id][prune_index]
                if d_id != '':
                    for non_solution_id in non_solution_ids:
                        if d_id == self.track_detections[non_solution_id][prune_index]:
                            prune_ids.add(non_solution_id)
        
        # Perform pruning
        if self.N == 0:
            prune_ids = non_solution_ids
        for k in sorted(prune_ids, reverse=True):
            del self.track_detections[k]
            del self.tracks[k]

        # Get the ID from each solution hypothesis
        track_ids = []
        for track in self.tracks:
            track_ids.append(track.get_track_id())
        self.traject_count = max(track_ids)+1

        new_tracks = {}
        # Identify tracks of new targets and the ones which have their
        # primary trackers too far from the solutions, so need to be re-initialized
        for i, track in enumerate(self.track_detections):
            if track[len(track)-1] != '' and track[len(track)-2] == '':
                new_id = self.tracks[i].get_track_id()
                new_box = self.tracks[i].get_last_detection()
                new_tracks[new_id] = new_box
            if self.tracks[i].are_trackers_lost():
                new_id = self.tracks[i].get_track_id()
                new_box = self.tracks[i].get_last_detection()
                new_tracks[new_id] = new_box
        
        self.frame_index += 1
        
        return solution_coordinates, track_ids, new_tracks

    def get_matching_score(self, new_hist, track_stack, lost_time):
        # Get score based on distance between color histograms and lost time
        distances = []
        for h in track_stack: # Compare with each histogram in the stack
            d = cv2.compareHist(new_hist, h, cv2.HISTCMP_BHATTACHARYYA)
            distances.append(d)
        mean = np.mean(distances) # Get the average distance
        if mean < self.color_score_th: # The lower, the better
            candidate = True
            const = (np.log(0.01))/self.lost_time_th
            time_score = np.exp(const*lost_time)*self.lost_time_weight # Score based on time
            color_score = (1-mean*(0.99/self.color_score_th))*self.color_score_weight # Score based on similarity
            score = time_score + color_score
        else:
            candidate = False
            score = 0
        return candidate, score
    
    def get_trackers_score(self, detection, tracker_results):
        # Get score based on distances between new detection and primary trackers
        is_inside = []
        distances = []
        is_lost = []
        for box in tracker_results:
            box_center = self.get_center(box=box)
            det_center = self.get_center(box=detection)
            dist = self.get_distance(c1=box_center, c2=det_center)
            if dist < self.d_th:
                is_inside.append(True)
                distances.append(dist)
                if dist >= self.d_th2:
                    is_lost.append(True)
                else:
                    is_lost.append(False)
            else:
                is_inside.append(False)
                distances.append(None)
                is_lost.append(True)
        score = 0
        if any(is_inside): # The new observation is considered to extend a track if one or more primary trackers are inside its gating area
            inside = True
            for i, distance in enumerate(distances):
                if distance is not None:
                    score += (1/self.d_th**2)*((distance-self.d_th)**2)*self.trackers_weights[i] # y=(1/th^2)*(x-th)^2
        else:
            inside = False
        if all(is_lost): # If all primary trackers exceed the threshold, they are lost
            trackers_lost = True
        else:
            trackers_lost = False
        return inside, score, trackers_lost

    def get_center(self, box):
        # Get bboxes' centers
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        
        xc = (x1+x2)/2
        yc = (y1+y2)/2
        center = (xc, yc)
        return center

    def get_distance(self, c1, c2):
        # Euclidean distance
        distance = np.sqrt( np.power(c1[1]-c2[1], 2) + np.power(c1[0]-c2[0], 2))
        return distance

    def get_color_histogram(self, frame, box):
        # Compute color histograms
        bins = self.bins
        img = frame.copy()
        img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])] # Get section of image bordered by bbox
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([img], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def get_conflicting_tracks(self, track_detections):
        # Identify conflicting tracks
        conflicting_tracks = []
        for i in range(len(track_detections)):
            for j in range(i + 1, len(track_detections)):
                left_ids = track_detections[i]
                right_ids = track_detections[j]
                for k in range(len(left_ids)):
                    if left_ids[k] != '' and right_ids[k] != '' and left_ids[k] == right_ids[k]:
                        conflicting_tracks.append((i, j))

        return conflicting_tracks
    
    def get_global_hypothesis(self, tracks, conflicting_tracks):
        """
        Generate a global hypothesis by finding the maximum weighted independent
        set of a graph with tracks as vertices, and edges between conflicting tracks.
        """
        gh_graph = WeightedGraph()
        for index, track in enumerate(tracks):
            s = track.get_track_score()
            gh_graph.add_weighted_vertex(str(index), s)

        gh_graph.set_edges(conflicting_tracks)

        mwis_ids = gh_graph.mwis()
        
        return mwis_ids