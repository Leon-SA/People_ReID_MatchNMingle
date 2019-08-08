#!/usr/bin/env python3

class Track:
    '''
    Class for each hypothesis
    '''
    def __init__(self, init_track_id, init_detection, init_hist):
        self.track_id = init_track_id # Track ID
        self.last_detection = init_detection # Last detection appended to track
        self.hist_stack = [init_hist] # Stack of data (color histograms)
        self.track_score = 0.001 # Initial track score
        self.frames_count = 0 # Count to control when to update stack
        self.stack_size = 25 # Number of color histograms that can be saved
        self.lost = False # Lost flag
        self.lost_time = 0 # Time of loss
        self.trackers_lost = False # Primary trackers lost flag
        self.frequency = 0.5 # Frequency for stack updating. Every 2 seconds.
    
    def get_track_id(self):
        return self.track_id

    def get_last_detection(self):
        return self.last_detection
    
    def get_track_score(self):
        return self.track_score
    
    def is_lost(self):
        return self.lost

    def are_trackers_lost(self):
        return self.trackers_lost

    def get_hist_stack(self):
        return self.hist_stack
    
    def get_lost_time(self):
        return self.lost_time

    def update(self, detection, hist, score, trackers_lost):
        # Extend hypothesis with a new observation
        if detection is None: # Extended with a dummy observation
            self.lost = True
            self.track_score += 0.001
            self.lost_time += 1/20 # 20 fps
        else:
            self.lost = False
            self.lost_time = 0
            self.frames_count += 1
            self.track_score += score
            self.last_detection = detection
            self.trackers_lost = trackers_lost
            # Stack updating
            if self.frames_count == (20/self.frequency):
                self.hist_stack.append(hist)
                if len(self.hist_stack) > self.stack_size: # Keep a maximum stack size
                    self.hist_stack.pop(0)
                self.frames_count = 0
