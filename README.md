# People_ReID_MatchNMingle
People tracking and re-identification in the MatchNMingle dataset using Multiple Hypothesis Tracking (MHT) and Color Histograms.

How to use:

1. Make sure the file 'paths.txt' has the right indicated addresses.

2. Run the tracker (from terminal):
	python3 tracker.py day camera initial_frame num_frames N_pruning

Where:
'day' and 'camera': set the desired video.
'initial_frame': set the frame of the video to start the tracking.
'num_frames': set how many frames are going to be processed.
'N_pruning': set the index pruning for the MHT algorithm.

For example, the command:
	python3 tracker.py 2 3 700 300 0
would run the tracker with an index pruning of 0, for the video "Day 2 Camera 3", starting from frame 700 and it would finish at frame 999 (300 frames processed).
