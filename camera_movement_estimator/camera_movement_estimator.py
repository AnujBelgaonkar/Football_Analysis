import pickle
import cv2
import numpy as np
import os
import sys 
sys.path.append('../')
from utils import measure_distance,measure_xy_distance

class CameraMovementEstimator():
    def __init__(self,frame):
        self.minimum_distance = 5
        self.previous_frame = None
        self.previous_features = None
        self.cumulative_movement = [0, 0]  # Track total camera movement

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] = 1
        mask_features[:,900:1050] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance =3,
            blockSize = 7,
            mask = mask_features
        )
        
        # Initialize with first frame
        self.previous_frame = first_frame_grayscale
        self.previous_features = cv2.goodFeaturesToTrack(first_frame_grayscale,**self.features)

    def add_adjust_positions_to_tracks(self,tracks, camera_movement):
        # Extract the movement values from the nested list
        movement = camera_movement[0] if isinstance(camera_movement, list) else camera_movement
        
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    position_adjusted = (position[0]-movement[0],
                                       position[1]-movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                movement = pickle.load(f)
                # Update cumulative movement
                if len(movement) > 0:
                    self.cumulative_movement = movement[-1]
                return movement

        frame = frames[0]  # We now process only one frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.previous_features is None or len(self.previous_features) == 0:
            self.previous_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            self.previous_frame = frame_gray
            return [self.cumulative_movement]

        new_features, _, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_frame, 
            frame_gray,
            self.previous_features, 
            None,
            **self.lk_params
        )

        max_distance = 0
        camera_movement_x, camera_movement_y = 0, 0

        if new_features is not None and len(new_features) > 0:
            for i, (new, old) in enumerate(zip(new_features, self.previous_features)):
                new_point = new.ravel()
                old_point = old.ravel()

                distance = measure_distance(new_point, old_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_point, new_point)

        if max_distance > self.minimum_distance:
            # Update cumulative movement
            self.cumulative_movement[0] += camera_movement_x
            self.cumulative_movement[1] += camera_movement_y
            # Update features for next frame
            self.previous_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
        
        self.previous_frame = frame_gray

        return [self.cumulative_movement]

    def draw_camera_movement(self,frames, camera_movement_per_frame):
        output_frames=[]

        # Extract the movement values from the nested list
        movement = camera_movement_per_frame[0] if isinstance(camera_movement_per_frame, list) else camera_movement_per_frame

        for frame_num, frame in enumerate(frames):
            # Draw directly on frame without copying
            cv2.rectangle(frame,(0,0),(500,100),(255,255,255),-1)
            
            x_movement, y_movement = movement  # Use the movement values directly
            cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame)

        return output_frames