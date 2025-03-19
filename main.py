from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import os
import pickle


def main():
    # Read Video
    cap = cv2.VideoCapture('input_videos/08fd33_4.mp4')
    if not cap.isOpened():
        raise ValueError("Failed to open video file")

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4V codec
    out = cv2.VideoWriter('output_videos/output_video.mp4', fourcc, fps, (width, height))

    # Read first frame for initialization
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Failed to read first frame")

    # Initialize components
    tracker = Tracker('models/best.pt')
    camera_movement_estimator = CameraMovementEstimator(first_frame)
    view_transformer = ViewTransformer()
    team_assigner = TeamAssigner()
    player_assigner = PlayerBallAssigner()
    speed_and_distance_estimator = SpeedAndDistance_Estimator()

    # Create stubs directory if it doesn't exist
    os.makedirs('stubs', exist_ok=True)
    stub_path = 'stubs/tracking_data.pkl'

    # Initialize tracks structure
    tracks = {
        "players": [],
        "referees": [],
        "ball": []
    }

    # Process first frame
    frame_tracks = tracker.process_frame(first_frame)
    for obj_type in tracks:
        tracks[obj_type].append(frame_tracks[obj_type][0])
    
    # Add positions to tracks
    tracker.add_position_to_tracks(tracks)
    
    # Initialize team colors with first frame
    team_assigner.assign_team_color(first_frame, tracks['players'][0])
    
    # Assign teams to players in first frame
    for track_id, track in tracks['players'][0].items():
        team = team_assigner.get_player_team(first_frame, track['bbox'], track_id)
        track['team'] = team
        track['team_color'] = team_assigner.team_colors[team]

    # Initialize team ball control tracking
    team_ball_control = []
    
    # Process first frame ball control
    if tracks['ball'][0]:  # If ball is detected in first frame
        ball_bbox = tracks['ball'][0][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(tracks['players'][0], ball_bbox)
        if assigned_player != -1:
            team_ball_control.append(tracks['players'][0][assigned_player]['team'])
        else:
            team_ball_control.append(1)  # Default to team 1 if no player has the ball
    else:
        team_ball_control.append(1)  # Default to team 1 if no ball detected

    # Process remaining frames
    frame_num = 1
    print(f"Processing video: {frame_count} frames total")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with tracking
        frame_tracks = tracker.process_frame(frame)
        
        # Update tracks
        for obj_type in tracks:
            tracks[obj_type].append(frame_tracks[obj_type][0])
        
        # Add positions to tracks for current frame
        tracker.add_position_to_tracks(tracks)  # Changed to process all tracks

        # Assign teams to players in current frame
        for track_id, track in tracks['players'][frame_num].items():
            team = team_assigner.get_player_team(frame, track['bbox'], track_id)
            track['team'] = team
            track['team_color'] = team_assigner.team_colors[team]

        # Update team ball control
        if tracks['ball'][frame_num]:  # If ball is detected
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(tracks['players'][frame_num], ball_bbox)
            if assigned_player != -1:
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            else:
                team_ball_control.append(team_ball_control[-1])  # Keep previous team if no player has the ball
        else:
            team_ball_control.append(team_ball_control[-1])  # Keep previous team if no ball detected

        # Save tracking data to stub periodically
        if frame_num % 100 == 0:  # Save every 100 frames
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        # Process frame and write to output
        processed_frame = process_frame(frame, frame_num, tracks, camera_movement_estimator, 
                                     view_transformer, team_assigner, player_assigner, 
                                     speed_and_distance_estimator, tracker, team_ball_control)
        
        out.write(processed_frame)
        frame_num += 1

        # Print progress
        if frame_num % 10 == 0:
            print(f"Processed frame {frame_num}/{frame_count}")

        # Free up memory
        del processed_frame

    # Cleanup
    cap.release()
    out.release()
    print("Video processing complete!")

def process_frame(frame, frame_num, tracks, camera_movement_estimator, view_transformer, 
                 team_assigner, player_assigner, speed_and_distance_estimator, tracker, team_ball_control):
    # Get camera movement for current frame
    camera_movement = camera_movement_estimator.get_camera_movement([frame],
                                                                read_from_stub=False)
    
    # First, process all historical frames up to current frame for position calculations
    frames_to_process = {
        k: v[:frame_num+1] for k, v in tracks.items()
    }
    
    # Add camera movement adjustment to all frames
    camera_movement_estimator.add_adjust_positions_to_tracks(frames_to_process, camera_movement)
    
    # Transform view for all processed frames
    view_transformer.add_transformed_position_to_tracks(frames_to_process)
    
    # Update speed and distance calculations using processed frames
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(frames_to_process)
    
    # Update the main tracks with the processed data
    for obj_type in tracks:
        for i in range(frame_num + 1):
            for track_id in tracks[obj_type][i]:
                if track_id in frames_to_process[obj_type][i]:
                    tracks[obj_type][i][track_id].update(frames_to_process[obj_type][i][track_id])
    
    # Get current frame tracks for display
    current_tracks = {
        k: [v[frame_num]] for k, v in tracks.items()
    }
    
    # Process ball assignment for current frame
    if frame_num in tracks['ball']:
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(tracks['players'][frame_num], ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
    
    # Draw annotations using current frame data
    processed_frame = frame.copy()
    processed_frame = tracker.draw_annotations([processed_frame], 
                                            current_tracks,
                                            np.array(team_ball_control[:frame_num+1]))[0]
    processed_frame = camera_movement_estimator.draw_camera_movement([processed_frame], camera_movement)[0]
    speed_and_distance_estimator.draw_speed_and_distance([processed_frame], current_tracks)
    
    return processed_frame

if __name__ == '__main__':
    main()