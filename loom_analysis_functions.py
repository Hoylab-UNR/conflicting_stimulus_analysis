import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import gaussian_filter1d
import ast
from matplotlib.collections import LineCollection

#All parameters should be updated to be defined the same way as in Thiago Branco's code

#WIP :Also I should convert all points to cm at the very beginning and time to seconds (make a decision if I need to resample), will make life easier, in addition I need to correct for the perspective distortion Tiago
#Brancos current bio behvaior paper has the correct approach. Gaussian filtering needs to be used consistently and with purpose, not just because it looks good, do this filtering at the beginning.
#All functions should be written with the assumption that the data is accurate and of good quality. Data should be preprocessed before being passed to these functions.
#Check that for all videos origin is the same, if not, make it the same.
#For every trial compute the following data, these 
    #Continuous variables (WIP: Need to filter this by likelihood at some point)
        # [x] Distance between cricket and mouse
        # [x] Distance between mouse and shelter
        # [x] Mouse speed
        # [x] Kmean vector of mouse heading
#Categorical variables (can all be computed from above)
    # [x] Cricket location in visual field - during 1st loom frame
    # [x] Max speed of mouse
    # [x] Escape to shelter
    # [ ] Escape latency (time to speed > SOME THRESHOLD)

def check_loom_true_positive(dlc_df,
                            mouse_centpos_trial,
                            cricket_pos_trial,
                            loom_start,
                            pix2cm_scale = 1,
                            approach_distance_threshold = 15,
                            approach_binocular_threshold = 45,
                            min_tracking_frames = 0,
                            min_tracking_percentage = 0,
                            verbose = False):
    """
    Check if mouse is close to and tracking cricket during loom stimulus onset.

    Parameters:
    -----------
    dlc_df : DataFrame
        DeepLabCut tracking data
    mouse_centpos_trial : array
        Mouse center position data
    cricket_pos_trial : array
        Cricket position data
    loom_start : int
        Frame number where loom starts
    pix2cm_scale : float
        Scale to convert pixels to cm
    approach_distance_threshold : float
        Maximum distance (in cm) for considering mouse close to cricket
    approach_binocular_threshold : float
        Maximum angle difference (in degrees) for considering mouse tracking cricket
    min_tracking_frames : int
        Minimum number of consecutive frames required for tracking
    min_tracking_percentage : float
        Minimum percentage of frames in window that must show tracking
    verbose : bool
        If True, print detailed information about why a false positive was detected
        
    Returns:
    --------
    bool
        True if mouse is tracking cricket, False otherwise
    """

    if pix2cm_scale == 1:
        print("WARNING: pix2cm scale not provided, using default value of 1")
    
    # Clamp window to valid frame indices to avoid out-of-bounds errors
    trial_length = len(mouse_centpos_trial)
    window_start = max(0, loom_start - 20)
    window_end = min(trial_length, loom_start + 20)
    
    # Check if window is valid
    if window_start >= window_end:
        if verbose:
            print(f"False positive: Invalid window (start={window_start}, end={window_end}, trial_length={trial_length})")
        return False
    
    # Check if mouse is close to cricket in a 20 frame window around loom start - can change to 20
    window_frames = range(window_start, window_end)
    window_frames_list = list(window_frames)
    
    # Ensure we have valid indices for indexing
    if len(window_frames_list) == 0:
        if verbose:
            print(f"False positive: Empty window frames")
        return False
    
    distances = mouse_dynamic_object_distance(mouse_centpos_trial[window_frames_list], cricket_pos_trial[window_frames_list])
    
    # Check for NaN values in distances
    if np.all(np.isnan(distances)):
        if verbose:
            print(f"False positive: All distances are NaN")
        return False
    
    if not any((distances / pix2cm_scale) < approach_distance_threshold):
        if verbose:
            print(f"False positive: Mouse was not within {approach_distance_threshold} cm of cricket during loom window")
            print(f"Minimum distance observed: {np.nanmin(distances / pix2cm_scale):.2f} cm")
        return False
        
    # Compute cricket azimuth and mouse heading for all frames in window
    cricket_azimuths = []
    mouse_headings = []
    valid_frames = []
    
    for i in window_frames_list:
        # Check if frame index is valid for dlc_df
        try:
            mouse_heading, cricket_azimuth = get_mouse_heading(dlc_df,
                                                            frame_num = i,
                                                            return_cricket_azimuth = True)
            # Check for NaN values
            if not (np.isnan(mouse_heading) or np.isnan(cricket_azimuth)):
                cricket_azimuths.append(cricket_azimuth)
                mouse_headings.append(mouse_heading)
                valid_frames.append(True)
            else:
                valid_frames.append(False)
        except (IndexError, KeyError) as e:
            if verbose:
                print(f"Warning: Frame {i} not available in dlc_df: {e}")
            valid_frames.append(False)
    
    # Check if we have any valid frames
    if len(cricket_azimuths) == 0:
        if verbose:
            print(f"False positive: No valid heading data in window")
        return False
            
    # Convert to numpy arrays for easier computation
    cricket_azimuths = np.array(cricket_azimuths)
    mouse_headings = np.array(mouse_headings)
    
    # Calculate angle differences
    angle_diffs  = ((mouse_headings - cricket_azimuths + np.pi) % (2*np.pi)) - np.pi

    tracking_frames = angle_diffs < np.deg2rad(approach_binocular_threshold) #mouse is tracking cricket
    
    # Check if we have enough consecutive tracking frames
    if np.sum(tracking_frames) > 0:
        tracking_runs = np.split(tracking_frames, np.where(np.diff(tracking_frames))[0] + 1)
        if len(tracking_runs) > 0:
            # Filter runs where run[0] is True and get max length
            true_runs = [len(run) for run in tracking_runs if len(run) > 0 and run[0]]
            if len(true_runs) > 0:
                max_consecutive = max(true_runs)
            else:
                max_consecutive = 0  # no tracking runs with True values
        else:
            max_consecutive = 0 #no tracking runs
    else:
        max_consecutive = 0 #no tracking frames
    
    # Check if we have enough total tracking frames
    total_tracking_frames = np.sum(tracking_frames)
    # Use actual number of valid frames (not the original window size)
    n_valid_frames = len(cricket_azimuths)
    if n_valid_frames > 0:
        tracking_percentage = total_tracking_frames / n_valid_frames
    else:
        tracking_percentage = 0.0
    
    if verbose and not (max_consecutive >= min_tracking_frames and tracking_percentage >= min_tracking_percentage):
        print(f"False positive: Tracking criteria not met")
        print(f"Maximum consecutive tracking frames: {max_consecutive} (minimum required: {min_tracking_frames})")
        print(f"Tracking percentage: {tracking_percentage:.2%} (minimum required: {min_tracking_percentage:.2%})")
    
    # Return True only if both conditions are met
    if not verbose:
        return (max_consecutive >= min_tracking_frames and 
                tracking_percentage >= min_tracking_percentage)
    else:
        return (max_consecutive >= min_tracking_frames and 
                tracking_percentage >= min_tracking_percentage), mouse_headings, cricket_azimuths
        

def get_pix2cm_scale(arena_kpts, seperate_xy = False):
    '''
    Get the pix2cm scale for the arena
    If seperate_xy is True, return the pix2cm scale for the x and y directions
    Have tested this and for now x and y scales are less than 2.5% different so we can use the same scale for both
    '''   
    if seperate_xy:
        distance_x = []
        distance_y = []
        distance_x.append(np.sqrt(np.sum((arena_kpts[1] - arena_kpts[0])**2))) # Distance between points 1-2
        distance_y.append(np.sqrt(np.sum((arena_kpts[2] - arena_kpts[1])**2))) # Distance between points 2-3  
        distance_x.append(np.sqrt(np.sum((arena_kpts[3] - arena_kpts[2])**2))) # Distance between points 3-4
        distance_y.append(np.sqrt(np.sum((arena_kpts[0] - arena_kpts[3])**2))) # Distance between points 4-1
        pix2cm_xscale = np.mean(distance_x, axis=0)/60
        pix2cm_yscale = np.mean(distance_y, axis=0)/60
        return pix2cm_xscale, pix2cm_yscale
    else:
        distances = []
        distances.append(np.sqrt(np.sum((arena_kpts[1] - arena_kpts[0])**2))) # Distance between points 1-2
        distances.append(np.sqrt(np.sum((arena_kpts[2] - arena_kpts[1])**2))) # Distance between points 2-3  
        distances.append(np.sqrt(np.sum((arena_kpts[3] - arena_kpts[2])**2))) # Distance between points 3-4
        distances.append(np.sqrt(np.sum((arena_kpts[0] - arena_kpts[3])**2))) # Distance between points 4-1
        pix2cm_scale = np.mean(distances, axis=0)/60
        return pix2cm_scale

def match_videos_to_labels(looming_video, dlc_labels):
    if '.mp4' not in looming_video:
        filename = os.path.basename(looming_video).split('.avi')[0]
    else:
        filename = os.path.basename(looming_video).split('.mp4')[0]
    matching_labels = [dlc_label for dlc_label in dlc_labels if filename in dlc_label]
    assert len(matching_labels) > 0, f"No matching DLC label found for filename: {filename}"
    dlc_label = matching_labels[0]
    return dlc_label

def correct_trial_timestamps(loom_start, loom_end, pad_onset, pad_offset, dlc_df):
    loom_duration = loom_end - loom_start
    trial_start = loom_start-pad_onset
    if trial_start < 0:
        trial_start = 0
        pad_onset = loom_start

    trial_end = loom_start+loom_duration+pad_offset

    if trial_end > len(dlc_df):
        trial_end = len(dlc_df)
        if loom_start + loom_duration > trial_end:
            pad_offset = 0
        else:
            pad_offset = trial_end - loom_end
    trial_duration = trial_end - trial_start

    return trial_start, trial_end, pad_onset, pad_offset, trial_duration

def get_arena_shelter_loom_labels(looming_video, arena_labels):
    arena_labels_df = pd.read_csv(arena_labels[0])
    filename= os.path.basename(looming_video)
    #find the row where the video file matches the filename
    row = arena_labels_df[arena_labels_df['video_file'] == filename]
    # Get the row where the video file matches the filename
    if row.empty:
        raise ValueError(f"File '{filename}' not found in arena labels dataframe")

    #assert that there is only one row
    assert len(row) == 1, f'Multiple {filename} found in arena labels dataframe'
    
    #Get the shelter kpts and arena kpts
    shelter_kpts = [(row.iloc[0].shelter_point1_x, row.iloc[0].shelter_point1_y), (row.iloc[0].shelter_point2_x, row.iloc[0].shelter_point2_y), (row.iloc[0].shelter_point3_x, row.iloc[0].shelter_point3_y), (row.iloc[0].shelter_point4_x, row.iloc[0].shelter_point4_y)]
    arena_kpts = [(row.iloc[0].arena_point1_x, row.iloc[0].arena_point1_y), (row.iloc[0].arena_point2_x, row.iloc[0].arena_point2_y), (row.iloc[0].arena_point3_x, row.iloc[0].arena_point3_y), (row.iloc[0].arena_point4_x, row.iloc[0].arena_point4_y)]
    loom_center = [(row.iloc[0].looming_point1_x, row.iloc[0].looming_point1_y), (row.iloc[0].looming_point2_x, row.iloc[0].looming_point2_y), (row.iloc[0].looming_point3_x, row.iloc[0].looming_point3_y), (row.iloc[0].looming_point4_x, row.iloc[0].looming_point4_y)]

    return np.array(arena_kpts), np.array(shelter_kpts), np.array(loom_center), row

def show_frame(fr, video_path, ax = None):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
    ret, frame = cap.read()
    if ax is None:
        plt.imshow(frame)
    else:
        ax.imshow(frame)

def get_mouse_and_cricket_positions(dlc_df,
                                    start_idx,
                                    end_idx,
                                    likelihood_threshold = 0.5): #nan filter should be remove likelihood filter
    dlc_array = np.array(dlc_df)
    if start_idx < 0:
        start_idx = 0
    if end_idx > len(dlc_array):
        end_idx = len(dlc_array)
    #calculate the mean of the x and y positions
    mouse_cent_xpos = np.mean(dlc_array[start_idx:end_idx, 0:12:3], axis=1)
    mouse_cent_ypos = np.mean(dlc_array[start_idx:end_idx, 1:12:3], axis=1)
    mouse_cent_likelihood = np.mean(dlc_array[start_idx:end_idx, 2:12:3], axis=1)
    mouse_cent_xpos[mouse_cent_likelihood < likelihood_threshold] = np.nan
    mouse_cent_ypos[mouse_cent_likelihood < likelihood_threshold] = np.nan

    cricket_xpos = dlc_array[start_idx:end_idx, 12]
    cricket_ypos = dlc_array[start_idx:end_idx, 13]
    cricket_likelihood = dlc_array[start_idx:end_idx, 14]
    cricket_xpos[cricket_likelihood < likelihood_threshold] = np.nan
    cricket_ypos[cricket_likelihood < likelihood_threshold] = np.nan

    return np.array([mouse_cent_xpos, mouse_cent_ypos]).T, np.array([cricket_xpos, cricket_ypos]).T

#   -------------Functions below are from Mooshik - WIP: Need to think which functions to keep here or in mooshik

def visualize_heading(image,
                    dlc_out,
                    frame_num,
                    heading,
                    arrow_len=30,
                    arrow_color = 'red',
                    axis=None):
    ''' WIP: This should call estimate heading function in analyze.py
    Visualizes the heading of the mouse

    Parameters:
    - image: The image to visualize the heading on.
    - dlc_out: A dictionary containing the output of DeepLabCut (DLC) analysis.
    - frame_num: The frame number to visualize the heading for.
    - heading: The heading angle in radians.
    - arrow_len: The length of the arrow representing the heading (default: 30).
    - axis: The axis object to plot the image and arrow on (default: None).

    Returns:
    None
    '''
    # drawing an arrow along theta of arrow_len
    xviz, yviz = pol2cart(arrow_len, heading)
    
    # mean of left and right ear keypoints to get midpoint
    x_midear = np.mean([dlc_out['leftear']['x'][frame_num], dlc_out['rightear']['x'][frame_num]])
    y_midear = np.mean([dlc_out['leftear']['y'][frame_num], dlc_out['rightear']['y'][frame_num]])

    if axis is None:
        plt.imshow(image)
        plt.arrow(x_midear, y_midear, xviz, yviz,
                  length_includes_head=True,
                  head_width=5, head_length=10, color=arrow_color )
    else:
        axis.imshow(image)
        axis.arrow(x_midear, y_midear, xviz, yviz,
                   length_includes_head=True,
                   head_width=5, head_length=10, color=arrow_color)
        
def get_mouse_heading(dlc_out,
                    frame_num,
                    return_cricket_azimuth = False,
                    object_azimuth = None):
    """
    Get the heading of the mouse for a given frame or range of frames.

    Parameters:
    -----------
    dlc_out : dict
        Dictionary containing DeepLabCut tracking data
    frame_num : int or tuple
        Frame number to analyze. If tuple (start, end), returns list of headings for frames in range
    return_cricket_azimuth : bool
        If True, also return cricket azimuth
    object_azimuth : array-like or None
        Optional object position to calculate azimuth to
        
    Returns:
    --------
    float or list
        Mouse heading(s) in radians. If frame_num is tuple, returns list of headings.
        If return_cricket_azimuth is True, returns tuple of (heading, cricket_azimuth)
    """
    def _get_single_frame_heading(frame):
        #mean of left and right ear keypoints to get midpoint
        x_midear = np.mean([dlc_out['leftear']['x'][frame], dlc_out['rightear']['x'][frame]])
        y_midear = np.mean([dlc_out['leftear']['y'][frame], dlc_out['rightear']['y'][frame]])

        #translating heading based on polar coordinates centered on midear
        try:
            dx, dy = dlc_out['nose']['x'][frame]- x_midear, dlc_out['nose']['y'][frame]-y_midear
        except KeyError:
            dx, dy = dlc_out['nosepoint']['x'][frame]- x_midear, dlc_out['nosepoint']['y'][frame]-y_midear

        _, heading = cart2pol(dx, dy)
        
        computed_object_azimuth = None
        if object_azimuth is not None:
            if isinstance(object_azimuth, list):
                computed_object_azimuth = []
                for obj in object_azimuth:
                    dx, dy = obj[0] - x_midear, obj[1] - y_midear
                    _, az = cart2pol(dx, dy)
                    computed_object_azimuth.append(az)
            elif isinstance(object_azimuth, np.ndarray):
                dx, dy = object_azimuth[0] - x_midear, object_azimuth[1] - y_midear
                _, computed_object_azimuth = cart2pol(dx, dy)

        if return_cricket_azimuth:
            try:
                dx, dy = dlc_out['stim']['x'][frame]- x_midear, dlc_out['stim']['y'][frame]-y_midear
            except KeyError:
                dx, dy = dlc_out['stimulus']['x'][frame]- x_midear, dlc_out['stimulus']['y'][frame]-y_midear
            _, cricket_azimuth = cart2pol(dx, dy)
            if computed_object_azimuth is not None:
                return heading, cricket_azimuth, computed_object_azimuth
            else:
                return heading, cricket_azimuth
        else:
            if computed_object_azimuth is not None:
                return heading, computed_object_azimuth
            else:
                return heading

    # Handle frame_num as tuple (range) or int
    if isinstance(frame_num, tuple):
        start_frame, end_frame = frame_num
        return [_get_single_frame_heading(frame) for frame in range(start_frame, end_frame)]
    else:
        return _get_single_frame_heading(frame_num)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

# ------------- End of Mooshik functions -------------

def get_loom_point(arena_kpts, pix2cm_scale = 1, loom_point_distance = 15):
    raise NotImplementedError("This function is not implemented yet")
    #WIP I have only used arena kpts so far, need to add loom screen kpts

    #find a point 15 units away from arena kpts 0 and 4 perpendicular to the line joining them
    normal_vector = arena_kpts[4] - arena_kpts[0]
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    loom_point = arena_kpts[0] + loom_point_distance/pix2cm_scale * normal_vector
    return loom_point

def get_mouse_loom_position(dlc_df, frame_num, loom_center, screen_height=90, pix2cm_scale = 1):
    if pix2cm_scale == 1:
        print("WARNING: pix2cm scale not provided, using default value of 1")
    #mean of left and right ear keypoints to get midpoint
    x_midear = np.mean([dlc_df['leftear']['x'][frame_num], dlc_df['rightear']['x'][frame_num]])
    y_midear = np.mean([dlc_df['leftear']['y'][frame_num], dlc_df['rightear']['y'][frame_num]])

    #compute loom stim azimuth and distance
    heading, loom_azimuth = get_mouse_heading(dlc_df, frame_num, object_azimuth = loom_center)
    loom_elevation = np.arctan2(screen_height, np.sqrt((loom_center[0] - x_midear)**2 + (loom_center[1] - y_midear)**2)/pix2cm_scale)
    return loom_azimuth, loom_elevation

def mouse_dynamic_object_distance(mouse_centpos, dynamic_object_location):
    return np.sqrt((mouse_centpos[:,0]-dynamic_object_location[:,0])**2 + (mouse_centpos[:,1]-dynamic_object_location[:,1])**2)

def mouse_static_object_distance(mouse_centpos, static_object_location):
    return np.sqrt((mouse_centpos[:,0]-static_object_location[0])**2 + (mouse_centpos[:,1]-static_object_location[1])**2)

def mouse_speed(mouse_centpos): #add timestamps for each frame
    displacement = np.diff(mouse_centpos, axis=0)
    speed = np.sqrt(np.sum(displacement**2, axis=1))
    return np.concatenate(([np.nan], speed))
    
def escape_latency(mouse_speed_loom, escape_threshold = 5):
    indices = np.where(mouse_speed_loom > escape_threshold)[0]
    if len(indices) == 0:
        return np.nan
    return indices[0]

def detect_freeze(
    distance: np.ndarray,
    speed: np.ndarray,
    fps: float = 30.0,  # frames per second #remove this and use timestamps
    freeze_speed_threshold: float = 2.5,  # cm/s for at least 0.5 seconds or 15 frames
    shelter_distance_threshold: float = 5.0,  # cm
    min_freeze_duration_frames: int = 15,  # minimum frames for a valid freeze
    smoothing_sigma: float = 1.0,  # sigma for Gaussian smoothing
    debug: bool = False
) -> Dict:
    """
    Detect freezing behavior in mouse speed data.

    Parameters:
    -----------
    distance: np.ndarray
        Distance to shelter
    speed: np.ndarray
        Speed of the mouse in cm/s
    fps: float
        Frame rate (frames per second)
    freeze_speed_threshold: float
        Speed below which the mouse is considered frozen (cm/s)
    shelter_distance_threshold: float
        Distance below which the mouse is considered to have reached shelter
    min_freeze_duration_frames: int
        Minimum number of frames for a freeze to be considered valid
    smoothing_sigma: float
        Sigma value for Gaussian smoothing of speed data
    debug: bool
        If True, return full result dict in addition to freeze/duration

    Returns:
    --------
    tuple: (freeze: bool, duration: float)
        If debug=True: (freeze: bool, duration: float, result: Dict)
    """
    # Ensure inputs are numpy arrays
    distance = np.asarray(distance)
    speed = np.asarray(speed)
    
    # Check that arrays have the same length
    if len(distance) != len(speed):
        raise ValueError("Distance and speed arrays must have the same length")
    
    # Smooth the speed signal after applying nan filter
    smooth_speed = np.copy(speed)
    valid_indices = ~np.isnan(smooth_speed)
    
    if np.any(valid_indices):
        smooth_speed[valid_indices] = gaussian_filter1d(smooth_speed[valid_indices], sigma=smoothing_sigma)
    
    # Find all potential freeze periods (speed below threshold)
    freeze_mask = smooth_speed < freeze_speed_threshold
    freeze_periods = []
    
    # Group consecutive freeze frames into periods
    i = 0
    while i < len(freeze_mask):
        if freeze_mask[i]:
            start_idx = i
            while i < len(freeze_mask) and freeze_mask[i]:
                i += 1
            end_idx = i - 1
            
            # Calculate freeze duration in frames
            duration_frames = end_idx - start_idx + 1
            
            # Only include freezes longer than minimum duration
            if duration_frames >= min_freeze_duration_frames:
                duration_seconds = duration_frames / fps
                freeze_periods.append({
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_time": start_idx / fps,
                    "end_time": end_idx / fps,
                    "duration_frames": duration_frames,
                    "duration": duration_seconds
                })
        else:
            i += 1
    
    
    # Prepare the result dictionary
    result = {
        "freeze_start_frame": 0,
        "freeze_end_frame": 0,
        "freeze_start_time": 0,
        "freeze_end_time": 0,
        "freeze_duration": 0,
        "all_freezes": freeze_periods,
        "freeze_detected": bool(freeze_periods)
    }
    
    # Find the longest freeze period
    if freeze_periods:
        longest_freeze = max(freeze_periods, key=lambda x: x["duration_frames"])
        result.update({
            "freeze_start_frame": longest_freeze["start_idx"],
            "freeze_end_frame": longest_freeze["end_idx"],
            "freeze_start_time": longest_freeze["start_time"],
            "freeze_end_time": longest_freeze["end_time"],
            "freeze_duration": longest_freeze["duration"]
        })
        freeze = True
        duration = longest_freeze["duration"]
    else:
        freeze = False
        duration = np.nan
        result.update({
            "freeze_start_frame": 0,
            "freeze_end_frame": 0,
            "freeze_start_time": 0,
            "freeze_end_time": 0,
            "freeze_duration": 0
        })
    if debug:
        return freeze, duration, result
    else:
        return freeze, duration

def analyze_mouse_escape(
    distance: np.ndarray,
    speed: np.ndarray,
    fps: float = 30.0,  # frames per second #remove this and use timestamps
    freeze_speed_threshold: float = 2.5,  # cm/s for at least 0.5 seconds or 15 frames
    escape_speed_threshold: float = 10.0,  # cm/s for determining direct escape
    shelter_distance_threshold: float = 5.0,  # cm
    min_freeze_duration_frames: int = 3,  # minimum frames for a valid freeze
    min_escape_duration_frames: int = 3,  # minimum frames for a valid escape
    smoothing_sigma: float = 1.0  # sigma for Gaussian smoothing
) -> Dict:
    """
    Analyze mouse freezing and escape behavior with fixed sampling rate.
    
    Parameters:
    -----------
    distance: np.ndarray
        Distance to shelter in cm
    speed: np.ndarray
        Speed of the mouse in cm/s
    fps: float
        Frame rate (frames per second)
    freeze_speed_threshold: float
        Speed below which the mouse is considered frozen (cm/s)
    escape_speed_threshold: float
        Speed threshold to determine start of escape when no freeze (cm/s)
    shelter_distance_threshold: float
        Distance below which the mouse is considered to have reached shelter (cm)
    min_freeze_duration_frames: int
        Minimum number of frames for a freeze to be considered valid
    min_escape_duration_frames: int
        Minimum number of frames for an escape segment to be considered valid
    smoothing_sigma: float
        Sigma value for Gaussian smoothing of speed data
        
    Returns:
    --------
    Dict with the following keys:
        - freeze_start_frame: Frame index when the longest freeze starts
        - freeze_end_frame: Frame index when the longest freeze ends
        - freeze_start_time: Time (in seconds) when the longest freeze starts
        - freeze_end_time: Time (in seconds) when the longest freeze ends
        - freeze_duration: Duration (in seconds) of the longest freeze
        - escape_avg_speed: Average speed during the longest contiguous escape
        - escape_max_speed: Maximum speed during the longest contiguous escape
        - all_freezes: List of all freeze periods (start, end, duration)
        - all_escapes: List of all escape periods (start, end, avg_speed, max_speed)
    """
    # Ensure inputs are numpy arrays
    distance = np.asarray(distance)
    speed = np.asarray(speed)
    
    # Check that arrays have the same length
    if len(distance) != len(speed):
        raise ValueError("Distance and speed arrays must have the same length")
    
    # Find where mouse reaches shelter
    shelter_indices = np.where(distance <= shelter_distance_threshold)[0]
    
    if len(shelter_indices) == 0:
        raise ValueError("Mouse never reaches shelter in this trial")
    
    # Get the first frame where mouse reaches shelter
    shelter_frame = shelter_indices[0]
    
    # Trim data to only include frames until mouse reaches shelter
    distance = distance[:shelter_frame+1]
    speed = speed[:shelter_frame+1]
    
    # Smooth the speed signal after applying nan filter
    smooth_speed = np.copy(speed)
    valid_indices = ~np.isnan(smooth_speed)
    
    if np.any(valid_indices):
        smooth_speed[valid_indices] = gaussian_filter1d(smooth_speed[valid_indices], sigma=smoothing_sigma)
    
    # Find all potential freeze periods (speed below threshold)
    freeze_mask = smooth_speed < freeze_speed_threshold
    freeze_periods = []
    
    # Group consecutive freeze frames into periods
    i = 0
    while i < len(freeze_mask):
        if freeze_mask[i]:
            start_idx = i
            while i < len(freeze_mask) and freeze_mask[i]:
                i += 1
            end_idx = i - 1
            
            # Calculate freeze duration in frames
            duration_frames = end_idx - start_idx + 1
            
            # Only include freezes longer than minimum duration
            if duration_frames >= min_freeze_duration_frames:
                duration_seconds = duration_frames / fps
                freeze_periods.append({
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_time": start_idx / fps,
                    "end_time": end_idx / fps,
                    "duration_frames": duration_frames,
                    "duration": duration_seconds
                })
        else:
            i += 1
    
    # Find all potential escape periods (after freezes until shelter)
    escape_periods = []
    
    # Case 1: Mouse freezes at least once
    if freeze_periods:
        for freeze in freeze_periods:
            end_idx = freeze["end_idx"]
            escape_start_idx = end_idx + 1
            
            if escape_start_idx >= len(distance):
                continue #escape starts at the end - no escape
                
            # Escape ends at shelter
            escape_end_idx = len(distance) - 1
            
            # Calculate escape duration in frames
            escape_duration_frames = escape_end_idx - escape_start_idx + 1
            
            # Filter escapes shorter than minimum duration
            if escape_duration_frames >= min_escape_duration_frames:
                escape_duration_seconds = escape_duration_frames / fps
                escape_speed_segment = smooth_speed[escape_start_idx:escape_end_idx+1]
                
                # Filter out NaN values for calculations
                valid_speed = escape_speed_segment[~np.isnan(escape_speed_segment)]
                
                if len(valid_speed) > 0:
                    escape_periods.append({
                        "start_idx": escape_start_idx,
                        "end_idx": escape_end_idx,
                        "start_time": escape_start_idx / fps,
                        "end_time": escape_end_idx / fps,
                        "duration_frames": escape_duration_frames,
                        "duration": escape_duration_seconds,
                        "avg_speed": np.mean(valid_speed),
                        "max_speed": np.max(valid_speed)
                    })
    
    # Case 2: Mouse doesn't freeze at all - direct escape
    if not freeze_periods:
        # Backtrack from shelter to find when speed exceeds escape threshold
        escape_end_idx = len(distance) - 1
        
        # Find the first frame where speed exceeds threshold (scanning backwards)
        for i in range(escape_end_idx, 0, -1):
            if smooth_speed[i] <= escape_speed_threshold:
                escape_start_idx = i
                break
        else:
            # If no point exceeds threshold, use the first frame
            escape_start_idx = 0
        
        escape_duration_frames = escape_end_idx - escape_start_idx + 1
        
        if escape_duration_frames >= min_escape_duration_frames:
            escape_duration_seconds = escape_duration_frames / fps
            escape_speed_segment = smooth_speed[escape_start_idx:escape_end_idx+1]
            
            # Filter out NaN values for calculations
            valid_speed = escape_speed_segment[~np.isnan(escape_speed_segment)]
            
            if len(valid_speed) > 0:
                escape_periods.append({
                    "start_idx": escape_start_idx,
                    "end_idx": escape_end_idx,
                    "start_time": escape_start_idx / fps,
                    "end_time": escape_end_idx / fps,
                    "duration_frames": escape_duration_frames,
                    "duration": escape_duration_seconds,
                    "avg_speed": np.mean(valid_speed),
                    "max_speed": np.max(valid_speed)
                })
    
    # Prepare the result dictionary
    result = {
        "freeze_start_frame": 0,
        "freeze_end_frame": 0,
        "freeze_start_time": 0,
        "freeze_end_time": 0,
        "freeze_duration": 0,
        "escape_avg_speed": None,
        "escape_max_speed": None,
        "all_freezes": freeze_periods,
        "all_escapes": escape_periods,
        "freeze_detected": bool(freeze_periods)
    }
    
    # Find the longest freeze period
    if freeze_periods:
        longest_freeze = max(freeze_periods, key=lambda x: x["duration_frames"])
        result.update({
            "freeze_start_frame": longest_freeze["start_idx"],
            "freeze_end_frame": longest_freeze["end_idx"],
            "freeze_start_time": longest_freeze["start_time"],
            "freeze_end_time": longest_freeze["end_time"],
            "freeze_duration": longest_freeze["duration"]
        })
    
    # Find the longest escape period (or sum all freeze periods before escape)
    if escape_periods:
        longest_escape = max(escape_periods, key=lambda x: x["duration_frames"])
        result.update({
            "escape_avg_speed": longest_escape["avg_speed"],
            "escape_max_speed": longest_escape["max_speed"],
            "escape_start_frame": longest_escape["start_idx"],
            "escape_end_frame": longest_escape["end_idx"],
            "escape_duration": longest_escape["duration"]
        })
    
    return result

def escape_to_shelter(mouse_shelter_distance_trial,
                    mouse_speed_trial,
                    escape_distance_threshold = 100, # same units as mouse_shelter_distance_trial
                    escape_speed_threshold = 25,  #cm/sec
                    escape_duration_threshold = 100, #frames
                    verbose=False):
    indices = np.where(mouse_shelter_distance_trial < escape_distance_threshold)[0]
    if len(indices) == 0:
        if verbose:
            print(f"No frames found where mouse is within {escape_distance_threshold} of shelter.")
            print(f"Minimum mouse distance was: {np.nanmin(mouse_shelter_distance_trial)}")
        return False
    else:
        first_escape_frame = indices[0]
        if first_escape_frame >= escape_duration_threshold:
            if verbose:
                print(f"Mouse reached shelter too late (frame {first_escape_frame}), after minimum escape duration threshold ({escape_duration_threshold} frames).")
            return False
        speed_above_threshold = np.any(mouse_speed_trial > escape_speed_threshold)
        if not speed_above_threshold:
            if verbose:
                print(f"Mouse did not exceed speed threshold ({escape_speed_threshold} cm/sec) during escape).")
            return False
        return True
    
    
# Main looming detection function
def detect_luminance_change(video_path,
                            roi_points,
                            threshold_percent=9,
                            min_area_percent=60,
                            return_luminance=False,
                            verbose=False):
    """
    Detect start and end frames of significant luminance drops in a specified region of interest.
    Memory-optimized version with error handling and resource management.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        # Read first frame and set up mask
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Error reading first frame")
        
        height, width = first_frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Ensure roi_points has the correct format for OpenCV
        roi_points = np.array(roi_points, dtype=np.int32)
        if roi_points.ndim == 1:
            # If it's a flat array, reshape it to (n_points, 2)
            roi_points = roi_points.reshape(-1, 2)
        
        # Ensure we have at least 3 points for a polygon
        if len(roi_points) < 3:
            raise ValueError(f"ROI must have at least 3 points, got {len(roi_points)}")
        
        cv2.fillPoly(mask, [roi_points], 255)
        roi_area = cv2.contourArea(roi_points)
        
        # Initialize variables with fixed-size deque for memory efficiency
        import gc
        from collections import deque
        window_size = 5
        luminance_history = deque(maxlen=window_size)
        baseline = None
        events = []
        current_event = None
        frame_number = 0
        
        # Initialize luminance storage if requested
        full_luminance = [] if return_luminance else None
        
        # Process frames with memory-efficient approach
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale and release original frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                del frame  # Explicitly release frame memory
                
                # Apply mask and calculate luminance
                roi = cv2.bitwise_and(gray, gray, mask=mask)
                dark_threshold = 30
                valid_pixels = roi[np.logical_and(roi > dark_threshold, mask > 0)]
                
                # Calculate current luminance
                if len(valid_pixels) < (roi_area * min_area_percent / 100):
                    current_luminance = luminance_history[-1] if luminance_history else None
                else:
                    current_luminance = float(np.mean(valid_pixels))  # Convert to float to reduce memory
                
                luminance_history.append(current_luminance)
                
                # Store luminance if requested
                if return_luminance:
                    full_luminance.append(current_luminance)
                
                # Set baseline
                if baseline is None and current_luminance is not None:
                    baseline = current_luminance
                
                # Process luminance changes
                if len(luminance_history) == window_size and None not in luminance_history:
                    avg_luminance = float(np.mean(luminance_history))
                    
                    if current_event is None:
                        if avg_luminance < (baseline * (1 - threshold_percent/100)):
                            current_event = frame_number
                    elif avg_luminance >= (baseline * (1 - threshold_percent/100)):
                        events.append((current_event, frame_number-1))
                        current_event = None
                
                frame_number += 1
                
                # Periodic garbage collection for long videos
                if frame_number % 1000 == 0:
                    gc.collect()
                
            except Exception as e:
                print(f"Error processing frame {frame_number}: {str(e)}")
                continue
                
    except Exception as e:
        raise RuntimeError(f"Video processing failed: {str(e)}")
        
    finally:
        cap.release()
        
    # Handle any unfinished event at end of video
    if current_event is not None:
        events.append((current_event, frame_number-1))
    
    # Return events and optionally luminance data
    if return_luminance:
        return events, full_luminance
    else:
        return events

# ------------- End of looming detection function -------------

#--- Visualization functions -------------

def plot_shelter_distance(mouse_shelter_distance_loom,
                         pad_onset,
                        line_color = 'green',
                        ax = None,
                        add_lines = True):
                        
    n_looms = len(mouse_shelter_distance_loom)
    #WIP: remove magic numbers, change units on x and y axis
    if ax is None:
        plt.figure(dpi = 100)
        ax = plt.gca()
    
    for i in range(n_looms):
        ax.plot(mouse_shelter_distance_loom[i], color = line_color)
        if add_lines:
            #add vertical line at loom start and end
            ax.axvline(x = pad_onset+1, color = 'black', linestyle = '--', alpha = 0.7, linewidth = 1)
            ax.axvline(x = pad_onset+15+1, color = 'black', linestyle = '--', alpha = 0.7, linewidth = 1)
            ax.axvline(x = pad_onset+30+1, color = 'black', linestyle = '--', alpha = 0.7, linewidth = 1)
            ax.axvline(x = pad_onset+45+1, color = 'black', linestyle = '--', alpha = 0.7, linewidth = 1)

        #Make three black circles at the top of the plot
        ax.scatter((pad_onset+1)+(15/2), 10, color = 'black', marker = 'o', s =80)
        ax.scatter((pad_onset+15+1)+(15/2), 10, color = 'black', marker = 'o', s =80)
        ax.scatter((pad_onset+30+1)+(15/2), 10, color = 'black', marker = 'o', s =80)
        #Add a grey box at the bottom of the plot
        ax.axhspan(20, 75, color='olive', alpha=0.2)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Distance to shelter (pixels)')
        #Change ticks on x axis to seconds
        ax.set_xticks(np.arange(0, len(mouse_shelter_distance_loom[0]), 30))
        ax.set_xticklabels(np.arange(0, len(mouse_shelter_distance_loom[0])/30, 1))
        #add text label saying escape threshold at the bottom of the plot
        ax.text(0.75, 0.07, f'Escape', transform=ax.transAxes, ha='left', va='top', fontsize=14, color = 'Olive')
    return ax

def plot_cricket_position_during_loom(heading_cricket_mouse_loom,
                                     ax = None,
                                     plot_type = 'histogram',
                                     binocular_angle = 40,
                                     dot_color = 'orange'):
      
    fig = plt.figure(figsize=(6, 10), dpi=100)
    if ax is None:
        ax = plt.subplot(111, projection='polar')
    
    r=1
    if plot_type == 'scatter':
        for i in range(len(heading_cricket_mouse_loom)):
            ax.scatter(heading_cricket_mouse_loom[i], 5, s=50, color = dot_color)
    elif plot_type == 'histogram':
        # Create histogram bins (36 bins = 10 degrees each)
        bins = np.linspace(-np.pi, np.pi, 37)  # 36 bins + 1 for edge
        
        # Plot polar histogram
        hist, bins = np.histogram(heading_cricket_mouse_loom, bins=bins)
        width = (bins[1] - bins[0])
        bars = ax.bar(bins[:-1], hist, width=width, alpha=0.7, color=dot_color)
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    angles = np.arange(0, 360, 45)
    ax.set_xticks(np.deg2rad(angles))
    ax.set_xticklabels(['{}°'.format(angle) for angle in angles])
    ax.set_yticklabels([])
    ax.set_ylim([0,10]) #- set this based on maximum distance?

    theta_start = np.deg2rad(-binocular_angle/2)
    theta_end = np.deg2rad(binocular_angle/2) 

    # Fill the area between the defined theta range
    ax.fill_between([theta_start, theta_end], 0, 40, color='red', alpha=0.1)

    # Adjust layout
    plt.tight_layout()
    return ax

# Find speed for a given id in metadata_df
def get_speed_for_id(metadata_df, mouse_id):
    matching_rows = metadata_df[metadata_df['id'] == mouse_id]
    if len(matching_rows) > 0:
        assert len(matching_rows) == 1, f"Expected 1 row for mouse ID {mouse_id}, got {len(matching_rows)}"
        return ast.literal_eval(matching_rows['speed'].iloc[0])
    else:
        print(f"Mouse ID {mouse_id} not found in metadata")
        return None
        
def get_mouse_centroid(dataframe):
    # Get x coordinates and check for nans
    x_coords = [dataframe.nose.x.values,
                dataframe.leftear.x.values,
                dataframe.rightear.x.values,
                dataframe.tailbase.x.values]
    centroid_x_sum = np.where(np.any(pd.isnull(x_coords), axis=0), 
                             np.nan,
                             sum(x_coords))

    # Get y coordinates and check for nans 
    y_coords = [dataframe.nose.y.values,
                dataframe.leftear.y.values,
                dataframe.rightear.y.values,
                dataframe.tailbase.y.values]
    centroid_y_sum = np.where(np.any(pd.isnull(y_coords), axis=0),
                             np.nan, 
                             sum(y_coords))
    centroid_x = centroid_x_sum/4.0
    centroid_y = centroid_y_sum/4.0
    return centroid_x, centroid_y

def get_distance_travelled(dataframe, pix2cm_scale):
    centroid_x, centroid_y = get_mouse_centroid(dataframe)
    # Calculate distance in cm per frame
    dx = np.diff(centroid_x)
    dy = np.diff(centroid_y)
    # If either dx or dy is nan, distance should be nan for that step
    distance_cm = np.sqrt(dx**2 + dy**2)
    distance_cm[np.isnan(dx) | np.isnan(dy)] = np.nan
    # Sum up all the distances to get total distance travelled
    # Use nansum to ignore nan values in the summation
    total_distance = np.nansum(distance_cm)
    return total_distance/pix2cm_scale
    
def get_time_in_shelter(dlc_df, shelter_location, likelihood_threshold=0.4, min_body_parts_low_likelihood=3, 
                          consecutive_frames=5, shelter_distance_threshold=50, fps=30.0):
    """
    Calculate time spent in shelter based on low likelihood tracking and proximity to shelter.
    
    Parameters:
    -----------
    dlc_label : str
        Path to DeepLabCut h5 file
    shelter_location : np.ndarray,
        Shelter center location [x, y]
    likelihood_threshold : float
        Likelihood threshold below which body part is considered not tracked (default: 0.5)
    min_body_parts_low_likelihood : int
        Minimum number of body parts (out of 4) with low likelihood (default: 3)
    consecutive_frames : int
        Number of consecutive frames required for shelter detection (default: 5)
    shelter_distance_threshold : float
        Maximum distance (in pixels) from shelter to consider mouse "close to shelter" (default: 50)
    fps : float
        Frames per second for time conversion (default: 30.0)
    
    Returns:
    --------
    float
        Total time spent in shelter (in seconds)
    """
    
    # Body parts to check (4 body parts: leftear, rightear, nose, tailbase)
    body_parts = ['leftear', 'rightear', 'nose', 'tailbase']
    
    # Get likelihood values for all body parts
    n_frames = len(dlc_df)
    low_likelihood_count = np.zeros(n_frames)
    
    for part in body_parts:
        if part in dlc_df.columns.levels[0] if isinstance(dlc_df.columns, pd.MultiIndex) else part in dlc_df.columns:
            try:
                likelihood = dlc_df[part]['likelihood'].values if isinstance(dlc_df.columns, pd.MultiIndex) else dlc_df[f'{part}_likelihood'].values
                low_likelihood_count += (likelihood < likelihood_threshold).astype(int)
            except (KeyError, AttributeError):
                # Try alternative column naming
                try:
                    likelihood = dlc_df[f'{part}']['likelihood'].values
                    low_likelihood_count += (likelihood < likelihood_threshold).astype(int)
                except (KeyError, AttributeError):
                    print(f"Likelihood not found for {part}")

    # Find frames where at least min_body_parts_low_likelihood parts have low likelihood
    frames_with_low_likelihood = low_likelihood_count >= min_body_parts_low_likelihood
    
    # Find consecutive sequences of at least consecutive_frames
    in_shelter_mask = np.zeros(n_frames, dtype=bool)
    
    i = 0
    while i < n_frames:
        if frames_with_low_likelihood[i]:
            # Check if we have consecutive_frames
            end_idx = i
            count = 0
            while end_idx < n_frames and frames_with_low_likelihood[end_idx]:
                count += 1
                end_idx += 1
            
            if count >= consecutive_frames:
                # Check if last legitimate location (before low likelihood) is close to shelter
                # Find last frame with good tracking before this sequence
                last_good_frame = i - 1
                while last_good_frame >= 0 and frames_with_low_likelihood[last_good_frame]:
                    last_good_frame -= 1
                
                if last_good_frame >= 0:
                    # Get mouse position at last good frame
                    try:
                        # Calculate mouse centroid
                        x_coords = []
                        y_coords = []
                        for part in body_parts:
                            try:
                                if isinstance(dlc_df.columns, pd.MultiIndex):
                                    x = dlc_df[part]['x'].iloc[last_good_frame]
                                    y = dlc_df[part]['y'].iloc[last_good_frame]
                                else:
                                    x = dlc_df[f'{part}_x'].iloc[last_good_frame]
                                    y = dlc_df[f'{part}_y'].iloc[last_good_frame]
                                
                                if not np.isnan(x) and not np.isnan(y):
                                    x_coords.append(x)
                                    y_coords.append(y)
                            except (KeyError, IndexError):
                                pass

                        if len(x_coords) > 0:
                            mouse_pos = np.array([np.mean(x_coords), np.mean(y_coords)])
                            distance_to_shelter = np.sqrt(np.sum((mouse_pos - shelter_location)**2))
                            
                            if distance_to_shelter <= shelter_distance_threshold:
                                # Mark these frames as in shelter
                                in_shelter_mask[i:end_idx] = True
                    except (KeyError, IndexError, ValueError):
                        pass

            i = end_idx
        else:
            i += 1
    
    # Calculate total time in shelter
    total_frames_in_shelter = np.sum(in_shelter_mask)
    total_time_seconds = total_frames_in_shelter / fps
    
    return total_time_seconds, in_shelter_mask


def get_number_of_approaches(mouse_id, speed, looming_timestamps):
    """
    Count the number of approaches/looming trials.
    
    Parameters:
    -----------
    mouse_id : str, optional
        Mouse ID
    speed : str
        Speed of the stimulus
    looming_timestamps : dict, optional
        Dictionary with looming timestamps. Keys should match video filename.
        If provided, will count number of looms from this dict.
    
    Returns:
    --------
    int
        Number of approaches/looming trials
    """
    looms = looming_timestamps[f'{mouse_id}_{speed}']
    return len(looms)

def filter_looms_by_time(loom_times, fps, max_seconds_from_first_loom):
    if max_seconds_from_first_loom is None or len(loom_times) == 0:
        return loom_times
    first_loom_start = loom_times[0][0]
    max_frames = max_seconds_from_first_loom * fps
    return [lt for lt in loom_times if (lt[0] - first_loom_start) <= max_frames]

def get_vid_start_and_end_time(dlc_df, likelihood_threshold=0.5, consecutive_frames=5):
    """
    Find first and last frames where stimulus likelihood is above threshold for consecutive frames.
    
    Parameters:
    -----------
    dlc_df : pd.DataFrame
        DataFrame with DeepLabCut data
    likelihood_threshold : float
        Minimum likelihood threshold for stimulus detection (default: 0.5)
    consecutive_frames : int
        Number of consecutive frames required (default: 5)
    
    Returns:
    --------
    dict
        Dictionary with 'start_frame' and 'end_frame' keys, or None if not found
    """
        
    # Get stimulus likelihood
    stim_likelihood = dlc_df['stimulus']['likelihood'].values
    
    # Find frames where stimulus likelihood is above threshold
    high_likelihood = stim_likelihood > likelihood_threshold
    
    # Find first sequence of consecutive_frames with high likelihood
    start_frame = None
    for i in range(len(high_likelihood) - consecutive_frames + 1):
        if np.all(high_likelihood[i:i+consecutive_frames]):
            start_frame = i
            break
    
    # Find last sequence of consecutive_frames with high likelihood
    end_frame = None
    for i in range(len(high_likelihood) - consecutive_frames, -1, -1):
        if np.all(high_likelihood[i:i+consecutive_frames]):
            end_frame = i + consecutive_frames - 1
            break
    
    if start_frame is None or end_frame is None:
        assert False, "Start and end frame not found"
        
    return {'start_frame': start_frame, 'end_frame': end_frame}

def resample_events_to_9000(events, current_total_samples, target_samples=9000):
    """
    Simple function to resample event positions to a new total sample count.
    
    Parameters:
    -----------
    events : list or array
        Sample numbers where events occur
    current_total_samples : int
        Current total number of samples
    target_samples : int
        Target number of samples (default: 9000)
    
    Returns:
    --------
    new_events : numpy array
        Adjusted event sample numbers
    """
    
    # Calculate scaling factor
    scale_factor = target_samples / current_total_samples
    
    # Scale event positions
    events = np.array(events)
    new_events = np.round(events * scale_factor).astype(int)
    
    # Handle 0-indexed vs 1-indexed
    if np.min(events) == 1:  # 1-indexed
        new_events = np.clip(new_events, 1, target_samples)
    else:  # 0-indexed
        new_events = np.clip(new_events, 0, target_samples - 1)
    
    return new_events


def plot_trajectory_with_speed(centpos, speed_plot, ax, vmin=None, vmax=None, linewidth=0.8, alpha=0.8):
    """
    Plot mouse trajectory colored by speed using LineCollection
    
    Parameters:
    -----------
    centpos : array
        Mouse center position data with shape (n_frames, 2) where columns are [x, y]
    speed_plot : array
        Speed values for each frame
    ax : matplotlib axis
        Axis to plot on
    vmin : float, optional
        Minimum value for color scaling
    vmax : float, optional
        Maximum value for color scaling
    linewidth : float, optional
        Thickness of the trajectory line (default: 2)
    alpha : float, optional
        Transparency/smoothness of the line (0-1, default: 1.0)
        
    Returns:
    --------
    line : LineCollection
        The line collection object for potential colorbar creation
    """
    # Create line segments from consecutive points
    points = np.array([centpos.T[0], centpos.T[1]]).T
    segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)

    # Remove NaNs from speed_plot and apply Gaussian smoothing
    speed_clean = np.copy(speed_plot)
    nan_mask = np.isnan(speed_clean)
    if np.any(nan_mask):
        # Interpolate NaN values
        valid_indices = np.where(~nan_mask)[0]
        if len(valid_indices) > 1:
            speed_clean[nan_mask] = np.interp(np.where(nan_mask)[0], valid_indices, speed_clean[valid_indices])
    
    # Apply Gaussian smoothing
    speed_smoothed = gaussian_filter1d(speed_clean, sigma=3)

    # Create LineCollection with speed as color
    lc = LineCollection(segments, cmap='inferno', linewidths=linewidth, alpha=alpha)
    lc.set_array(speed_smoothed[:-1])  # Use smoothed speed values for coloring (one less than points)
    
    # Set color limits if provided
    if vmin is not None or vmax is not None:
        lc.set_clim(vmin=vmin, vmax=vmax)

    # Plot
    line = ax.add_collection(lc)    
    return line

def estimate_stim_position(dlc_df, arena_kpts):
    p1 = arena_kpts[2,:]
    p2 = arena_kpts[3,:]

    # Calculate the slope and intercept
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    intercept = p1[1] - slope * p1[0]

    # Project stimulus points onto the fitted line
    stim_x = dlc_df['stimulus']['x'].values
    stim_y = dlc_df['stimulus']['y'].values
    stim_x_proj = (stim_x + slope*stim_y - slope*intercept) / (1 + slope**2)
    stim_y_proj = slope*stim_x_proj + intercept

    return stim_x_proj, stim_y_proj

from scipy import interpolate

# Assuming 'signal' is your array with NaNs
def interpolate_nans(signal):
    # Check if there are any NaNs to interpolate
    if not np.any(np.isnan(signal)):
        return signal
    
    # Find indices of non-NaN values
    valid_indices = np.where(~np.isnan(signal))[0]
    
    # If there are no valid points or too few for interpolation, return original
    if len(valid_indices) <= 1:
        return signal
    
    valid_values = signal[valid_indices]
    
    # Create interpolation function using only valid points
    interp_func = interpolate.interp1d(valid_indices, valid_values, 
                                      bounds_error=False, 
                                      fill_value=(valid_values[0], valid_values[-1]))
    
    # Apply to all indices
    all_indices = np.arange(len(signal))
    interpolated_signal = interp_func(all_indices)
    
    # Only replace NaN values, keep original non-NaN values
    result = signal.copy()
    nan_mask = np.isnan(signal)
    result[nan_mask] = interpolated_signal[nan_mask]
    
    return result

def correct_heading_coordinates(dlc_df, likelihood_threshold = 0.4):
    keys = ['leftear', 'rightear', 'tailbase', 'nose']
    heading_df = dlc_df.copy()
    for key in keys:
        likelihood_mask = dlc_df[key]['likelihood'] < likelihood_threshold
        # Use .loc to avoid SettingWithCopyWarning
        heading_df.loc[likelihood_mask, (key, 'x')] = np.nan
        heading_df.loc[likelihood_mask, (key, 'y')] = np.nan
        
        # Interpolate the NaN values
        x_values = heading_df[key]['x'].values
        y_values = heading_df[key]['y'].values
        
        # Apply interpolation and update using .loc
        heading_df.loc[:, (key, 'x')] = interpolate_nans(x_values)
        heading_df.loc[:, (key, 'y')] = interpolate_nans(y_values)
    return heading_df

def correct_stimulus_coordinates(dlc_df, arena_kpts, pix2cm_scale = 1, frame_size = (480, 640), likelihood_threshold = 0.4, stimulus_likelihood_threshold = 0.1):
    if pix2cm_scale == 1:
        print('No pix2cm_scale provided, using default value of 1')

    keypoints = ['stimulus', 'leftear', 'rightear', 'tailbase', 'nose']
    dlc_df_corrected = dlc_df.copy()
    for key in keypoints:
        if key == 'stimulus':
            stim_x_proj, stim_y_proj = estimate_stim_position(dlc_df, arena_kpts)
            dlc_df_corrected.loc[:, (key, 'x')] = stim_x_proj 
            dlc_df_corrected.loc[:, (key, 'y')] = stim_y_proj 
            dlc_df_corrected.loc[:, 'stimulus_distance'] = np.sqrt((arena_kpts[2,0] - stim_x_proj)**2 + (arena_kpts[2,1] - stim_y_proj)**2) / pix2cm_scale
        
        dlc_df_corrected.loc[:, (key, 'x')] = (frame_size[1] - dlc_df[key]['x'] - (frame_size[1] - arena_kpts[2,0])) / pix2cm_scale
        dlc_df_corrected.loc[:, (key, 'y')] = (frame_size[0] - dlc_df[key]['y'] - (frame_size[0] - arena_kpts[2,1])) / pix2cm_scale
    
        likelihood_mask = dlc_df[key]['likelihood'] < likelihood_threshold
        
        dlc_df_corrected.loc[likelihood_mask, (key, 'x')] = np.nan
        dlc_df_corrected.loc[likelihood_mask, (key, 'y')] = np.nan
        
        # Interpolate the NaN values
        x_values = dlc_df_corrected[key]['x'].values
        y_values = dlc_df_corrected[key]['y'].values
        
        # Apply interpolation and update using .loc
        dlc_df_corrected.loc[:, (key, 'x')] = interpolate_nans(x_values)
        dlc_df_corrected.loc[:, (key, 'y')] = interpolate_nans(y_values)
    
    # Set likelihood values below threshold to NaN for stimulus_distance
    likelihood_mask = dlc_df['stimulus']['likelihood'] < likelihood_threshold
    dlc_df_corrected.loc[likelihood_mask, ('stimulus_distance')] = np.nan
    
    # Interpolate the stimulus_distance NaN values
    stimulus_distance_values = dlc_df_corrected['stimulus_distance'].values
    dlc_df_corrected.loc[:, 'stimulus_distance'] = interpolate_nans(stimulus_distance_values)

    # # Set likelihood values below threshold to NaN for stimulus_distance
    likelihood_mask = dlc_df['stimulus']['likelihood'] < stimulus_likelihood_threshold
    dlc_df_corrected.loc[likelihood_mask, ('stimulus_distance')] = np.nan

    return dlc_df_corrected