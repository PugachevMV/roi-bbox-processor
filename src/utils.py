from typing import List, Dict
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from typing import Union
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def load_json_files(file_path: str):
    """
    Loads data from a JSON file at the specified path.

    :param file_path: Path to the JSON file.
    :return: Data loaded from the JSON file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def roi_tlwh_2_tlbr(roi_tlwh: List[List[int]]) -> List[List[int]]:
    """
    Converts a list of ROI coordinates from TLWH (Top-Left, Width, Height) format to TLBR (Top-Left, Bottom-Right) format.

    :param roi_tlwh: List of ROIs in TLWH format (x1, y1, width, height).
    :return: List of ROIs in TLBR format (y1, x1, y2, x2).
    """
    roi_tlbr = []
    for roi in roi_tlwh:
        x1, y1, width, height = roi
        roi_tlbr.append([y1, x1, y1 + height, x1 + width])
    
    return roi_tlbr

def extract_bboxes_and_scores(player_bboxes_list: Dict[int, List[Dict[str, Union[List[float], float]]]]) -> Dict[int, torch.Tensor]:

    """
    Extracts bounding boxes and scores for all frames in the provided player_bboxes_list,
    returning a dictionary with frame IDs as keys and prediction tensors (bounding boxes and scores) as values.

    Args:
        player_bboxes_list (dict): A dictionary where each key is a frame ID and the value is a list of detections
                                   (each detection contains a bounding box and a score).

    Returns:
        dict: A dictionary with frame IDs as keys and prediction tensors (bounding boxes and scores) as values.
    """
    
    frame_bboxes_scores_dict = {}

    for frame_id, detections in player_bboxes_list.items():
        if not detections:
            logger.warning(f"No objects detected for frame {frame_id}")
            frame_bboxes_scores_dict[frame_id] = torch.empty(0, 5)
            continue

        frame_predictions = []
        for det in detections:
            bbox = det['bbox']
            score = det['score']
            frame_predictions.append(bbox + [score])

        frame_bboxes_scores_dict[frame_id] = torch.tensor(frame_predictions, dtype=torch.float32)

    return frame_bboxes_scores_dict

def is_bounding_box_near_roi_boundary(bbox: List[float], roi_tlbr: List[float], delta: int = 10) -> bool:
    """
    Checks if the bounding box is near the boundary of a given ROI, with some tolerance (delta).

    Args:
        bbox (list): Bounding box of the object [x1, y1, x2, y2].
        roi_tlbr (list): ROI of the camera [y1, x1, y2, x2].
        delta (int): The amount by which the ROI boundary is expanded or contracted for checking the bounding box,  default is 10.


    Returns:
        bool: True if the bbox is near the ROI boundary, False otherwise.
    """
    x1, y1, x2, y2 = bbox
    roi_y1, roi_x1, roi_y2, roi_x2 = roi_tlbr

    # Adjust the ROI boundaries with the given delta
    roi_x1 = int(roi_x1 + delta)
    roi_y1 = int(roi_y1 + delta)
    roi_x2 = int(roi_x2 - delta)
    roi_y2 = int(roi_y2 - delta)

    # Check for vertical overlap near the boundary
    vertical_overlap = ((y1 < roi_y1 and y2 > roi_y1) or (y1 < roi_y2 and y2 > roi_y2)) and (x1 >= roi_x1 and x2 <= roi_x2)

    # Check for horizontal overlap near the boundary
    horizontal_overlap = ((x1 < roi_x1 and x2 > roi_x1) or (x1 < roi_x2 and x2 > roi_x2)) and (y1 >= roi_y1 and y2 <= roi_y2)
    
    return horizontal_overlap or vertical_overlap


def find_bboxes_near_roi_boundaries(frames_dict: List[Dict], roi_tlbr: List[List[float]], delta: int = 10) -> List[int]:
    """
    Finds bounding boxes that are near the boundary of any camera (ROI).

    Args:
        frames_dict (list): List of detections for each frame.
        roi_tlbr (list): List of camera ROIs (each in TLBR format).
        delta (int): TThe amount by which the ROI boundary is expanded or contracted for checking the bounding box, default is 10.

    Returns:
        list: List of object IDs whose bounding boxes are near the camera boundary.
    """
    bbox_ids_near_boundaries = []

    for frame_id, detection in enumerate(frames_dict):
        bbox = detection['bbox']
        
        # Check if the bounding box intersects with any ROI boundary
        for roi in roi_tlbr:
            if is_bounding_box_near_roi_boundary(bbox, roi, delta):
                bbox_ids_near_boundaries.append(frame_id)
                break

    return bbox_ids_near_boundaries


def merge_multiple_bboxes(bbox_indices, player_bboxes):
    """
    Merges multiple bounding boxes by selecting the maximum coordinates for the top-left 
    corner and minimum coordinates for the bottom-right corner, and calculates a weighted 
    average score based on the area of each bounding box.

    Args:
        bbox_indices (list): A list of indices representing bounding boxes to be merged.
        player_bboxes (dict): A dictionary containing all the bounding box predictions for frames,
                              where each entry is a dictionary with 'bbox', 'score', 'frame', 
                              'detection_class', 'X', 'Y', etc.

    Returns:
        dict: A merged bounding box with the maximum coordinates for the top-left corner and 
              minimum coordinates for the bottom-right corner, along with the weighted average score.
    """
    
    # Extracting the frame and detection_class from the first bounding box in the list
    frame = player_bboxes[bbox_indices[0]]['frame']  
    detection_class = player_bboxes[bbox_indices[0]]['detection_class']
    
    # Check if all bounding boxes belong to the same frame
    for bbox_index in bbox_indices:
        assert player_bboxes[bbox_index]['frame'] == frame, f"Frames don't match: {player_bboxes[bbox_index]['frame']} != {frame}"

        # Check if the detection class matches for all bounding boxes
        if player_bboxes[bbox_index]['detection_class'] != detection_class:
            logger.warning(f"Detection class mismatch for frame {frame}: "
                           f"{player_bboxes[bbox_index]['detection_class']} != {detection_class}")

    # Initialize the variables to store the maximum and minimum coordinates for the merged bounding box
    max_x1 = min(player_bboxes[bbox_index]['bbox'][0] for bbox_index in bbox_indices)
    max_y1 = min(player_bboxes[bbox_index]['bbox'][1] for bbox_index in bbox_indices)
    min_x2 = max(player_bboxes[bbox_index]['bbox'][2] for bbox_index in bbox_indices)
    min_y2 = max(player_bboxes[bbox_index]['bbox'][3] for bbox_index in bbox_indices)

    # Calculate the weighted score based on the area of each bounding box
    total_weighted_score = 0
    total_area = 0
    for bbox_index in bbox_indices:
        x1, y1, x2, y2 = player_bboxes[bbox_index]['bbox']
        area = (x2 - x1) * (y2 - y1)
        score = player_bboxes[bbox_index]['score']
        
        total_weighted_score += score * area  # Weighted score
        total_area += area  # Total area for all bounding boxes

    weighted_score = total_weighted_score / total_area if total_area > 0 else 0

    # Create the merged bounding box result
    merged_bbox = {
        'frame': frame,
        'detection_class': detection_class,
        'bbox': [max_x1, max_y1, min_x2, min_y2],
        'score': weighted_score,
        'X': player_bboxes[bbox_indices[0]]['X'],
        'Y': player_bboxes[bbox_indices[0]]['Y']
    }

    return merged_bbox


def draw_boxes(player_data_list, frame_id, image_size=(8000, 5000), roi_tlbr=None, title="title"):
    """
    Draws bounding boxes for a specified frame_id.

    Also draws Regions of Interest (ROIs) if provided.

    Args:
        player_data_list (list of dict): List of player detections, where each detection contains bounding box coordinates and other information.
        frame_id (int): Frame number for which to visualize the detections.
        image_size (tuple): Size of the image canvas (width, height).
        roi_tlbr (list of lists, optional): List of ROIs in [top, left, bottom, right] format to be drawn on the image.
        title (str, optional): Title to be displayed above the plot.

    Notes:
        - Bounding boxes are drawn in red color.
        - ROIs are drawn as blue dashed rectangles.
        - Each bounding box is labeled with its corresponding score (if available).

    """

    player_data_dict = defaultdict(list)
    for item in player_data_list:
        frame_number = item['frame']
        player_data_dict[frame_number].append(item)

    if frame_id not in player_data_dict:
        print(f"No objects found for frame {frame_id}.")
        return
    
    detections = player_data_dict[frame_id]

    if not detections:
        print(f"No detections available for frame {frame_id}.")
        return

    width, height = image_size

    fig, ax = plt.subplots(1, figsize=(15, 10))
    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])  # Invert y-axis for correct display
    ax.set_title(title)

    # Draw ROIs if provided
    if roi_tlbr:
        for roi in roi_tlbr:
            y1, x1, y2, x2 = roi
            roi_rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
            ax.add_patch(roi_rect)
    
    # Draw bounding boxes for detections
    for idx, det in enumerate(detections):
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox

        edge_color = 'red'

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor=edge_color, facecolor='none')
        ax.add_patch(rect)

        # Label the bounding box with score
        score = det.get('score', 0)
        ax.text(x1, y1 - 5, f"{score:.3f}", color=edge_color, fontsize=12)
        ax.set_title(title, fontsize=18)

    plt.show()
