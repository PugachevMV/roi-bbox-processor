from collections import defaultdict

from utils import roi_tlwh_2_tlbr, extract_bboxes_and_scores, find_bboxes_near_roi_boundaries, merge_multiple_bboxes
from nms import nms_with_boundary_threshold, greedy_nmm_with_boundary_threshold

class BoundingBoxProcessor:
    def __init__(self,
                 player_bboxes_list,
                 roi_tlwh_list,
                 delta=20,
                 match_metric="IOU",
                 match_threshold=0.5,
                 match_threshold_boundary=0.1):
        """
        Initializes the BoundingBoxProcessor class with player bounding boxes and ROI (Region of Interest) data.

        Arguments:
            player_bboxes_list (dict): A dictionary containing bounding box information for each player.
            roi_tlwh_list (list): A dictionary containing the Region of Interest (ROI) data in TLWH (Top-Left, Width, Height) format.
            delta (int): A parameter to adjust the boundary expansion for each ROI.
            match_metric (str): The metric used for calculating the match between bounding boxes in NMS or greedy NMS (e.g., "IOU").
            match_threshold (float): The threshold for the matching metric when calculating NMS or greedy NMS for boxes far from the boundary.
            match_threshold_boundary (float): The threshold for the matching metric when calculating NMS or greedy NMS for boxes near the boundary.

        Description:
            This method initializes the BoundingBoxProcessor class, setting up all necessary internal data structures 
            for further processing.
        """
        self.player_bboxes_list = player_bboxes_list
        self.roi_tlwh_list = roi_tlwh_list
        self.delta = delta
        self.match_metric = match_metric
        self.match_threshold = match_threshold
        self.match_threshold_boundary = match_threshold_boundary

        self.roi_tlbr_list = roi_tlwh_2_tlbr(roi_tlwh_list)

        player_bboxes_dict = defaultdict(list)
        for item in player_bboxes_list:
            frame_number = item['frame']
            player_bboxes_dict[frame_number].append(item)
        self.player_bboxes_dict = player_bboxes_dict

        self.frame_bboxes_scores_dict = extract_bboxes_and_scores(player_bboxes_dict)

    def get_bbox_ids_on_boundaries(self):
        """
        Finds and returns a dictionary of bounding box IDs that intersect with the ROI boundaries.

        Returns:
            dict: A dictionary where the key is the frame ID and the value is a list of bounding box IDs that intersect the boundary.
        """
        bbox_ids_on_boundaries_dict = {}
        for key in self.frame_bboxes_scores_dict:
            bbox_ids_on_boundaries_dict[key] = find_bboxes_near_roi_boundaries(self.player_bboxes_dict[key], self.roi_tlbr_list, delta=self.delta)
        return bbox_ids_on_boundaries_dict

    def get_roi_tlbr_list(self):
        """
        Returns the list of ROI coordinates in TLBR (Top-Left, Bottom-Right) format.

        Returns:
            list: A list of ROI coordinates for each frame, where each entry contains the coordinates in TLBR format [y1, x1, y2, x2].
        """
        return self.roi_tlbr_list


class NMSWithBoundaryProcessor(BoundingBoxProcessor):
    def __init__(self, *args, **kwargs):
        """
        Inherits from BoundingBoxProcessor and implements NMS for bounding box filtering.
        """
        super().__init__(*args, **kwargs)

    def filter_bboxes(self, use_bbox_ids_on_boundaries=True):
        """
        Applies the Non-Maximum Suppression (NMS) procedure with a boundary threshold to the bounding boxes.

        The method processes each frame in the set of bounding box predictions, applies NMS with different thresholds for boundary objects, 
        and returns the filtered player data based on these bounding boxes.

        If `use_bbox_ids_on_boundaries` is set to True, it will consider those bounding boxes that intersect with the ROI boundaries.
        If set to False, no boundary-based filtering will be applied.
        
        Return value:
            list: A list of filtered player data based on bounding boxes that have passed through the NMS procedure.
        """
        bbox_ids_on_boundaries_dict = self.get_bbox_ids_on_boundaries() if use_bbox_ids_on_boundaries else None
        filtered_nms_dict = {}

        for frame_id, predictions in self.frame_bboxes_scores_dict.items():
            filtered_nms_dict[frame_id] = nms_with_boundary_threshold(
                predictions,
                match_metric=self.match_metric,
                match_threshold=self.match_threshold,
                bbox_ids_on_boundaries=bbox_ids_on_boundaries_dict[frame_id] if use_bbox_ids_on_boundaries else None,
                match_threshold_boundary=self.match_threshold_boundary
            )

        filtered_player_data = []
        for frame in sorted(self.player_bboxes_dict.keys()):
            for ind_bbox in filtered_nms_dict[frame]:
                filtered_player_data.append(self.player_bboxes_dict[frame][ind_bbox])

        return filtered_player_data


class GreedyNMMWithBoundaryProcessor(BoundingBoxProcessor):
    def __init__(self, *args, **kwargs):
        """
        Inherits from BoundingBoxProcessor and implements Greedy NMM for bounding box filtering.
        """
        super().__init__(*args, **kwargs)

    def filter_bboxes(self,use_bbox_ids_on_boundaries=True):
        """
        Applies Greedy Non-Maximum Suppression (Greedy NMM) with boundary threshold to the bounding boxes.
        
        The method processes each frame in the set of bounding box predictions, applies Greedy NMM with different thresholds for boundary objects, 
        and returns the filtered player data based on these bounding boxes.
        
        If `use_bbox_ids_on_boundaries` is set to True, it will consider those bounding boxes that intersect with the ROI boundaries.
        If set to False, no boundary-based filtering will be applied.

        Returns:
            list: A list of filtered player data based on bounding boxes that have passed through the Greedy NMM procedure.
        """
        bbox_ids_on_boundaries_dict = self.get_bbox_ids_on_boundaries() if use_bbox_ids_on_boundaries else None

        greedy_nms_filtered_dict = {}
        for frame_id, predictions in self.frame_bboxes_scores_dict.items():
            greedy_nms_filtered_dict[frame_id] = greedy_nmm_with_boundary_threshold(
                predictions,
                match_metric=self.match_metric,
                match_threshold=self.match_threshold,
                bbox_ids_on_boundaries=bbox_ids_on_boundaries_dict[frame_id] if use_bbox_ids_on_boundaries else None,
                match_threshold_boundary=self.match_threshold_boundary
            )

        filtered_player_data = []
        for frame in sorted(self.player_bboxes_dict.keys()):
            for greedy_nms_indices in greedy_nms_filtered_dict[frame]:
                # Check if there are any boxes in the list for this index
                if len(greedy_nms_filtered_dict[frame][greedy_nms_indices]) == 0:
                    filtered_player_data.append(self.player_bboxes_dict[frame][greedy_nms_indices])
                else:
                    # Merge multiple bounding boxes
                    bbox_indices_to_merge = greedy_nms_filtered_dict[frame][greedy_nms_indices] + [greedy_nms_indices]
                    merged_bbox = merge_multiple_bboxes(bbox_indices_to_merge, self.player_bboxes_dict[frame])
                    filtered_player_data.append(merged_bbox)
                    
        return filtered_player_data
