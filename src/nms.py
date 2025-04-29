import torch

def nms_with_boundary_threshold(
    predictions: torch.tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
    bbox_ids_on_boundaries: list = None,
    match_threshold_boundary: float = 0.3
):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object with different thresholds for boundary and non-boundary objects.

    Args:
        predictions: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes, 5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap threshold for
            match metric for non-boundary objects.
        bbox_ids_on_boundaries: (list) List of indices of objects that intersect the ROI boundaries.
        match_threshold_boundary: (float) The overlap threshold for
            match metric for objects on the boundary.

    Returns:
        A list of filtered indexes, Shape: [ ,]
    """

    if bbox_ids_on_boundaries is None:
        bbox_ids_on_boundaries = []
    bbox_ids_on_boundaries = set(bbox_ids_on_boundaries)
    # we extract coordinates for every
    # prediction box present in P
    x1 = predictions[:, 0]
    y1 = predictions[:, 1]
    x2 = predictions[:, 2]
    y2 = predictions[:, 3]

    # we extract the confidence scores as well
    scores = predictions[:, 4]

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)

    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for
    # filtered prediction boxes
    keep = []

    while len(order) > 0:
        
        # extract the index of the
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]
        #print(idx)
        # push S in filtered predictions list
        keep.append(idx.tolist())

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break

        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=order)

        if match_metric == "IOU":
            # find the union of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            union = (rem_areas - inter) + areas[idx]
            # find the IoU of every prediction in P with S
            match_metric_value = inter / union

        elif match_metric == "IOS":
            # find the smaller area of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            smaller = torch.min(rem_areas, areas[idx])
            # find the IoS of every prediction in P with S
            match_metric_value = inter / smaller
        else:
            raise ValueError()

        # Decide which threshold to use based on whether the current box is on the boundary
        if int(idx) in bbox_ids_on_boundaries:
            match_threshold_used = match_threshold_boundary  # Use stricter threshold for boundary objects
        else:
            match_threshold_used = match_threshold  # Use regular threshold for other objects
        # Keep the boxes with IoU/IoS less than the threshold
        mask = match_metric_value < match_threshold_used
        order = order[mask]
        

    return keep




def greedy_nmm_with_boundary_threshold(
    object_predictions_as_tensor: torch.tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
    bbox_ids_on_boundaries: list = None,
    match_threshold_boundary: float = 0.3
):
    """
    Apply greedy version of non-maximum merging to avoid detecting too many
    overlapping bounding boxes for a given object with different thresholds for boundary and non-boundary objects.

    Args:
        object_predictions_as_tensor: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap threshold for
            match metric for non-boundary objects.
        bbox_ids_on_boundaries: (list) List of indices of objects that intersect the ROI boundaries.
        match_threshold_boundary: (float) The overlap threshold for
            match metric for objects on the boundary.

    Returns:
        keep_to_merge_list: (Dict[int:List[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    if bbox_ids_on_boundaries is None:
        bbox_ids_on_boundaries = []
    bbox_ids_on_boundaries = set(bbox_ids_on_boundaries)

    keep_to_merge_list = {}

    # Extract coordinates for every prediction box present in P
    x1 = object_predictions_as_tensor[:, 0]
    y1 = object_predictions_as_tensor[:, 1]
    x2 = object_predictions_as_tensor[:, 2]
    y2 = object_predictions_as_tensor[:, 3]

    # Extract the confidence scores as well
    scores = object_predictions_as_tensor[:, 4]

    # Calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)

    # Sort the prediction boxes according to their confidence scores
    order = scores.argsort()

    while len(order) > 0:
        # Extract the index of the prediction with the highest score (S)
        idx = order[-1]

        # Push S in filtered predictions list
        keep_to_merge_list[idx.tolist()] = []

        # Remove S from P
        order = order[:-1]

        # Sanity check
        if len(order) == 0:
            break

        # Select coordinates of BBoxes according to the indices in order
        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)

        # Find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # Find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # Take max with 0.0 to avoid negative w and h due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # Find the intersection area
        inter = w * h

        # Find the areas of BBoxes according to the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=order)

        if match_metric == "IOU":
            # Find the union of every prediction T in P with the prediction S
            # Note that areas[idx] represents area of S
            union = (rem_areas - inter) + areas[idx]
            # Find the IoU of every prediction in P with S
            match_metric_value = inter / union

        elif match_metric == "IOS":
            # Find the smaller area of every prediction T in P with the prediction S
            # Note that areas[idx] represents area of S
            smaller = torch.min(rem_areas, areas[idx])
            # Find the IoS of every prediction in P with S
            match_metric_value = inter / smaller
        else:
            raise ValueError()

        # Determine which threshold to use (based on whether the box is on the boundary)
        if int(idx) in bbox_ids_on_boundaries:
            match_threshold_used = match_threshold_boundary  # Use stricter threshold for boundary objects
        else:
            match_threshold_used = match_threshold  # Use regular threshold for non-boundary objects

        # Keep the boxes with IoU/IoS less than the threshold
        mask = match_metric_value < match_threshold_used
        matched_box_indices = order[(mask == False).nonzero().flatten()].flip(dims=(0,))  # noqa: E712
        unmatched_indices = order[(mask == True).nonzero().flatten()]  # noqa: E712

        # Update box pool
        order = unmatched_indices[scores[unmatched_indices].argsort()]

        # Add the matched boxes to the merge list
        for matched_box_ind in matched_box_indices.tolist():
            keep_to_merge_list[idx.tolist()].append(matched_box_ind)

    return keep_to_merge_list
