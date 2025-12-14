from medpy import metric
import numpy as np

def test_single_case(score_map):
    """
    Converts a softmax probability score_map (shape: [2, H, W, D]) into a hard segmentation using argmax.
    """
    label_map = np.argmax(score_map, axis=0)
    return label_map

def calculate_metric_percase(pred, gt):
    """
    Computes 3D evaluation metrics using medpy on binary volumes.
    """
    if np.sum(gt) == 0 and np.sum(pred) == 0:
        return 1.0, 1.0, 0.0, 0.0
    elif np.sum(gt) == 0 or np.sum(pred) == 0:
        return 0.0, 0.0, 0.0, 0.0
    else:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        hd = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, jc, hd, asd