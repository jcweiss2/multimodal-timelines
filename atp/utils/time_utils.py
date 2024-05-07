from datetime import datetime
import re
import math

import numpy as np
import torch


cls_label = {
    'type': {'bounding': 0, 'probabilistic': 1},
    'inf': {'inf': 0, 'valid': 1},
}

event_label_idx = {'lb': 0, 'ub': 1, 'lb_std': 2, 'ub_std': 3, 'd_std': 4}

map_std_to_secs = {
    'NA': 0,
    '1h': 3600,
    '2h': 7200,
    '6h': 21600,
    '4h': 14400,
    '12h': 43200,
    '1d': 86400,
    '2d': 172800,
    '1w': 604800,
    '1m': 2592000,
    '1y': 31536000,
    '10y': 315360000,
    '100y': 3153600000,
}

map_std_to_cls_full = {
    '1h': 0,
    '2h': 1,
    '4h': 2,
    '6h': 3,
    '12h': 4,
    '1d': 5,
    '2d': 6,
    '1w': 7,
    '1m': 8,
    '1y': 9,
    '10y': 10,
    '100y': 11,
}

# map_std_to_cls_simple = {
    # '1h': 0, '2h': 0,
    # '4h': 1, '6h': 1, '12h': 1,
    # '1d': 2, '2d': 2, '1w': 2, '1m': 3,
    # '1y': 3, '10y': 3, '100y': 3,
# }

map_lbubstd_to_cls_simple = {
    '1h': 0, '2h': 0,
    '4h': 1, '6h': 1, '12h': 1,
    '1d': 2, '2d': 2, '1w': 2, '1m': 2, '1y': 2, '10y': 2, '100y': 2,
}

map_durstd_to_cls_simple = {
    '1h': 0,
    '2h': 1, '4h': 1, '6h': 1, '12h': 1,
    '1d': 2, '2d': 2, '1w': 2, '1m': 2, '1y': 2, '10y': 2, '100y': 2,
}


# for adjust datetime of ub,lb in pevent using admittime, and tranlate to a delta in secs
def adjust_by_admittime_in_secs(admit_time, input_time):
    fmt_code = "%Y-%m-%d %H:%M:%S"
    admit_time = datetime.strptime(admit_time, fmt_code)
    t_input = datetime.strptime(input_time, fmt_code)
    delta = (t_input - admit_time).total_seconds()
    return delta

def adjust_by_admittime_in_secs_struct(admit_time, input_time):
    fmt_code1 = "%Y-%m-%d %H:%M:%S"
    fmt_code2 = "%Y-%m-%dT%H:%M:%SZ"
    admit_time = datetime.strptime(admit_time, fmt_code1)
    t_input = datetime.strptime(input_time, fmt_code2)
    delta = (t_input - admit_time).total_seconds()
    return delta

"""_summary_
label: ['2183-01-18 01:03:22', '2183-01-18 02:15:28', '2h', '2h', '1h']
label_idx = {
    'lb':0, 'ub':1, 'lb_std':2, 'ub_std':3, 'd_std':4
}
label_feat_idx = {
    'lb_ind':0, 'ub_ind':1, "prob_ind":4, "lb_v":2, "ub_v":3, "lb_std":5, "up_std":6, "d_std":7
}

"""

# TODO: two of these labels should be binary

"""

return [lb_ind, ub_ind, prob_ind, lb_v, ub_v, lb_std, ub_std, d_std]


"""

def get_signed_log(val):
    # Get signed value of log
    # Returns log(val) if val > 1.0, log(-val) if val < -1.0, and 0.0 otherwise
    if isinstance(val, torch.Tensor):
        val_abs = val.abs()
        return torch.where(val_abs < 1.0, torch.zeros_like(val),
                           torch.sign(val)*torch.log(val_abs))
    if isinstance(val, np.ndarray):
        val_abs = np.abs(val)
        return np.where(val_abs < 1.0, np.zeros_like(val),
                        np.sign(val)*np.log(val_abs))
    if abs(val) <= 1.0: return 0.0
    return math.copysign(1.0, val) * math.log(abs(val))

def get_signed_exp(val):
    if isinstance(val, torch.Tensor):
        val_abs = val.abs()
        return torch.where(val == 0.0, torch.zeros_like(val),
                           torch.sign(val)*torch.exp(val_abs))
    if isinstance(val, np.ndarray):
        val_abs = np.abs(val)
        return np.where(val == 0.0, np.zeros_like(val),
                        np.sign(val)*np.exp(val_abs))
    if val == 0.0: return 0.0
    return math.copysign(1.0, val) * math.exp(abs(val))

def get_signed_log1p(val):
    # Get odd function version of log1p 
    # Returns log(val+1.0) if val >= 0.0 and log(-val+1.0) if val < 0.0
    if isinstance(val, torch.Tensor):
        return torch.sign(val) * torch.log(val.abs() + 1.0)
    if isinstance(val, np.ndarray):
        return np.sign(val) * np.log(np.abs(val) + 1.0)
    return math.copysign(1.0, val) * math.log(abs(val)+1.0)

def get_signed_expm1(val):
    # Get inverse of get_signed_log1p
    if isinstance(val, torch.Tensor):
        return torch.sign(val) * (torch.exp(val.abs())-1.0)
    if isinstance(val, np.ndarray):
        return np.sign(val) * (np.exp(np.abs(val))-1.0)
    return math.copysign(1.0, val) * (math.exp(abs(val))-1.0)

def convert_val_to_reg(val, label_type):
    # Convert a second value into regression label (second, minute, or log second)
    if label_type == "sec":
        return val
    elif label_type == "min":
        return val / 60
    elif label_type == "hour":
        return val / 3600
    elif label_type == "day":
        return val / 86400
    elif label_type == "logsec":
        return get_signed_log1p(val)
    else:
        raise ValueError("Wrong label type: %s" % label_type)

def convert_reg_to_val(reg, label_type):
    # Convert regression label (second, minute, or log second) into seconds
    if label_type == "sec":
        return reg
    elif label_type == "min":
        return reg * 60
    elif label_type == "hour":
        return reg * 3600
    elif label_type == "day":
        return reg * 86400
    elif label_type == "logsec":
        return get_signed_expm1(reg)
    else:
        raise ValueError("Wrong label type: %s" % label_type)

def convert_val_to_cls(val, thres=86400.0, num_classes=4):
    if not isinstance(thres, list):
        if num_classes == 4:
            thres = [-thres, 0.0, thres]
        elif num_classes == 3:
            thres = [0.0, thres]

    # ret = [0] * (len(thres) + 1)
    for i, t in enumerate(thres):
        if val < t:
            return i
            # ret[i] = 1
    # if sum(ret) == 0:
        # ret[-1] = 1
    # return ret
    return len(thres)

def convert_pevent_to_label(pevent, label_type, admit_time,
                            mean_label_type="sec", std_label_type="log_sec"):
    # pevent (input): (mean_lb, mean_ub, std_lb(str), std_ub(str), std_dur(str))
    # label (output): (lb_ind, ub_ind, prob_ind, mean_lb, mean_ub, std_lb, std_ub, std_dur)
    # mean_label_type, std_label_type: can be either "sec", "min", or "log_sec"

    lb_ind = 0 if pevent[0] in ['Inf', '-Inf', 'inf', '-inf'] else 1
    ub_ind = 0 if pevent[1] in ['Inf', '-Inf', 'inf', '-inf'] else 1

    lb_v, ub_v = 0, 0
    if lb_ind == 1:
        lb_v = adjust_by_admittime_in_secs(admit_time, pevent[0])
        lb_v = convert_val_to_reg(lb_v, mean_label_type)
    if ub_ind == 1:
        ub_v = adjust_by_admittime_in_secs(admit_time, pevent[1])
        ub_v = convert_val_to_reg(ub_v, mean_label_type)
    else:
        raise ValueError("UB inf??")

    prob_ind = 1 if label_type == 'probabilistic' else 0
    lb_std = map_std_to_secs[pevent[2]] if label_type=='probabilistic' else 0
    lb_std = convert_val_to_reg(lb_std, std_label_type)
    ub_std = map_std_to_secs[pevent[3]] if label_type=='probabilistic' else 0
    ub_std = convert_val_to_reg(ub_std, std_label_type)
    d_std =  map_std_to_secs[pevent[4]] if label_type=='probabilistic' else 0
    d_std = convert_val_to_reg(d_std, std_label_type)

    return [lb_ind, ub_ind, prob_ind, lb_v, ub_v, lb_std, ub_std, d_std]

def convert_pevent_to_label_cls(pevent, label_type, admit_time,
                                mean_threshold=86400.0, mean_num_classes=4,
                                std_cls_type='simple'):
    # pevent (input): (mean_lb, mean_ub, std_lb(str), std_ub(str), std_dur(str))
    # label (output): (lb_ind, ub_ind, prob_ind, mean_lb, mean_ub, std_lb, std_ub, std_dur)
    # All labels are in classification labels
    # mean_threshold converts lb_mean and ub_mean into one of 4 classes (divided by -thres, 0, thres)
    # std_cls_type can be either "simple" (3 classes) or "full" (12 classes)

    lb_ind = 0 if pevent[0] in ['Inf', '-Inf', 'inf', '-inf'] else 1
    ub_ind = 0 if pevent[1] in ['Inf', '-Inf', 'inf', '-inf'] else 1
    prob_ind = 1 if label_type == 'probabilistic' else 0

    label = [lb_ind, ub_ind, prob_ind]

    lb_v, ub_v = 0, 0
    if lb_ind == 1:
        lb_v = adjust_by_admittime_in_secs(admit_time, pevent[0])
    if ub_ind == 1:
        ub_v = adjust_by_admittime_in_secs(admit_time, pevent[1])
    else:
        raise ValueError("UB inf??")
    lb_v = convert_val_to_cls(lb_v, thres=mean_threshold, num_classes=mean_num_classes)
    ub_v = convert_val_to_cls(ub_v, thres=mean_threshold, num_classes=mean_num_classes)

    if std_cls_type == 'simple':
        map_lbubstd = map_lbubstd_to_cls_simple
        map_durstd = map_durstd_to_cls_simple
    elif std_cls_type == 'full':
        map_lbubstd = map_std_to_cls_full
        map_durstd = map_std_to_cls_full
    else:
        raise ValueError(f"Wrong std cls type: {std_cls_type}")
    lb_std = map_lbubstd[pevent[2]] if label_type=='probabilistic' else 0
    ub_std = map_lbubstd[pevent[3]] if label_type=='probabilistic' else 0
    d_std = map_durstd[pevent[4]] if label_type=='probabilistic' else 0

    return [lb_ind, ub_ind, prob_ind, lb_v, ub_v, lb_std, ub_std, d_std]

