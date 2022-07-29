#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import importlib
import os
import sys


def get_exp_by_file(exp_file,is_binary_backbone=False,is_binary_head=False,clip_grad=False,data_dir=None):
    try:
        sys.path.append(os.path.dirname(exp_file))
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        exp = current_exp.Exp(is_binary_backbone=is_binary_backbone,is_binary_head=is_binary_head,clip_grad=clip_grad,data_dir=data_dir)
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp


def get_exp_by_name(exp_name,is_binary_backbone=False,is_binary_head=False,clip_grad=False,data_dir=None):
    exp = exp_name.replace("-", "_")  # convert string like "yolox-s" to "yolox_s"
    module_name = ".".join(["yolox", "exp", "default", exp])
    exp_object = importlib.import_module(module_name).Exp(is_binary_backbone=is_binary_backbone,is_binary_head=is_binary_head,clip_grad=clip_grad,data_dir=data_dir)
    return exp_object


def get_exp(exp_file=None, exp_name=None,is_binary_backbone=False,is_binary_head=False,clip_grad=False,data_dir=None):
    """
    get Exp object by file or name. If exp_file and exp_name
    are both provided, get Exp by exp_file.

    Args:
        exp_file (str): file path of experiment.
        exp_name (str): name of experiment. "yolo-s",
    """
    print("build.py  is_binary_backbone : ",is_binary_backbone )
    print("build.py  is_binary_head : ",is_binary_head )
    print("build.py  clip_grad : ",clip_grad )
    assert (
        exp_file is not None or exp_name is not None
    ), "plz provide exp file or exp name."
    if exp_file is not None:
        return get_exp_by_file(exp_file,is_binary_backbone=is_binary_backbone,is_binary_head=is_binary_head,clip_grad=clip_grad,data_dir=data_dir)
    else:
        return get_exp_by_name(exp_name,is_binary_backbone=is_binary_backbone,is_binary_head=is_binary_head,clip_grad=clip_grad,data_dir=data_dir)
