#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self, is_binary_backbone=False, is_binary_head=False, clip_grad=False, data_dir=None,scheduler=None):
        super(Exp, self).__init__(is_binary_backbone=is_binary_backbone, is_binary_head=is_binary_head,
                                  clip_grad=clip_grad, data_dir=data_dir,scheduler=scheduler)
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
