#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np


def gen_golden_data_simple():
    input_x = np.random.uniform(-1, 1, [8, 2048]).astype(np.float32)

    exp = np.exp(input_x)
    sum_ = np.sum(exp, axis=-1).reshape(-1, 1)
    # sum_inv = 1.0 / sum_
    # golden = (exp * sum_inv).astype(np.float32)
    golden = (exp / sum_).astype(np.float32)
    # golden = (exp).astype(np.float32)

    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
