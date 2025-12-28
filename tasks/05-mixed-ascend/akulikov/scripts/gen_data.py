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
import os


def gen_golden_data():
    M = 64 * 14
    N = 64 * 16
    K = 256

    np.random.seed(4)
    input_a = np.random.uniform(-1, 1, [M, K]).astype(np.float16)
    input_b = np.random.uniform(-1, 1, [K, N]).astype(np.float16)
    input_x = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32))).astype(np.float32)
    
    exp = np.exp(input_x)
    sum_ = np.sum(exp, axis=-1).reshape(-1, 1)
    golden = (exp / sum_).astype(np.float32)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_a.tofile("./input/x1_gm.bin")
    input_b.tofile("./input/x2_gm.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
