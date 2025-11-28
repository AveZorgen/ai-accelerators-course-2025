/**
 * @file softmax_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"

constexpr int32_t TOTAL_LENGTH = 8 * 2048;  // total length of data
constexpr int32_t USE_CORE_NUM = 8;         // num of core used
// length computed of each core
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;
constexpr int32_t TILE_NUM = 8;    // split data into 8 tiles for each core
constexpr int32_t BUFFER_NUM = 2;  // tensor num for each queue
constexpr int32_t LOOP_COUNT = TILE_NUM * BUFFER_NUM;
// separate to 2 parts, due to double buffer
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / LOOP_COUNT;

template <typename T>
class KernelAdd {
 public:
  __aicore__ inline KernelAdd() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR z) {
    xGm.SetGlobalBuffer((__gm__ T*)x + BLOCK_LENGTH * AscendC::GetBlockIdx(),
                        BLOCK_LENGTH);
    zGm.SetGlobalBuffer((__gm__ T*)z + BLOCK_LENGTH * AscendC::GetBlockIdx(),
                        BLOCK_LENGTH);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(T));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(T));
    pipe.InitBuffer(tmpQueue, (1 + TILE_LENGTH) * sizeof(T));
  }
  __aicore__ inline void Process() {
    AscendC::LocalTensor<T> tmpLocal = tmpQueue.Get<T>();
    T scalar(0);
    AscendC::Duplicate(tmpLocal, scalar, TILE_LENGTH);

    for (int32_t i = 0; i < LOOP_COUNT; i++) {
      CopyIn(i, xGm);
      Compute(i, tmpLocal);
      CopyOut(i);
    }
    AscendC::LocalTensor<T> tmpLocal2 = tmpLocal[TILE_LENGTH];

    // AscendC::SumParams sumParams = {
    //     1, (TILE_LENGTH * sizeof(T) + 32 - 1) / 32 * 32 / sizeof(T),
    //     TILE_LENGTH};
    // AscendC::Sum(tmpLocal2, tmpLocal, sumParams);

    const uint32_t shape[] = {1, TILE_LENGTH};
    AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR>(tmpLocal2, tmpLocal,
                                                        shape, true);

    AscendC::Duplicate(tmpLocal, tmpLocal2.GetValue(0), TILE_LENGTH);
    // AscendC::Duplicate(tmpLocal, 1 / tmpLocal2.GetValue(0), TILE_LENGTH);

    for (int32_t i = 0; i < LOOP_COUNT; i++) {
      CopyIn(i, zGm);
      Compute2(i, tmpLocal);
      CopyOut(i);
    }
  }

 private:
  __aicore__ inline void CopyIn(int32_t progress,
                                const AscendC::GlobalTensor<T>& srcGlobal) {
    AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, srcGlobal[progress * TILE_LENGTH], TILE_LENGTH);
    inQueueX.EnQue(xLocal);
  }
  __aicore__ inline void Compute(int32_t progress,
                                 AscendC::LocalTensor<T>& tmpLocal) {
    AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
    AscendC::Exp(zLocal, xLocal, TILE_LENGTH);
    AscendC::Add(tmpLocal, tmpLocal, zLocal, TILE_LENGTH);
    outQueueZ.EnQue<T>(zLocal);
    inQueueX.FreeTensor(xLocal);
  }
  __aicore__ inline void Compute2(int32_t progress,
                                  AscendC::LocalTensor<T>& tmpLocal) {
    AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
    AscendC::Div(zLocal, xLocal, tmpLocal, TILE_LENGTH);
    // AscendC::Mul(zLocal, xLocal, tmpLocal, TILE_LENGTH);
    outQueueZ.EnQue<T>(zLocal);
    inQueueX.FreeTensor(xLocal);
  }
  __aicore__ inline void CopyOut(int32_t progress) {
    AscendC::LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
    AscendC::DataCopy(zGm[progress * TILE_LENGTH], zLocal, TILE_LENGTH);
    outQueueZ.FreeTensor(zLocal);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
  AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
  AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue;
  AscendC::GlobalTensor<T> xGm;
  AscendC::GlobalTensor<T> zGm;
};

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR z) {
  KernelAdd<float> op;
  op.Init(x, z);
  op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void softmax_custom_do(uint32_t blockDim, void* stream, uint8_t* x,
                       uint8_t* z) {
  softmax_custom<<<blockDim, nullptr, stream>>>(x, z);
}
#endif
