/**
 * @file matmul_leakyrelu_custom.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b) {
  return (a + b - 1) / b;
}

/**
 * @brief  Copy tiling data to TCubeTiling ptr from tiling gm addr.
 * @param  tiling: TCubeTiling ptr which needs to copy tiling data.
 * @param  tilingGM: tiling gm addr.
 * @retval None
 */
__aicore__ inline void CopyTiling(TCubeTiling* tiling, GM_ADDR tilingGM) {
  uint32_t* ptr = reinterpret_cast<uint32_t*>(tiling);
  auto tiling32 = reinterpret_cast<__gm__ uint32_t*>(tilingGM);

  for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
    *ptr = *(tiling32 + i);
  }
  return;
}

template <typename aType, typename bType, typename cType, typename biasType>
class MatmulLeakyKernel {
 public:
  __aicore__ inline MatmulLeakyKernel(){};
  __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                              GM_ADDR workspace, const TCubeTiling& tiling,
                              AscendC::TPipe* pipe);
  __aicore__ inline void Process(AscendC::TPipe* pipe);

  __aicore__ inline void MatmulCompute();
  __aicore__ inline void LeakyReluCompute();
  __aicore__ inline void CopyOut(uint32_t count);
  __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling& tiling,
                                    int32_t& offsetA, int32_t& offsetB,
                                    int32_t& offsetC, int32_t& offsetBias);

  Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
         MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
         MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, cType>,
         MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>>
      matmulObj;

  AscendC::GlobalTensor<aType> aGlobal;
  AscendC::GlobalTensor<bType> bGlobal;
  AscendC::GlobalTensor<cType> cGlobal;
  AscendC::GlobalTensor<biasType> biasGlobal;
  AscendC::LocalTensor<cType> reluOutLocal;
  TCubeTiling tiling;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> reluOutQueue_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue;
  AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueueX, inQueueY;
  AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueueZ;
};

/**
 * @brief  Set matmulLeaky input and output gm addr of current core.
 * @param  a: A matrix gm addr.
 * @param  b: B matrix gm addr.
 * @param  bias: Bias gm addr.
 * @param  c: C matrix gm addr.
 * @param  workspace: Temporary gm space addr required by matmul calc.
 * @param  tiling: matmul tiling data.
 * @param  pipe: Global memory and sync management TPipe object.
 * @retval None
 */
template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace,
    const TCubeTiling& tiling, AscendC::TPipe* pipe) {
  this->tiling = tiling;
  aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType*>(a),
                          tiling.M * tiling.Ka);
  bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType*>(b),
                          tiling.Kb * tiling.N);
  cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType*>(c),
                          tiling.M * tiling.N);

  int32_t offsetA, offsetB, offsetC, offsetBias;
  CalcOffset(AscendC::GetBlockIdx(), tiling, offsetA, offsetB, offsetC,
             offsetBias);  // Calculate the gm offset based on the blockidx.
  aGlobal = aGlobal[offsetA];
  bGlobal = bGlobal[offsetB];
  cGlobal = cGlobal[offsetC];
  pipe->InitBuffer(
      reluOutQueue_, 1,
      tiling.baseM * tiling.baseN * sizeof(cType));  // Init output buffer.

  pipe->InitBuffer(
      tmpQueue, (tiling.baseM + tiling.baseM * tiling.baseN) * sizeof(cType));
  pipe->InitBuffer(inQueueX, 2, tiling.baseM * tiling.baseN * sizeof(cType));
  pipe->InitBuffer(outQueueZ, 2, tiling.baseM * tiling.baseN * sizeof(cType));
}

/**
 * @brief  Main process of matmul calculation
 * @param  pipe: Global memory and sync management TPipe object.
 * @retval None
 */
template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulLeakyKernel<aType, bType, cType, biasType>::Process(
    AscendC::TPipe* pipe) {
  uint32_t computeRound = 0;
  const uint32_t roundN = tiling.singleCoreN / tiling.baseN;

#ifdef CUSTOM_ASCEND310P
  // Set temp UB space when on ASCEND310P
  AscendC::TBuf<> tmpMMFormatUb;
  AscendC::LocalTensor<uint8_t> mmformatUb;
  pipe->InitBuffer(tmpMMFormatUb, tiling.baseM * tiling.baseN * sizeof(cType));
  mmformatUb =
      tmpMMFormatUb.Get<uint8_t>(tiling.baseM * tiling.baseN * sizeof(cType));
  matmulObj.SetLocalWorkspace(mmformatUb);
#endif
  matmulObj.SetTensorA(aGlobal);
  matmulObj.SetTensorB(bGlobal);
  AscendC::LocalTensor<cType> tmpLocalFull = tmpQueue.Get<cType>();
  AscendC::LocalTensor<cType> tmpLocal = tmpLocalFull[0];
  AscendC::LocalTensor<cType> tmpLocal2 =
      tmpLocalFull[tiling.baseM * tiling.baseN];
  AscendC::Duplicate(tmpLocal, cType(0), tiling.baseM * tiling.baseN);

  while (matmulObj.template Iterate<true>()) {  // Once Iterate, compute baseM *
                                                // baseN, sync is set true here.
    reluOutLocal = reluOutQueue_.AllocTensor<cType>();
    matmulObj.template GetTensorC<true>(reluOutLocal, false, true);
    AscendC::Exp(reluOutLocal, reluOutLocal, tiling.baseM * tiling.baseN);
    AscendC::Add(tmpLocal, tmpLocal, reluOutLocal, tiling.baseM * tiling.baseN);
    reluOutQueue_.EnQue(reluOutLocal);

    bool is_last_in_row = ((computeRound + 1) % roundN) == 0;
    if (!is_last_in_row) {
      CopyOut(computeRound);
    }
    if (is_last_in_row) {
      AscendC::SumParams sumParams = {static_cast<uint32_t>(tiling.baseM),
                                      static_cast<uint32_t>(tiling.baseN),
                                      static_cast<uint32_t>(tiling.baseN)};
      AscendC::Sum(tmpLocal2, tmpLocal, sumParams);
      int row = ((computeRound + 1) - roundN) * tiling.baseN * tiling.baseM;

      uint32_t srcShape_[] = {static_cast<uint32_t>(tiling.baseM), 1};
      uint32_t dstShape_[] = {static_cast<uint32_t>(tiling.baseM),
                              static_cast<uint32_t>(tiling.baseN)};
      AscendC::Broadcast<cType, 2, 1>(tmpLocal, tmpLocal2, dstShape_,
                                      srcShape_);

      for (int i = 0; i < roundN; i++) {
        if (i < roundN - 1) {
          AscendC::LocalTensor<cType> xLocal = inQueueX.AllocTensor<cType>();
          AscendC::DataCopyParams repeatParams = {
              static_cast<uint16_t>(tiling.baseM), /* blockCount */
              static_cast<uint16_t>(tiling.baseN * sizeof(cType) /
                                    AscendC::DEFAULT_C0_SIZE), /* blockLen */
              static_cast<uint16_t>((tiling.singleCoreN - tiling.baseN) *
                                    sizeof(cType) /
                                    AscendC::DEFAULT_C0_SIZE), /* srcGap */
              0,                                               /* dstGap */
          };
          AscendC::DataCopy(xLocal, cGlobal[row + i * tiling.baseN],
                            repeatParams);
          inQueueX.EnQue(xLocal);

          AscendC::LocalTensor<cType> xLocal_ = inQueueX.DeQue<cType>();
          AscendC::LocalTensor<cType> zLocal = outQueueZ.AllocTensor<cType>();

          AscendC::Div(zLocal, xLocal_, tmpLocal, tiling.baseM * tiling.baseN);

          outQueueZ.EnQue<cType>(zLocal);
          inQueueX.FreeTensor(xLocal_);

          AscendC::LocalTensor<cType> zLocal_ = outQueueZ.DeQue<cType>();
          AscendC::DataCopyParams repeatParams_ = {
              static_cast<uint16_t>(tiling.baseM), /* blockCount */
              static_cast<uint16_t>(tiling.baseN * sizeof(cType) /
                                    AscendC::DEFAULT_C0_SIZE), /* blockLen */
              0,                                               /* srcGap */
              static_cast<uint16_t>((tiling.singleCoreN - tiling.baseN) *
                                    sizeof(cType) /
                                    AscendC::DEFAULT_C0_SIZE), /* dstGap */
          };
          AscendC::DataCopy(cGlobal[row + i * tiling.baseN], zLocal_,
                            repeatParams_);
          outQueueZ.FreeTensor(zLocal_);
        } else {
          AscendC::LocalTensor<cType> xLocal_ = reluOutQueue_.DeQue<cType>();
          AscendC::Div(xLocal_, xLocal_, tmpLocal, tiling.baseM * tiling.baseN);

          uint32_t startOffset_last =
              (computeRound / roundN * tiling.baseM * tiling.singleCoreN +
               (computeRound % roundN) * tiling.baseN);
          AscendC::DataCopyParams copyParam_last = {
              (uint16_t)tiling.baseM,
              (uint16_t)(tiling.baseN * sizeof(cType) /
                         AscendC::DEFAULT_C0_SIZE),
              0,
              (uint16_t)((tiling.singleCoreN - tiling.baseN) * sizeof(cType) /
                         AscendC::DEFAULT_C0_SIZE)};
          AscendC::DataCopy(cGlobal[startOffset_last], xLocal_, copyParam_last);

          reluOutQueue_.FreeTensor(xLocal_);
        }
      }
      AscendC::Duplicate(tmpLocal, cType(0), tiling.baseM * tiling.baseN);
    }

    computeRound++;
  }
  matmulObj.End();
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulLeakyKernel<aType, bType, cType, biasType>::MatmulCompute() {
  reluOutLocal = reluOutQueue_.AllocTensor<cType>();
  matmulObj.template GetTensorC<true>(reluOutLocal, false, true);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulLeakyKernel<aType, bType, cType, biasType>::LeakyReluCompute() {
  LeakyRelu(reluOutLocal, reluOutLocal, (cType)0.001,
            tiling.baseM * tiling.baseN);
  reluOutQueue_.EnQue(reluOutLocal);
}

/**
 * @brief  Copy leakyRelu out result to GM.
 * @param  count: Iterate count(once Iterate, compute baseM * baseN).
 * @retval None
 */
template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulLeakyKernel<aType, bType, cType, biasType>::CopyOut(uint32_t count) {
  reluOutQueue_.DeQue<cType>();
  const uint32_t roundN = tiling.singleCoreN / tiling.baseN;
  uint32_t startOffset = (count / roundN * tiling.baseM * tiling.singleCoreN +
                          count % roundN * tiling.baseN);

  AscendC::DataCopyParams copyParam = {
      (uint16_t)tiling.baseM,
      (uint16_t)(tiling.baseN * sizeof(cType) / AscendC::DEFAULT_C0_SIZE), 0,
      (uint16_t)((tiling.singleCoreN - tiling.baseN) * sizeof(cType) /
                 AscendC::DEFAULT_C0_SIZE)};
  DataCopy(cGlobal[startOffset], reluOutLocal, copyParam);
  reluOutQueue_.FreeTensor(reluOutLocal);
}

/**
 * @brief  Calculate the gm offset based on the blockidx.
 * @param  blockIdx: Current Core blockidx.
 * @param  tiling: Matmul tiling data.
 * @param  offsetA: Gm offset of A matrix.
 * @param  offsetB: Gm offset of B matrix.
 * @param  offsetC: Gm offset of C matrix.
 * @param  offsetBias: Gm offset of Bias matrix.
 * @retval None
 */
template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulLeakyKernel<aType, bType, cType, biasType>::CalcOffset(
    int32_t blockIdx, const TCubeTiling& tiling, int32_t& offsetA,
    int32_t& offsetB, int32_t& offsetC, int32_t& offsetBias) {
  auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
  auto mCoreIndx = blockIdx % mSingleBlocks;
  auto nCoreIndx = blockIdx / mSingleBlocks;

  offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
  offsetB = nCoreIndx * tiling.singleCoreN;
  offsetC = mCoreIndx * tiling.N * tiling.singleCoreM +
            nCoreIndx * tiling.singleCoreN;
  offsetBias = nCoreIndx * tiling.singleCoreN;
}

/**
 * @brief  matmul_leakyrelu kernel function entry
 * @param  a: A matrix gm addr.
 * @param  b: B matrix gm addr.
 * @param  bias: Bias gm addr.
 * @param  c: Out gm addr.
 * @param  workspace: Temporary gm space addr required by matmul calc.
 * @param  tilingGm: Tiling data addr.
 * @retval None
 */
extern "C" __global__ __aicore__ void matmul_leakyrelu_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tilingGm) {
  AscendC::TPipe pipe;
  TCubeTiling tiling;
  CopyTiling(&tiling, tilingGm);

  MatmulLeakyKernel<half, half, float, float> matmulLeakyKernel;
  matmulLeakyKernel.Init(a, b, c, workspace, tiling, &pipe);
  REGIST_MATMUL_OBJ(
      &pipe, GetSysWorkSpacePtr(), matmulLeakyKernel.matmulObj,
      &matmulLeakyKernel.tiling);  // Initialize the matmul object.
  matmulLeakyKernel.Process(&pipe);
}