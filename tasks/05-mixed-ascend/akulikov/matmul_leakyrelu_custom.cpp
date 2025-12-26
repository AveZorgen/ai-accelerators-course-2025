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
  AscendC::printf("computeRound:%d:%d\n", tmpLocal.GetSize(),
                  tmpLocal2.GetSize());
  AscendC::Duplicate(tmpLocal, cType(0), tiling.baseM * tiling.baseN);
  while (matmulObj.template Iterate<true>()) {  // Once Iterate, compute baseM *
                                                // baseN, sync is set true here.
    reluOutLocal = reluOutQueue_.AllocTensor<cType>();
    matmulObj.template GetTensorC<true>(reluOutLocal, false, true);
    AscendC::Add(tmpLocal, tmpLocal, reluOutLocal, tiling.baseM * tiling.baseN);
    reluOutQueue_.EnQue(reluOutLocal);
    CopyOut(computeRound);
    if ((computeRound + 1) % (tiling.singleCoreN / tiling.baseN) == 0) {
      AscendC::SumParams sumParams = {tiling.baseM, tiling.baseN, tiling.baseN};
      AscendC::Sum(tmpLocal2, tmpLocal, sumParams);
      AscendC::printf("computeRound:%d:%d:%lf\n", AscendC::GetBlockIdx(),
                      computeRound, tmpLocal2.GetValue(0));
      for (int i = 0; i < tiling.singleCoreN; i++) {
        // load [baseM, 1] -> exp -> div by [baseM, 1]
      }
      /* or */
      for (int i = 0; i < tiling.singleCoreN / tiling.baseN; i++) {
        // load [baseM, baseN] -> exp -> div by [baseM, 1]
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
  const uint32_t roundM = tiling.singleCoreM / tiling.baseM;
  const uint32_t roundN = tiling.singleCoreN / tiling.baseN;
  uint32_t startOffset = (count / roundN * tiling.baseM * tiling.N +
                          count % roundN * tiling.baseN);

  AscendC::DataCopyParams copyParam = {
      (uint16_t)tiling.baseM,
      (uint16_t)(tiling.baseN * sizeof(cType) / AscendC::DEFAULT_C0_SIZE), 0,
      (uint16_t)((tiling.N - tiling.baseN) * sizeof(cType) /
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
