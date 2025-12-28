/**
 * @file matmul_leakyrelu_custom_tiling.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <cassert>
#include <fstream>
#include <iostream>

#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
using namespace matmul_tiling;
using namespace std;

/**
 * @brief  Generate matmul tiling.
 * @param  socVersion: Platform socversion.
 * @param  tilingBuf data buffer.
 */
void GenerateTiling(const char* socVersion, uint8_t* tilingBuf) {
  int M = 64 * 14;
  int N = 64 * 16;
  int K = 256;

  TPosition leftPosition = TPosition::GM;
  CubeFormat leftFormat = CubeFormat::ND;
  DataType leftDtype = DataType::DT_FLOAT16;
  bool isTransA = false;

  TPosition rightPosition = TPosition::GM;
  CubeFormat rightFormat = CubeFormat::ND;
  DataType rightDtype = DataType::DT_FLOAT16;
  bool isTransB = false;

  TPosition resultPosition = TPosition::GM;
  CubeFormat resultFormat = CubeFormat::ND;
  DataType resultDtype = DataType::DT_FLOAT;

  TPosition biasPosition = TPosition::GM;
  CubeFormat biasFormat = CubeFormat::ND;
  DataType biasDtype = DataType::DT_FLOAT;
  bool isBias = false;

  int usedCoreNum = 2;
  int baseM = 64;
  int baseN = 64;

  optiling::TCubeTiling tilingData;
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
  MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

  tilingApi.SetDim(usedCoreNum);  // Set the number of cores that participate in
                                  // multi-core computaion is 2.
  tilingApi.SetAType(leftPosition, leftFormat, leftDtype, isTransA);
  tilingApi.SetBType(rightPosition, rightFormat, rightDtype, isTransB);
  tilingApi.SetCType(resultPosition, resultFormat, resultDtype);
  tilingApi.SetBiasType(biasPosition, biasFormat, biasDtype);

  tilingApi.SetOrgShape(M, N, K);
  tilingApi.SetShape(M, N, K);
  tilingApi.SetSingleShape(M / usedCoreNum, N, K);
  tilingApi.SetBias(isBias);
  tilingApi.SetTraverse(
      MatrixTraverse::FIRSTN);  // Set the matmul travse is FIRSTM.
  tilingApi.SetFixSplit(baseM, baseN,
                        -1);  // Set the fixed baseM=128, baseN=256.
  tilingApi.SetBufferSpace(-1, -1, -1);

  int64_t res = tilingApi.GetTiling(tilingData);  // Get matmul tiling data.
  tilingData.set_stepM(1);  // Set the matmul tiling stepM=1.
  tilingData.set_stepN(1);  // Set the matmul tiling stepN=1.
  if (res == -1) {
    std::cout << "gen tiling failed" << std::endl;
  }
  uint32_t tcubeTilingSize = tilingData.GetDataSize();
  tilingData.SaveToBuffer(tilingBuf, tcubeTilingSize);
  return;
}
