/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <cuda.h>
#include <cassert>
#include "conv.h"

#define ROWS_BLOCKDIM_X       16
#define COLUMNS_BLOCKDIM_X    16
#define ROWS_BLOCKDIM_Y       4
#define COLUMNS_BLOCKDIM_Y    8
#define ROWS_RESULT_STEPS     8
#define COLUMNS_RESULT_STEPS  8
#define ROWS_HALO_STEPS       1
#define COLUMNS_HALO_STEPS    1

__global__ void conv_rows(
    float *__restrict dst,
    const float *__restrict src,
    const float *__restrict kernel,
    const int imageW,
    const int imageH,
    const int pitch)
{
  __shared__ float l_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

  int gidX = blockIdx.x;
  int gidY = blockIdx.y;
  int lidX = threadIdx.x;
  int lidY = threadIdx.y;
  //Offset to the left halo edge
  const int baseX = (gidX * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + lidX;
  const int baseY = gidY * ROWS_BLOCKDIM_Y + lidY;

  src += baseY * pitch + baseX;
  dst += baseY * pitch + baseX;

  //Load main data
  for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X] = src[i * ROWS_BLOCKDIM_X];

  //Load left halo
  for(int i = 0; i < ROWS_HALO_STEPS; i++)
    l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X]  = (baseX + i * ROWS_BLOCKDIM_X >= 0) ? src[i * ROWS_BLOCKDIM_X] : 0;

  //Load right halo
  for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
    l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X]  = (baseX + i * ROWS_BLOCKDIM_X < imageW) ? src[i * ROWS_BLOCKDIM_X] : 0;

  //Compute and store results
  __syncthreads();
  for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++){
    float sum = 0;

    for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
      sum += kernel[KERNEL_RADIUS - j] * l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X + j];

    dst[i * ROWS_BLOCKDIM_X] = sum;
  }
}

__global__ void conv_cols(
    float *__restrict dst,
    const float *__restrict src,
    const float *__restrict kernel,
    const int imageW,
    const int imageH,
    const int pitch)
{
  __shared__ float l_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];


  int gidX = blockIdx.x;
  int gidY = blockIdx.y;
  int lidX = threadIdx.x;
  int lidY = threadIdx.y;

  //Offset to the upper halo edge
  const int baseX = gidX * COLUMNS_BLOCKDIM_X + lidX;
  const int baseY = (gidY * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + lidY;
  src += baseY * pitch + baseX;
  dst += baseY * pitch + baseX;

  //Load main data
  for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y] = src[i * COLUMNS_BLOCKDIM_Y * pitch];

  //Load upper halo
  for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
    l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y >= 0) ? src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

  //Load lower halo
  for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
    l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y]  = (baseY + i * COLUMNS_BLOCKDIM_Y < imageH) ? src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

  //Compute and store results
  __syncthreads();
  for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
    float sum = 0;

    for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
      sum += kernel[KERNEL_RADIUS - j] * l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y + j];

    dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
  }
}

void convolutionRows(
    float* dst,
    const float* src,
    const float* kernel,
    const unsigned int imageW,
    const unsigned int imageH,
    const unsigned int pitch)
{
  assert ( ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS );
  assert ( imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 );
  assert ( imageH % ROWS_BLOCKDIM_Y == 0 );

  dim3 block (ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
  dim3 grid (imageW / ROWS_RESULT_STEPS / ROWS_BLOCKDIM_X, imageH/ROWS_BLOCKDIM_Y );

  conv_rows<<<grid, block>>>(
      dst,
      src,
      kernel,
      imageW,
      imageH,
      imageW );

}

void convolutionColumns(
    float* dst,
    const float* src,
    const float* kernel,
    const unsigned int imageW,
    const unsigned int imageH,
    const unsigned int pitch)
{
  assert ( COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS );
  assert ( imageW % COLUMNS_BLOCKDIM_X == 0 );
  assert ( imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0 );

  dim3 block (COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
  dim3 grid (imageW / COLUMNS_BLOCKDIM_X, imageH / COLUMNS_RESULT_STEPS / COLUMNS_BLOCKDIM_Y);

  conv_cols<<<grid, block>>>(
      dst,
      src,
      kernel,
      imageW,
      imageH,
      imageW );
}
