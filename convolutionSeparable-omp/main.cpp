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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "conv.h"

int main(int argc, char **argv)
{
  const unsigned int imageW = 3072;
  const unsigned int imageH = 3072;

  float* h_Kernel    = (float*)malloc(KERNEL_LENGTH * sizeof(float));
  float* h_Input     = (float*)malloc(imageW * imageH * sizeof(float));
  float* h_Buffer    = (float*)malloc(imageW * imageH * sizeof(float));
  float* h_OutputCPU = (float*)malloc(imageW * imageH * sizeof(float));
  float* h_OutputGPU = (float*)malloc(imageW * imageH * sizeof(float));

  srand(2009);
  for(unsigned int i = 0; i < KERNEL_LENGTH; i++)
    h_Kernel[i] = (float)(rand() % 16);

  for(unsigned int i = 0; i < imageW * imageH; i++)
    h_Input[i] = (float)(rand() % 16);

#pragma omp target data map(to:h_Kernel[0:KERNEL_LENGTH], h_Input[0:imageW*imageH]) \
                        map(alloc: h_Buffer[0:imageW*imageH]) \
                        map(from: h_OutputGPU[0:imageW*imageH]) 
  {

    //Just a single run or a warmup iteration
    convolutionRows(
        h_Buffer,
        h_Input,
        h_Kernel,
        imageW,
        imageH,
        imageW);

    convolutionColumns(
        h_OutputGPU,
        h_Buffer,
        h_Kernel,
        imageW,
        imageH,
        imageW);

    const int numIterations = 0;

    for(int iter = 0; iter < numIterations; iter++){
      convolutionRows(
          h_Buffer,
          h_Input,
          h_Kernel,
          imageW,
          imageH,
          imageW);

      convolutionColumns(
          h_OutputGPU,
          h_Buffer,
          h_Kernel,
          imageW,
          imageH,
          imageW);
    }
  }


  printf("Comparing against Host/C++ computation...\n"); 
  convolutionRowHost(h_Buffer, h_Input, h_Kernel, imageW, imageH, KERNEL_RADIUS);
  convolutionColumnHost(h_OutputCPU, h_Buffer, h_Kernel, imageW, imageH, KERNEL_RADIUS);
  double sum = 0, delta = 0;
  double L2norm;
  for(unsigned int i = 0; i < imageW * imageH; i++){
    delta += (h_OutputCPU[i] - h_OutputGPU[i]) * (h_OutputCPU[i] - h_OutputGPU[i]);
    sum += h_OutputCPU[i] * h_OutputCPU[i];
  }
  L2norm = sqrt(delta / sum);
  printf("Relative L2 norm: %.3e\n\n", L2norm);

  free(h_OutputGPU);
  free(h_OutputCPU);
  free(h_Buffer);
  free(h_Input);
  free(h_Kernel);

  if (L2norm < 1e-6)
    printf("PASS\n"); 
  else
    printf("FAIL\n");

  return 0;
}
