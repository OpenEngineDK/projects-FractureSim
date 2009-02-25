#ifndef __IMAGED_CU__
#define __IMAGED_CU__

//---------------------------------------------------------------------------

#include "ImageD.h"

//---------------------------------------------------------------------------

//inline 

__host__ float *newCUDAImageF(ImageD **maskd, ImageD *maskh, bool copy_data = true)
{
  float * datapos;
//  printf(cudaGetErrorString(cudaMalloc((void **)&datapos, maskh->dim.x*maskh->dim.y*maskh->dim.z*sizeof(float))));
//  printf(cudaGetErrorString(cudaMalloc((void **)maskd, sizeof(ImageD))));
//  printf("newCUDAImageF: %i \n",maskh->dim.x);

  CUDA_SAFE_CALL(cudaMalloc((void **)&datapos, maskh->dim.x*maskh->dim.y*maskh->dim.z*sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void **)maskd, sizeof(ImageD)));

  //cudaGetErrorString(cudaMalloc((void **)&datapos, maskh->dim.x*maskh->dim.y*maskh->dim.z*sizeof(float)));
  //cudaGetErrorString(cudaMalloc((void **)maskd, sizeof(ImageD)));

 // copy image data+dimension
  CUDA_SAFE_CALL(cudaMemcpy(*maskd, maskh, sizeof(ImageD), cudaMemcpyHostToDevice)); 

 CUDA_SAFE_CALL(cudaMemcpy(*maskd, &datapos, sizeof(float*), cudaMemcpyHostToDevice)); 
// printf("datapos: %i\n", datapos);
// assert(datapos != 0);
// assert(datapos == (*maskd)->data);

// printf("newCUDAImageF: %i == %i \n",datapos,(*maskd)->data);
 
  // copy data
  if (copy_data)
    CUDA_SAFE_CALL(cudaMemcpy(datapos, maskh->data, maskh->dim.x*maskh->dim.y*maskh->dim.z*sizeof(float), cudaMemcpyHostToDevice)); 
// printf("newCUDAImageF: %i \n",maskh->dim.x);
// printf("newCUDAImageF: %i \n",(*maskd)->dim.x);

  return datapos;
}



//---------------------------------------------------------------------------

ImageD * newImage(int3 dim, float3 scale, float3 origin, bool allocate_space)
{
  ImageD *res = (ImageD *)malloc(sizeof(ImageD));
  res->dim = dim;
  res->origin = origin;
  res->scale = scale;

  if (allocate_space)
  {
	 res->data = (float*) malloc(res->dim.x*res->dim.y*res->dim.z*sizeof(float));

	 for(int i=0;i<res->dim.x*res->dim.y*res->dim.z;i++) res->data[i]=0;
  }
  return res;
}

//---------------------------------------------------------------------------

void freeCUDAImage(ImageD* img)
{
	CUDA_SAFE_CALL(cudaFree(img->data));
	CUDA_SAFE_CALL(cudaFree(img));
}

//---------------------------------------------------------------------------

void freeImage(ImageD* img)
{
	free(img->data);
	free(img);
}

//---------------------------------------------------------------------------

#endif /* __IMAGED_CU__ */

