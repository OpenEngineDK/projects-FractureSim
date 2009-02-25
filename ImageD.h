#ifndef IMAGED
#define IMAGED

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "cutil.h"
//#include "GridD.h"

//typedef GridD<float> ImageD;


//---------------------------------------------------------------------------

class ImageD
{
private:

protected:

public:
  float *data;
  int3 dim;
   float3 scale;
  float3 origin;


  ~ImageD();

  float *getData(void);
  __device__ __host__ float getElement(int x,int y,int z);
  __device__ __host__ bool setElement(int x,int y,int z,float element);
  ImageD *getSlice(int z);
  __device__ __host__ float & operator[] (int i);
  __device__ __host__ float operator[] (int i) const;
  ImageD & operator = (const ImageD &);
  __device__ __host__ float interpolate(float3 pos);
  __device__ __host__ int getIndexOf(int3 pos) {return dim.x * dim.y * pos.z + dim.x * pos.y + pos.x;}
  __device__ __host__ float interpolatePhysicalCoords(float3 pos);
};




__device__ float ImageD::interpolate(float3 pos)
{
  float value = 10; // an outside value
  int indexX = (int) floor(pos.x);
  int indexY = (int) floor(pos.y);
  int indexZ = (int) floor(pos.z);
  float deltaX = pos.x-indexX;
  float deltaY = pos.y-indexY;
  float deltaZ= pos.z-indexZ;

  if (pos.x>=0 && pos.x<dim.x-1 && pos.y>=0 && pos.y<dim.y-1 && pos.z>=0 && pos.z<dim.z-1 )
  {
	int offsetDOWN = dim.x*dim.y*(indexZ);
    float valueDOWN = (1-deltaX)*(1-deltaY)*data[indexY*dim.x+indexX+offsetDOWN]+
      deltaX*(1-deltaY)*data[indexY*dim.x+indexX+1+offsetDOWN]+
      deltaX*deltaY*data[(indexY+1)*dim.x+indexX+1+offsetDOWN]+
      (1-deltaX)*deltaY*data[(indexY+1)*dim.x+indexX+offsetDOWN];

	int offsetUP= dim.x*dim.y*(indexZ+1);
    float valueUP = (1-deltaX)*(1-deltaY)*data[indexY*dim.x+indexX+offsetUP]+
      deltaX*(1-deltaY)*data[indexY*dim.x+indexX+1+offsetUP]+
      deltaX*deltaY*data[(indexY+1)*dim.x+indexX+1+offsetUP]+
      (1-deltaX)*deltaY*data[(indexY+1)*dim.x+indexX+offsetUP];

	value = (1-deltaZ)*valueDOWN + deltaZ*valueUP;  
	
	
	//XXXXXXXXXXXXXXXXXXX added a minus!!!!
  }
  return value; 
}

//---------------------------------------------------------------------------

ImageD::~ImageD()
{
	printf("trouble");
 //free(data);
}

//---------------------------------------------------------------------------

float *ImageD::getData(void)
{
  return data;
}

//---------------------------------------------------------------------------

float ImageD::getElement(int x,int y,int z)
{
//remove this line for more speed XXXXXXXXXXXXXXXXXXXXXXX
	if (x>=0 && x<dim.x && y>=0 && y<dim.y && z>=0 && z<dim.z )
		return data[z*dim.x*dim.y+y*dim.x+x];
	else 
		return  0;
}

//---------------------------------------------------------------------------


__device__ bool ImageD::setElement(int x,int y,int z,float element)
{
//remove this line for more speed XXXXXXXXXXXXXXXXXXXXXXX
  if (x<dim.x && y<dim.y && z<dim.z)
	data[z*dim.x*dim.y+y*dim.x+x]=element;

  return true;
}

//---------------------------------------------------------------------------


//---------------------------------------------------------------------------


__device__ float & ImageD::operator[] (int i)
{
  return data[i];
}

//---------------------------------------------------------------------------


__device__ float ImageD::operator[] (int i) const
{
  return data[i];
}

//---------------------------------------------------------------------------

ImageD & ImageD::operator= (const ImageD &image)
{
  //if (&data == this) return *this;

  this->~ImageD();

  dim = image.dim;

  if (data!=NULL) free(data);

  data = (float *) malloc(dim.x*dim.y*dim.z*sizeof(float));

  for (int i = 0; i < dim.x*dim.y*dim.z; i++)
    data[i] = image.data[i];

  return *this;
}

//---------------------------------------------------------------------------

__device__ float ImageD::interpolatePhysicalCoords(float3 pos)
{
	float x,y,z;
	x = (pos.x-origin.x)/scale.x;
	y = (pos.y-origin.y)/scale.y;
	z = (pos.z-origin.z)/scale.z;
	return interpolate(make_float3(x,y,z));
}

#endif
