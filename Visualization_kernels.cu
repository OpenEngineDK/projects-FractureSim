#include <stdio.h>
#include <stdlib.h>
#include "TetrahedralMesh.h"
//#include "cutil_math.h"
#include "float_utils.h"
#include "VisualShapes.h"


/**
 * This kernel function applies the matrix transformation to each
 * vertex in the polygon model. Start as many threats as there are
 * vertices in the model.
 *
 * @param vert is the array of vertices in the polygon model
 * @param mat is the array of 4 x float4 matrices.
 */
__global__
void applyTransformation_k(float4* vert, float4* mat, unsigned int numVerts) {
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=numVerts)
		return;
    
    vert[me_idx].x = vert[me_idx].x + mat[0].x;
}


void applyTransformation(VisualBuffer& vb) {
    unsigned int gridSize = vb.numElm;
    unsigned int blockSize = (float)vb.byteSize / sizeof(float4) / (float)vb.numElm;
    unsigned int blockSizeCeil = (int)ceil((float)blockSize/128.0f);
    
    //printf("Grid: %i  block: %i blockCeil: %i\n", gridSize, blockSize, blockSizeCeil);
    applyTransformation_k<<<make_uint3(gridSize,1,1), make_uint3(blockSizeCeil,1,1)>>>(vb.bufExt, vb.buf, blockSize);
    CUT_CHECK_ERROR("Error applying transformations");
}


__global__ void
extractSurface_k(float3 *tris, Tetrahedron *tetrahedra, Point *points, unsigned int numTets)
{
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=numTets)
		return;

	int zeroVertices[4];
	int numZeroVertices = 0;

	Tetrahedron tet = tetrahedra[me_idx];

	if (tet.x<0) return; // illegal tetrahedron

//	if (points[tet.x].distance==0)
			zeroVertices[numZeroVertices++] = tet.x;
//	if (points[tet.y].distance==0)
			zeroVertices[numZeroVertices++] = tet.y;
//	if (points[tet.z].distance==0)
			zeroVertices[numZeroVertices++] = tet.z;
//	if (points[tet.w].distance==0)
			zeroVertices[numZeroVertices++] = tet.w;

//	printf("numZeroes: %i", numZeroVertices);

	if (numZeroVertices>=3 )
	{
		for (int i=0; i<3; i++)
			tris[(3*me_idx)+i] = crop_last_dim(points[zeroVertices[i]]);
	}
	else
	{
		for (int i=0; i<3; i++)
			tris[(3*me_idx)+i] = make_float3(0,0,0);
	}
}


__global__ void
extractSurfaceWithDisplacements_k(float3 *tris, Tetrahedron *tetrahedra, Point *points, float4 *displacements, unsigned int numTets)
{
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=numTets)
		return;

	int4 tet = tetrahedra[me_idx];

	if (tet.x<0) return; // illegal tetrahedron

	int zeroVertices[4];
	int numZeroVertices = 0;


//	if (points[tetrahedra[me_idx].x].distance==0)
			zeroVertices[numZeroVertices++] = tet.x;
//	if (points[tetrahedra[me_idx].y].distance==0)
			zeroVertices[numZeroVertices++] = tet.y;
//	if (points[tetrahedra[me_idx].z].distance==0)
			zeroVertices[numZeroVertices++] = tet.z;
//	if (points[tetrahedra[me_idx].w].distance==0)
			zeroVertices[numZeroVertices++] = tet.w;

	//	printf("numZeroes: %i", numZeroVertices);

	if (numZeroVertices>=3)
	{
		for (int i=0; i<3; i++)
		{
			float3 pos = crop_last_dim(points[zeroVertices[i]]);
			float3 displacement = crop_last_dim(displacements[zeroVertices[i]]);
			pos.x += displacement.x;  
			pos.y += displacement.y;  
			pos.z += displacement.z;  
			tris[(3*me_idx)+i] = pos;
		}

	}
	else
	{
		for (int i=0; i<3; i++)
			tris[(3*me_idx)+i] = make_float3(0,0,0);
	}
}

// cross product
inline __host__ __device__ float3 cross(float3 a, float3 b)
{ 
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

__device__
float3 calcNormal(float4 *v0, float4 *v1, float4 *v2)
{
    float3 edge0 = crop_last_dim(*v1 - *v0);

    float3 edge1 = crop_last_dim(*v2 - *v0);

    // note - it's faster to perform normalization in vertex shader rather than here
    return cross(edge0, edge1);
}


__global__ void
updateSurfacePositionsFromDisplacements_k(float3 *tris, float3 *normals, TriangleSurface surface, Point *points, float4 *displacements)
{
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=surface.numFaces) return;

	Triangle triangle = surface.faces[me_idx];

	float4 pos, pos2, pos3, displacement;

	pos = points[triangle.x-1];
	displacement = displacements[triangle.x-1];
	pos.x += displacement.x;  
	pos.y += displacement.y;  
	pos.z += displacement.z;  
	tris[(3*me_idx)+0] = crop_last_dim(pos);

	pos2 = points[triangle.y-1];
	displacement = displacements[triangle.y-1];
	pos2.x += displacement.x;  
	pos2.y += displacement.y;  
	pos2.z += displacement.z;  
	tris[(3*me_idx)+1] = crop_last_dim(pos2);

	pos3 = points[triangle.z-1];
	displacement = displacements[triangle.z-1];
	pos3.x += displacement.x;  
	pos3.y += displacement.y;  
	pos3.z += displacement.z;  
	tris[(3*me_idx)+2] = crop_last_dim(pos3);

	float3 normal = calcNormal(&pos,&pos2,&pos3);

	normals[(3*me_idx)+0] = normal;
	normals[(3*me_idx)+1] = normal;
	normals[(3*me_idx)+2] = normal;
}

__global__ void
updateMeshCentersFromDisplacements_k(float3 *centers,
                                     TetrahedralMesh mesh, float4 *displacements)
{
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=mesh.numTetrahedra) return;

    Tetrahedron tetra = mesh.tetrahedra[me_idx];

	float4 pos0, pos1, pos2, pos3;

    float4* points = mesh.points;

    pos0 = points[tetra.x] + displacements[tetra.x];
    pos1 = points[tetra.y] + displacements[tetra.y];
    pos2 = points[tetra.z] + displacements[tetra.z];
    pos3 = points[tetra.w] + displacements[tetra.w];

    float4 center = (pos0 + pos1 + pos2 + pos3) / 4.0;

    centers[me_idx] = crop_last_dim(center);
    //centers[me_idx*2+1] = crop_last_dim(center);
}


__global__ void
updateMeshCentersFromDisplacements2_k(float4* buffer,
                                      TetrahedralMesh mesh, float4 *displacements)
{
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=mesh.numTetrahedra) return;

    Tetrahedron tetra = mesh.tetrahedra[me_idx];

	float4 pos0, pos1, pos2, pos3;

    float4* points = mesh.points;

    pos0 = points[tetra.x] + displacements[tetra.x];
    pos1 = points[tetra.y] + displacements[tetra.y];
    pos2 = points[tetra.z] + displacements[tetra.z];
    pos3 = points[tetra.w] + displacements[tetra.w];

    float4 center = (pos0 + pos1 + pos2 + pos3) / 4.0;

    //PointShape p(center);
    //p.CopyToBuf(bufArray, me_idx);

    /*    float4 pos = center;
    float4 dir = {0, 1.0, 0, 0};
    VectorShape v(pos, pos + dir);
    v.CopyToBuf(buffer, me_idx);
    */

    PolyShape p;
    p.CopyToBuf(buffer, me_idx);
}

