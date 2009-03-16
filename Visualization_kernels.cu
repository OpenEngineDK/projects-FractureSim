#define BLOCKSIZE 128

#include <CUDA.h>

#include <stdio.h>
#include <stdlib.h>
#include "TetrahedralMesh.h"

//#include "cutil_math.h"
#include "float_utils.h"
#include "Shapes.h"
#include "VboManager.h"

// cross product
inline __host__ __device__ float4 cross(float4 a, float4 b)
{ 
    return make_float4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0.0); 
}

__device__
float4 calcNormal(float4 *v0, float4 *v1, float4 *v2)
{
    float4 edge0 = *v1 - *v0;

    float4 edge1 = *v2 - *v0;

    // note - it's faster to perform normalization in vertex shader rather than here
    return cross(edge0, edge1);
}


__global__
void applyTransformation_k(float4* model, float4* vert, float4* mat, unsigned int numVerts, unsigned int numThreads) {
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=numVerts)
		return;
    
    vert[me_idx].x = model[me_idx].x  + mat[0].x;
    
}

/**
 * This function applies the matrix transformation to each
 * vertex in the polygon model. Start as many threats as there are
 * vertices in the model.
 */
void applyTransformation(VisualBuffer& vb) {
    unsigned int gridSize = vb.numElm;
    unsigned int blockSize = (float)vb.byteSize / sizeof(float4) / (float)vb.numElm;

    unsigned int blockSizeCeil = (int)ceil((float)blockSize/128.0f);
    unsigned int numVerticesInModel = vb.numIndices / vb.numElm; // this is the number of indices in one model.

    //printf("Grid: %i  block: %i blockCeil: %i\n", gridSize, blockSize, blockSizeCeil);
    //printf("modelAddr: %i - bufAddr: %i\n", vb.modelBuf, vb.buf);
    applyTransformation_k<<<make_uint3(gridSize,1,1), make_uint3(blockSizeCeil,1,1)>>>(vb.modelBuf, vb.buf, vb.matBuf, numVerticesInModel, blockSize);
    CUT_CHECK_ERROR("Error applying transformations");
}


__global__ void
updateSurface_k(float4* vertBuf, float4* normBuf, Surface surface, Point *points, float4 *displacements) {
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=surface.numFaces) return;
    
	Triangle triangle = surface.faces[me_idx];

	float4 pos, pos2, pos3, displacement;

	pos = points[triangle.x];
	displacement = displacements[triangle.x];
	pos.x += displacement.x;  
	pos.y += displacement.y;  
	pos.z += displacement.z;  
    vertBuf[(me_idx*3)+0] = pos;

	pos2 = points[triangle.y];
	displacement = displacements[triangle.y];
	pos2.x += displacement.x;  
	pos2.y += displacement.y;  
	pos2.z += displacement.z;  
    vertBuf[(me_idx*3)+1] = pos2;

	pos3 = points[triangle.z];
	displacement = displacements[triangle.z];
	pos3.x += displacement.x;  
	pos3.y += displacement.y;  
	pos3.z += displacement.z;  
	vertBuf[(me_idx*3)+2] = pos3;
    /*
    float4 normal = calcNormal(&pos,&pos2,&pos3);

	normBuf[(3*me_idx)+0] = normal;
	normBuf[(3*me_idx)+1] = normal;
	normBuf[(3*me_idx)+2] = normal;
    */
}

void updateSurface(Solid* solid, VboManager* vbom) {
	int gridSize = (int)ceil(((float)solid->surface->numFaces)/BLOCKSIZE);

    updateSurface_k<<<make_uint3(gridSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>(vbom->GetBuf(SURFACE_VERTICES).buf, 
                                                                             vbom->GetBuf(SURFACE_VERTICES).buf,
                                                                             *solid->surface, 
                                                                             solid->vertexpool->data, 
                                                                             solid->vertexpool->Ui_t);
}


__global__ void 
updateCenterOfMass_k(float4* buf, Body mesh, Point* points, float4* displacements) {
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=mesh.numTetrahedra) return;

    Tetrahedron tetra = mesh.tetrahedra[me_idx];

	float4 pos0, pos1, pos2, pos3;

    pos0 = points[tetra.x] + displacements[tetra.x];
    pos1 = points[tetra.y] + displacements[tetra.y];
    pos2 = points[tetra.z] + displacements[tetra.z];
    pos3 = points[tetra.w] + displacements[tetra.w];

    float4 center = (pos0 + pos1 + pos2 + pos3) / 4.0;

    buf[me_idx] = center;
}

void updateCenterOfMass(Solid* solid, VboManager* vbom) {
	int gridSize = (int)ceil(((float)solid->body->numTetrahedra)/BLOCKSIZE);

    updateCenterOfMass_k<<<make_uint3(gridSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>(vbom->GetBuf(CENTER_OF_MASS).buf, 
                                                                                  *solid->body, 
                                                                                  solid->vertexpool->data, 
                                                                                  solid->vertexpool->Ui_t);

}

__global__ void 
updateBodyMesh_k(float4* vertBuf, float4* colrBuf, Body mesh, Point* points,
                 float4* displacements, float minX) {
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=mesh.numTetrahedra) return;
    
    Tetrahedron tetra = mesh.tetrahedra[me_idx];

	float4 pos0, pos1, pos2, pos3;

    pos0 = points[tetra.x] + displacements[tetra.x];
    pos1 = points[tetra.y] + displacements[tetra.y];
    pos2 = points[tetra.z] + displacements[tetra.z];
    pos3 = points[tetra.w] + displacements[tetra.w];

	me_idx *= 12;
    int col_idx = me_idx;
    
    /*    if ( pos0.x < minX ||
         pos1.x < minX ||
         pos2.x < minX ||
         pos3.x < minX ) {
        for (unsigned int i=0; i<12; i++) {
            buf[me_idx++] = make_float4(0.0,0.0,0.0,0.0);
            }
            } */
    //else {
    float val = (float)me_idx / (float)mesh.numTetrahedra;
    float4 col = make_float4(val, val, 0.5, 1.0);
    //    float4 col = make_float4(0.2, 0.1, 0.5, 1.0);

    // 0     2     3
    vertBuf[me_idx++] = pos0;
    vertBuf[me_idx++] = pos2;
    vertBuf[me_idx++] = pos3;

    colrBuf[col_idx++] = col;
    colrBuf[col_idx++] = col;
    colrBuf[col_idx++] = col;

    // 0     3     1
    vertBuf[me_idx++] = pos0;
    vertBuf[me_idx++] = pos3;
    vertBuf[me_idx++] = pos1;
    
    colrBuf[col_idx++] = col;
    colrBuf[col_idx++] = col;
    colrBuf[col_idx++] = col;

    // 0     1     2
    vertBuf[me_idx++] = pos0;
    vertBuf[me_idx++] = pos1;
    vertBuf[me_idx++] = pos2;

    colrBuf[col_idx++] = col;
    colrBuf[col_idx++] = col;
    colrBuf[col_idx++] = col;

    // 1     2     3
    vertBuf[me_idx++] = pos1;
    vertBuf[me_idx++] = pos2;
    vertBuf[me_idx++] = pos3;

    colrBuf[col_idx++] = col;
    colrBuf[col_idx++] = col;
    colrBuf[col_idx++] = col;
        // }

}

void updateBodyMesh(Solid* solid, VboManager* vbom, float minX) {
	int gridSize = (int)ceil(((float)solid->body->numTetrahedra)/BLOCKSIZE);

    updateBodyMesh_k
        <<<make_uint3(gridSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>
        (vbom->GetBuf(BODY_MESH).buf, 
         vbom->GetBuf(BODY_COLORS).buf,
         *solid->body, 
         solid->vertexpool->data, 
         solid->vertexpool->Ui_t, minX);
}

__global__ void
updateStressTensors_k(float4* buf, Body mesh) {
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=mesh.numTetrahedra) return;

    PolyShape p;
    p.CopyToBuf(buf, me_idx);
}

void updateStressTensors(Solid* solid, VboManager* vbom) {
	int gridSize = (int)ceil(((float)solid->body->numTetrahedra)/BLOCKSIZE);

    

    updateStressTensors_k<<<make_uint3(gridSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>(vbom->GetBuf(STRESS_TENSORS).matBuf, 
                                                                                   *solid->body);
}



__global__ void
updateMeshCentersFromDisplacements2_k(float4* buffer, Body mesh, 
                                      Point* points, float4 *displacements)
{
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=mesh.numTetrahedra) return;

    Tetrahedron tetra = mesh.tetrahedra[me_idx];

	float4 pos0, pos1, pos2, pos3;

    pos0 = points[tetra.x] + displacements[tetra.x];
    pos1 = points[tetra.y] + displacements[tetra.y];
    pos2 = points[tetra.z] + displacements[tetra.z];
    pos3 = points[tetra.w] + displacements[tetra.w];

    float4 center = (pos0 + pos1 + pos2 + pos3) / 4.0;

    //PointShape p(center);
    //p.CopyToBuf(bufArray, me_idx);
    /*
    float4 pos = center;
    float4 dir = {0, 1.0, 0, 0};
    VectorShape v(pos, pos + dir);
    v.CopyToBuf(buffer, me_idx);
    */
    
    PolyShape p;
    p.CopyToBuf(buffer, me_idx);
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

/*
__global__ void
updateSurfacePositionsFromDisplacements_k(float3 *tris, float3 *normals, Surface surface, Point *points, float4 *displacements) {
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=surface.numFaces) return;

	Triangle triangle = surface.faces[me_idx];

	float4 pos, pos2, pos3, displacement;

	pos = points[triangle.x];
	pos2 = points[triangle.y];
	pos3 = points[triangle.z];

	displacement = displacements[triangle.x];
	pos.x += displacement.x;  
	pos.y += displacement.y;  
	pos.z += displacement.z;  
	tris[(3*me_idx)+0] = crop_last_dim(pos);

	displacement = displacements[triangle.y];
	pos2.x += displacement.x;  
	pos2.y += displacement.y;  
	pos2.z += displacement.z;  
	tris[(3*me_idx)+1] = crop_last_dim(pos2);

	displacement = displacements[triangle.z];
	pos3.x += displacement.x;  
	pos3.y += displacement.y;  
	pos3.z += displacement.z;  
	tris[(3*me_idx)+2] = crop_last_dim(pos3);

	float3 normal = calcNormal(&pos,&pos2,&pos3);

	normals[(3*me_idx)+0] = normal;
	normals[(3*me_idx)+1] = normal;
	normals[(3*me_idx)+2] = normal;
}
*/


__global__ void updateMeshCentersFromDisplacements_k
(float3 *centers, Body mesh, Point* points, float4 *displacements) {
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=mesh.numTetrahedra) return;

    Tetrahedron tetra = mesh.tetrahedra[me_idx];

	float4 pos0, pos1, pos2, pos3;

    pos0 = points[tetra.x] + displacements[tetra.x];
    pos1 = points[tetra.y] + displacements[tetra.y];
    pos2 = points[tetra.z] + displacements[tetra.z];
    pos3 = points[tetra.w] + displacements[tetra.w];

    float4 center = (pos0 + pos1 + pos2 + pos3) / 4.0;

    centers[me_idx] = crop_last_dim(center);
    //centers[me_idx*2+1] = crop_last_dim(center);
}




