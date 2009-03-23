#define BLOCKSIZE 128

#include <CUDA.h>

#include <stdio.h>
#include <stdlib.h>

#include "Solid.h"
#include "Shapes.h"
#include "VboManager.h"
#include "ColorRamp.h"

#define crop_last_dim make_float3


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
    return cross(edge1, edge0);
}


__global__
void applyTransformation_k(float4* model, float4* vert, float4* mat, unsigned int numVerts, unsigned int numThreads) {
	int me_idx = (numVerts*blockIdx.x) + threadIdx.x;

	if (threadIdx.x>=numVerts)
		return;
    
    int m_idx = 4*blockIdx.x;

    vert[me_idx].x = dot(mat[m_idx + 0], model[threadIdx.x]);
    vert[me_idx].y = dot(mat[m_idx + 1], model[threadIdx.x]);
    vert[me_idx].z = dot(mat[m_idx + 2], model[threadIdx.x]);
    vert[me_idx].w = dot(mat[m_idx + 3], model[threadIdx.x]);
}

/**
 * This function applies the matrix transformation to each
 * vertex in the polygon model. Start as many threats as there are
 * vertices in the model.
 */
void applyTransformation(VisualBuffer& vb) {
    unsigned int gridSize = vb.numElm;
    unsigned int numVerticesInModel = vb.numIndices / vb.numElm; // this is the number of indices in one model.
    //unsigned int blockSize = numVerticesInModel; // number of indices pr. elm (box=36)
    unsigned int blockSize = (int)ceil((float)numVerticesInModel/BLOCKSIZE) * BLOCKSIZE;

    //printf("Grid: %i  block: %i - numThreads: %i - numVerts: %i\n", gridSize, blockSize, vb.numIndices, numVerticesInModel);
    //printf("modelAddr: %i - bufAddr: %i\n", vb.modelBuf, vb.buf);
   
    applyTransformation_k<<<make_uint3(gridSize,1,1), make_uint3(blockSize,1,1)>>>(vb.modelBuf, vb.buf, vb.matBuf, numVerticesInModel, vb.numIndices);
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
    
    float4 normal = calcNormal(&pos,&pos3,&pos2);
	normBuf[(3*me_idx)+0] = normal;
	normBuf[(3*me_idx)+1] = normal;
	normBuf[(3*me_idx)+2] = normal;
}

void updateSurface(Solid* solid, VboManager* vbom) {
	int gridSize = (int)ceil(((float)solid->surface->numFaces)/BLOCKSIZE);

    updateSurface_k<<<make_uint3(gridSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>(vbom->GetBuf(SURFACE_VERTICES).buf, 
                                                                             vbom->GetBuf(SURFACE_NORMALS).buf,
                                                                             *solid->surface, 
                                                                             solid->vertexpool->data, 
                                                                             solid->vertexpool->Ui_t);
}


__global__ void 
updateCenterOfMass_k(float4* buf, Body body, Point* points, float4* displacements) {
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=body.numTetrahedra) return;

    Tetrahedron tetra = body.tetrahedra[me_idx];
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
updateBodyMesh_k(float4* vertBuf, float4* colrBuf, float4* normBuf,
                 Body mesh, Point* points, float4* displacements, float minX) {
	
    int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=mesh.numTetrahedra) return;
    
    Tetrahedron tetra = mesh.tetrahedra[me_idx];

	float4 a, b, c, d;

    a = points[tetra.x] + displacements[tetra.x];
    b = points[tetra.y] + displacements[tetra.y];
    c = points[tetra.z] + displacements[tetra.z];
    d = points[tetra.w] + displacements[tetra.w];

    // Jump index with 12 since there is 4 faces pr. tetrahedra each with 3 vertices.
	me_idx *= 12;
    //   int colr_idx = me_idx;
    int norm_idx = me_idx;

    /*    if ( a.x < minX ||
         b.x < minX ||
         c.x < minX ||
         d.x < minX ) {
        for (unsigned int i=0; i<12; i++) {
            vertBuf[me_idx++] = make_float4(0.0,0.0,0.0,0.0);
        }
    } else 
    */
{
        //float4 col = GetColor(color_ramp_idx, 0.0, mesh.numTetrahedra);
        //float4 col = make_float4(val, 0.0, 0.0, 1.0);
        float4 col = make_float4(0.2, 0.1, 0.5, 1.0);

        // 0     2     3
        vertBuf[me_idx++] = a;
        vertBuf[me_idx++] = b;
        vertBuf[me_idx++] = c;

        // 0     3     1
        vertBuf[me_idx++] = a;
        vertBuf[me_idx++] = c;
        vertBuf[me_idx++] = d;

        // 0     1     2
        vertBuf[me_idx++] = b;
        vertBuf[me_idx++] = d;
        vertBuf[me_idx++] = c;

        // 1     2     3
        vertBuf[me_idx++] = a;
        vertBuf[me_idx++] = d;
        vertBuf[me_idx++] = b;

        // Colors are applied in Physics_k.cu
        /*        // ---------- COLORS -------------------
        colrBuf[colr_idx++] = col;
        colrBuf[colr_idx++] = col;
        colrBuf[colr_idx++] = col;
    
        colrBuf[colr_idx++] = col;
        colrBuf[colr_idx++] = col;
        colrBuf[colr_idx++] = col;

        colrBuf[colr_idx++] = col;
        colrBuf[colr_idx++] = col;
        colrBuf[colr_idx++] = col;

        colrBuf[colr_idx++] = col;
        colrBuf[colr_idx++] = col;
        colrBuf[colr_idx++] = col;
        */
        // -----------  HARD NORMALS  -------------
        float4 normal = calcNormal(&a,&b,&c);
        normBuf[norm_idx++] = normal;
        normBuf[norm_idx++] = normal;
        normBuf[norm_idx++] = normal;

        // Calculate hard normals
        normal = calcNormal(&a,&c,&d);
        normBuf[norm_idx++] = normal;
        normBuf[norm_idx++] = normal;
        normBuf[norm_idx++] = normal;

        // Calculate hard normals
        normal = calcNormal(&b,&d,&c);
        normBuf[norm_idx++] = normal;
        normBuf[norm_idx++] = normal;
        normBuf[norm_idx++] = normal;

        // Calculate hard normals
        normal = calcNormal(&a,&d,&b);
        normBuf[norm_idx++] = normal;
        normBuf[norm_idx++] = normal;
        normBuf[norm_idx++] = normal;
    }
}

void updateBodyMesh(Solid* solid, VboManager* vbom, float minX) {
	int gridSize = (int)ceil(((float)solid->body->numTetrahedra)/BLOCKSIZE);

    updateBodyMesh_k
        <<<make_uint3(gridSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>
        (vbom->GetBuf(BODY_MESH).buf, 
         vbom->GetBuf(BODY_COLORS).buf,
         vbom->GetBuf(BODY_NORMALS).buf,
         *solid->body, 
         solid->vertexpool->data, 
         solid->vertexpool->Ui_t, 
         minX);
}

__global__ void
updateStressTensors_k(float4* buf, Body mesh) {
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=mesh.numTetrahedra) return;

    Matrix4f m;
    m.SetPos(me_idx, me_idx,0);
    m.SetScale(1, 1, 1);
    

    m.CopyToBuf(buf, me_idx);
}

void updateStressTensors(Solid* solid, VboManager* vbom) {
	int gridSize = (int)ceil(((float)solid->body->numTetrahedra)/BLOCKSIZE);

    updateStressTensors_k<<<make_uint3(gridSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>(vbom->GetBuf(STRESS_TENSORS).matBuf, 
                                                                                   *solid->body);
}


__global__ void
planeClipping_k(Body body, Point* points, float4* displacements, 
                float4* bodyMesh, float4* bodyColr, 
                float4* com, float minX) {

    int me_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (me_idx>=body.numTetrahedra) return;    

    Tetrahedron tetra = body.tetrahedra[me_idx];
	float4 a, b, c, d;
    a = points[tetra.x] + displacements[tetra.x];
    b = points[tetra.y] + displacements[tetra.y];
    c = points[tetra.z] + displacements[tetra.z];
    d = points[tetra.w] + displacements[tetra.w];

    // Jump index with 12 since there is 4 faces pr. tetrahedra each with 3 vertices.
	int vert_idx = me_idx * 12;

    if ( a.x < minX ||
         b.x < minX ||
         c.x < minX ||
         d.x < minX ) {
        for (unsigned int i=0; i<12; i++) {
            bodyMesh[vert_idx++] = make_float4(0.0,0.0,0.0,0.0);
        }
        com[me_idx] = make_float4(0.0,0.0,0.0,0.0);
    } else {
        for (unsigned int i=0; i<12; i++) {
            //            float dist = com[me_idx].x;// / 10.0f;//((minX - com[me_idx].x));
            /*if( dist < 0 ) { 
                dist *= -1.0f;
            }
            if( dist > 1.0 ) dist = 1.0;
            */
            bodyColr[vert_idx++].w = 1.0f;//dist;//1.0 - dist;
        } 
    }
}

void planeClipping(Solid* solid, VboManager* vbom, float minX) {
	int gridSize = (int)ceil(((float)solid->body->numTetrahedra)/BLOCKSIZE);

    planeClipping_k<<<make_uint3(gridSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>
        (*solid->body,
         solid->vertexpool->data,
         solid->vertexpool->Ui_t,
         vbom->GetBuf(BODY_MESH).buf,
         vbom->GetBuf(BODY_COLORS).buf,
         vbom->GetBuf(CENTER_OF_MASS).buf,
         minX);
    
    CHECK_FOR_CUDA_ERROR();


}
