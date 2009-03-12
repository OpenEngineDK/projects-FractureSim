#define BLOCKSIZE 128

#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <string.h>
#include <GL/glew.h>

#ifdef WIN32
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif

#include <cufft.h>
#include <cutil.h>
//#include <cutil_interop.h>
#include <cuda_gl_interop.h>
//#include <cutil_gl_error.h>
#include <cuda.h>

#include "CUDA.h"
#include "float_utils.h"

#include "Visualization_kernels.cu"
#include "FEM_kernels.cu"

#include "TetrahedralMesh.h"
/*#include "ImageD.h"
#include "ImageD.cu"
*/
#include "Vector3D.h"

//Body* mesh;

//static BinaryMask * hmask, *dmask;
//static ImageD *imageh;
//static ImageD* imaged;

//static float *dmass, *dA, *dB, *dC;

#define NUM_VBO 10

//this i a bit of a hack
//a maximum of NUM_VBO elements can be displayed 
static GLuint vbo[NUM_VBO];// = {0,0,0,0,0,0,0,0,0,0};
static GLuint normalVbo[NUM_VBO];// = {0,0,0,0,0,0,0,0,0,0};
static GLuint centersVbo[NUM_VBO];// = {0,0,0,0,0,0,0,0,0,0};

//static float3 *ABC, *Ui_t, *Ui_tminusdt, *pointForces, *externalForces;
//static ShapeFunctionDerivatives *shape_function_deriv;



////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void
createVBO(GLuint* vbo, int numTriangles)
{
    // create buffer object
    glGenBuffers( 1, vbo);
    glBindBuffer( GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = numTriangles* 3 * sizeof(float3);
    glBufferData( GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);

    glBindBuffer( GL_ARRAY_BUFFER, 0);

    // register buffer object with CUDA
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(*vbo));

    CUT_CHECK_ERROR_GL();
}

void drawCoordinates(void)
{
	//Draw coordinate axes
	glEnable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	//glTranslatef(0,-8,0);	
	glBegin(GL_LINES);
	glColor3f(1,0,0);
	glVertex3f(0,0,0);
	glVertex3f(2,0,0);

	glColor3f(0,1,0);
	glVertex3f(0,0,0);
	glVertex3f(0,2,0);

	glColor3f(0,0,1);
	glVertex3f(0,0,0);
	glVertex3f(0,0,2);
	glEnd();
	//glTranslatef(0,8,0);	

	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
}

void drawTrianglesFromVBO(GLuint vbo, GLuint normalVbo,  int numTriangles, float4 color)
{
	float4 ambcolor;

	ambcolor.x = 1.0 * color.x;
	ambcolor.y = 1.0 * color.y;
	ambcolor.z = 1.0 * color.z;
	ambcolor.w = 0.1 * color.w;

	glMaterialfv(GL_FRONT, GL_DIFFUSE, (GLfloat*)&color);
	glMaterialfv(GL_FRONT, GL_AMBIENT, (GLfloat*)&ambcolor);
    glColor4f(color.x,color.y,color.z,color.w);

	//glColorMaterial(GL_FRONT, GL_AMBIENT);
	//glEnable(GL_COLOR_MATERIAL);

	glShadeModel(GL_FLAT);
    glEnable(GL_DEPTH_TEST);

	glEnable(GL_AUTO_NORMAL); 
	glEnable(GL_NORMALIZE); 
	// render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBufferARB(GL_ARRAY_BUFFER, normalVbo);
    glNormalPointer(GL_FLOAT, sizeof(float)*3, 0);
    glEnableClientState(GL_NORMAL_ARRAY);

    //glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_TRIANGLES, 0,numTriangles * 3);
//	glPointSize(10);
//	  glDrawArrays(GL_POINTS, 0, mesh->numTetrahedra * 3);
    
    glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDisable(GL_DEPTH_TEST);
    //glDisable(GL_CULL_FACE);
    //glDisable(GL_LIGHTING);
	//glDisable(GL_COLOR_MATERIAL);
}

void drawTetrahedraFromVBO(GLuint centersVbo, int numTetrahedra, float4 color) {
	float4 ambcolor;
	ambcolor.x = 1.0 * color.x;
	ambcolor.y = 1.0 * color.y;
	ambcolor.z = 1.0 * color.z;
	ambcolor.w = 0.1 * color.w;

	glMaterialfv(GL_FRONT, GL_DIFFUSE, (GLfloat*)&color);
	glMaterialfv(GL_FRONT, GL_AMBIENT, (GLfloat*)&ambcolor);
    glColor4f(color.x,color.y,color.z,color.w);

	//glColorMaterial(GL_FRONT, GL_AMBIENT);
	//glEnable(GL_COLOR_MATERIAL);

	glShadeModel(GL_FLAT);
    glEnable(GL_DEPTH_TEST);

	glEnable(GL_AUTO_NORMAL); 
	glEnable(GL_NORMALIZE); 
	// render from the vbo

    glBindBuffer(GL_ARRAY_BUFFER, centersVbo);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glNormalPointer(GL_FLOAT, sizeof(float)*3, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    //glColor3f(1.0, 0.0, 0.0);
    //glDrawArrays(GL_TRIANGLES, 0,numTriangles * 3);
	glPointSize(2);
    glDrawArrays(GL_POINTS, 0, numTetrahedra);
    
    glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDisable(GL_DEPTH_TEST);
    //glDisable(GL_CULL_FACE);
    //glDisable(GL_LIGHTING);
	//glDisable(GL_COLOR_MATERIAL);
}

void cleanupDisplay(void) {
	printf("Deleting VBOs \n");
	for (int i = 0; i<NUM_VBO; i++ )
	    cudaGLUnregisterBufferObject(vbo[i]);
    CUT_CHECK_ERROR("cudaGLUnregisterBufferObject failed");
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
	for (int i = 0; i<NUM_VBO; i++ )
	    glDeleteBuffersARB(1, &vbo[i]);

	for (int i = 0; i<NUM_VBO; i++ )
	    glDeleteBuffersARB(1, &normalVbo[i]);

	printf("Exiting...\n");
	//exit(0);

}

void display(unsigned int object_number, Solid* solid, float4* buf) {
    Body* mesh = solid->body;
    //TetrahedralTLEDState *state = solid->state;
    Surface* surface = solid->surface;

//	int tetSize = mesh->numTetrahedra / BLOCKSIZE;
//	int pointSize = solid->vertexpool->size / BLOCKSIZE;

	if (vbo[object_number] == 0)
		createVBO(&(vbo[object_number]), surface->numFaces);

	if (normalVbo[object_number] == 0)
		createVBO(&(normalVbo[object_number]), surface->numFaces);

	if (centersVbo[object_number] == 0)
		createVBO(&(centersVbo[object_number]), mesh->numTetrahedra);

    float3 *d_pos, *d_normal, *d_centers;
	CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&d_pos, vbo[object_number]));
    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&d_normal, normalVbo[object_number]));
    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&d_centers, centersVbo[object_number]));

	//extractSurfaceWithDisplacements_k<<<make_uint3(tetSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>(dptr, mesh->tetrahedra, solid->vertexpool->data, solid->vertexpool->Ui_t);

	int gridSize = (int)ceil(((float)surface->numFaces)/BLOCKSIZE);
    updateSurfacePositionsFromDisplacements_k<<<make_uint3(gridSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>(d_pos, d_normal, *surface, solid->vertexpool->data, solid->vertexpool->Ui_t);

    //update tedrahedra centers
	int gridSize2 = (int)ceil(((float)mesh->numTetrahedra)/BLOCKSIZE);
    //	updateMeshCentersFromDisplacements_k<<<make_uint3(gridSize2,1,1), make_uint3(BLOCKSIZE,1,1)>>>(d_centers, *mesh, solid->vertexpool->data, solid->vertexpool->Ui_t);
	
    updateMeshCentersFromDisplacements2_k<<<make_uint3(gridSize2,1,1), make_uint3(BLOCKSIZE,1,1)>>>(buf, *mesh, solid->vertexpool->data, solid->vertexpool->Ui_t);
	
  
    


    CUT_CHECK_ERROR("Error extracting surface");
    // unmap buffer object
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vbo[object_number]));
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( normalVbo[object_number]));
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( centersVbo[object_number]));

	float4 color;

	switch (object_number) {	
	case 0 : color = make_float4(1,0,0,0.5); break;
	case 1 : color = make_float4(0,0.5,0.5,0.5); break;
	case 2 : color = make_float4(0,0,1,0.5); break;
	case 3 : color = make_float4(0.5,0.5,0,0.5); break;
	case 4 : color = make_float4(1,0,0,1); break;
	case 5 : color = make_float4(1,0,0,1); break;
	case 6 : color = make_float4(1,0,0,1); break;
	case 7 : color = make_float4(1,0,0,1); break;
	case 8 : color = make_float4(1,0,0,1); break;
	case 9 : color = make_float4(1,0,0,1); break;
	}

	drawTrianglesFromVBO(vbo[object_number], normalVbo[object_number], surface->numFaces, color);

	color = make_float4(0,1,0,0.5);
    drawTetrahedraFromVBO(centersVbo[object_number], mesh->numTetrahedra, color);
}

Tetrahedron fixTetrahedronOrientation(Tetrahedron tet, Point *hpoints) {
    Tetrahedron res = tet;
 
    //the x,y,z and w points are called a,b,c and d
    Vector3D a = crop_last_dim(hpoints[tet.x]);
    Vector3D b = crop_last_dim(hpoints[tet.y]);
    Vector3D c = crop_last_dim(hpoints[tet.z]);
    Vector3D d = crop_last_dim(hpoints[tet.w]);

    Vector3D ab = b-a;
    Vector3D ac = c-a;
    Vector3D ad = d-a;
		
    Vector3D abxac = cross(ab,ac);
		
    float projection = dot(abxac, ad);

    if (projection<0) {
        //printf("Switching a and b\n");
        res.x = tet.y;
        res.y = tet.x;
    }    
    return res; 
}

//must return smallest length encountered
float CPUPrecalculation(Solid* solid, int blockSize, unsigned int& return_maxNumForces, float density, float smallestAllowedVolume, float smallestAllowedLength) {

    Body* mesh = solid->body;

    float totalSmallestLengthSquared = 9e9;
    double totalVolume = 0;
    Tetrahedron *htetrahedra = 
        (Tetrahedron*)malloc(sizeof(Tetrahedron) * mesh->numTetrahedra);
    Point *hpoints =
        (Point*)malloc(sizeof(Point) * solid->vertexpool->size);

    //copy datastructure back and compact
    cudaMemcpy(hpoints, solid->vertexpool->data, sizeof(Point) * solid->vertexpool->size, cudaMemcpyDeviceToHost);
    cudaFree(solid->vertexpool->data);

    cudaMemcpy(htetrahedra, mesh->tetrahedra, sizeof(Tetrahedron) * mesh->numTetrahedra, cudaMemcpyDeviceToHost);
    cudaFree(mesh->tetrahedra);

    unsigned int tmpPointCount = solid->vertexpool->size;

	//	int res = tmpPointCount % blockSize;

	//	if (res>0)
	//	{
	//		tmpPointCount += blockSize - res;
	//	}

    float* mass = (float*)malloc(sizeof(float) * tmpPointCount);
    memset(mass, 0, sizeof(float) * tmpPointCount);
    for (unsigned int i = 0; i < mesh->numTetrahedra; i++) {
        if (htetrahedra[i].x >= 0) {
            htetrahedra[i] = fixTetrahedronOrientation(htetrahedra[i],hpoints);
        }
    }

    unsigned int tmpTetCount = mesh->numTetrahedra;
	//	res = tmpTetCount % blockSize;
    //		if (res>0) {
    //			tmpTetCount += blockSize - res;
    //		}

    int4* writeIndices = (int4*)malloc(sizeof(int4) * tmpTetCount);
    mesh->numWriteIndices = tmpTetCount;

    Tetrahedron* tets = (Tetrahedron*)malloc(sizeof(Tetrahedron) * tmpTetCount);
    float* initialVolume = (float*)malloc(sizeof(float) * tmpTetCount); 
    int* numForces = (int*)malloc(sizeof(int) * tmpPointCount);
    int maxNumForces = 0;
    for (unsigned int i = 0; i < tmpPointCount; i++) {
        numForces[i] = 0;
    }
		
    unsigned int counter = 0;
    for (unsigned int i = 0; i < mesh->numTetrahedra; i++) {
        if (htetrahedra[i].x >= 0 && 
            htetrahedra[i].y >= 0 && 
            htetrahedra[i].z >= 0 &&
            htetrahedra[i].w >= 0) { // if id's are not zero

            tets[counter].x = htetrahedra[i].x;
            tets[counter].y = htetrahedra[i].y;
            tets[counter].z = htetrahedra[i].z;
            tets[counter].w = htetrahedra[i].w;

            writeIndices[counter].x = numForces[htetrahedra[i].x]++;
            if (writeIndices[counter].x+1 > maxNumForces)
                maxNumForces = writeIndices[counter].x+1;
            writeIndices[counter].y = numForces[htetrahedra[i].y]++;
            if (writeIndices[counter].y+1 > maxNumForces)
                maxNumForces = writeIndices[counter].y+1;
            writeIndices[counter].z = numForces[htetrahedra[i].z]++;
            if (writeIndices[counter].z+1 > maxNumForces)
                maxNumForces = writeIndices[counter].z+1;
            writeIndices[counter].w = numForces[htetrahedra[i].w]++;
            if (writeIndices[counter].w+1 > maxNumForces)
                maxNumForces = writeIndices[counter].w+1;

//			printf(" %i ", writeIndices[counter].w);

            // calculate volume and smallest length
            Vector3D a = crop_last_dim(hpoints[htetrahedra[i].x]);
            Vector3D b = crop_last_dim(hpoints[htetrahedra[i].y]);
            Vector3D c = crop_last_dim(hpoints[htetrahedra[i].z]);
            Vector3D d = crop_last_dim(hpoints[htetrahedra[i].w]);

            Vector3D ab = b-a; // these 3 are used for volume calc
            Vector3D ac = c-a;
            Vector3D ad = d-a;

            Vector3D bc = c-b;
            Vector3D cd = d-c;
            Vector3D bd = d-a;

            float smallestLengthSquared = ab.squaredLength();
            
            float sql = ac.squaredLength();
            if (sql<smallestLengthSquared) smallestLengthSquared = sql;
            sql = ad.squaredLength();
            if (sql<smallestLengthSquared) smallestLengthSquared = sql;
            sql = bc.squaredLength();
            if (sql<smallestLengthSquared) smallestLengthSquared = sql;
            sql = cd.squaredLength();
            if (sql<smallestLengthSquared) smallestLengthSquared = sql;
            sql = bd.squaredLength();
            if (sql<smallestLengthSquared) smallestLengthSquared = sql;
            
            if (smallestLengthSquared < 
                smallestAllowedLength*smallestAllowedLength) {
                continue;
            }

            if (smallestLengthSquared<totalSmallestLengthSquared) 
                totalSmallestLengthSquared = smallestLengthSquared;

            Vector3D cross_product = cross(ab,ac);
            float cross_length = cross_product.length();
            //Length of vector ad projected onto cross product normal
            float projected_length = dot(ad, cross_product/cross_length);
            float volume = (1.0f/6.0f) * projected_length*cross_length;
            if (volume<smallestAllowedVolume) {
                continue;
            }
            totalVolume += volume;

/*
				static float smallestvolume = 100000;

				if (volume<smallestvolume) 
				{
					smallestvolume=volume;
					printf("smallest element volume: %g\n", smallestvolume);
				}

				static float largestvolume = 0;
				if (volume>largestvolume) 
				{
					largestvolume=volume;
					printf("largest element volume: %g\n", largestvolume);
				}
*/
//				printf("volume: %g\n", volume);

            initialVolume[counter] = volume;
            if (volume<0.1) 
                printf("volume: %f \n",volume);

/*				if	(dot(ad, cross_product)<0)
				{
					printf("volume problem ");
				//	continue;
				}
*/
//				float density = 0.0000001f;
//				float density = 0.001f;

            mass[htetrahedra[i].x] += volume * 0.25 * density;
            mass[htetrahedra[i].y] += volume * 0.25 * density;
            mass[htetrahedra[i].z] += volume * 0.25 * density;
            mass[htetrahedra[i].w] += volume * 0.25 * density;
            
            counter++;
        }
    }
        //std::cout << "max num forces:" << maxNumForces << std::endl;
        //maxNumForces = 64;
        //getchar();

		// these are the padded ones
    for (unsigned int i = counter; i < tmpTetCount; i++) {
        tets[i].x = -1;
        tets[i].y = -1;
        tets[i].z = -1;
        tets[i].w = -1;
    }
    printf("Total volume: %f\n", totalVolume);
    mesh->numTetrahedra = tmpTetCount;

    //copy points to a padded array
    cudaMalloc((void**)&(solid->vertexpool->data), sizeof(Point) * tmpPointCount);
    cudaMemcpy(solid->vertexpool->data, hpoints, sizeof(Point) * solid->vertexpool->size, cudaMemcpyHostToDevice);
    solid->vertexpool->size = tmpPointCount;

//		for (int i=0; i<solid->vertexpool->size; i++) {
//			printf("Vertex %i: %f, %f, %f\n", i, 
//                 (points[i].x), (points[i].y), (points[i].z));
//		}

    cudaMalloc((void**)&(mesh->tetrahedra), 
               sizeof(Tetrahedron) * mesh->numTetrahedra);
    cudaMalloc((void**)&(mesh->writeIndices),
               sizeof(int4) * mesh->numWriteIndices);
    cudaMalloc((void**)&(mesh->volume), 
               sizeof(float) * mesh->numTetrahedra);
    cudaMalloc((void**)&(solid->vertexpool->mass), 
               sizeof(float) * solid->vertexpool->size);

    cudaMemcpy(mesh->tetrahedra, tets, 
               sizeof(Tetrahedron) * mesh->numTetrahedra, 
               cudaMemcpyHostToDevice);
    cudaMemcpy(mesh->writeIndices, writeIndices, 
               sizeof(int4) * mesh->numWriteIndices, cudaMemcpyHostToDevice);
    cudaMemcpy(mesh->volume, initialVolume, 
               sizeof(float) * mesh->numTetrahedra, cudaMemcpyHostToDevice);
    cudaMemcpy(solid->vertexpool->mass, mass,
               sizeof(float) * solid->vertexpool->size, cudaMemcpyHostToDevice);

    free(hpoints);
    free(htetrahedra);
    free(numForces);

    for (unsigned int i = 0; i < tmpPointCount; i++) {
        if (mass[i] == 0) {
            printf("warning: point without mass detected\n");
        }
    }

	//	for (int i = 0; i < mesh->numWriteIndices; i++) {
	//		printf("%i, %i, %i, %i \n",
    //             writeIndices[i].x, writeIndices[i].y,
    //             writeIndices[i].z, writeIndices[i].w );
	//	}

    CUT_CHECK_ERROR("Error deleting");
    free(tets);
    free(initialVolume);
    free(writeIndices);
    free(mass);

    return_maxNumForces = maxNumForces;
    return sqrtf(totalSmallestLengthSquared);
}

void calculateGravityForces(Solid* solid) {
	int pointSize = (int)ceil(((float)solid->vertexpool->size)/BLOCKSIZE);
	calculateDrivingForces_k
        <<<make_uint3(pointSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>
        (solid->vertexpool->data, solid->vertexpool->mass,
         solid->vertexpool->externalForces, solid->vertexpool->size);
}

void applyFloorConstraint(Solid* solid, float floorYPosition) {
	int pointSize = (int)ceil(((float)solid->vertexpool->size)/BLOCKSIZE);
	applyGroundConstraint_k
        <<<make_uint3(pointSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>
        (solid->vertexpool->data, solid->vertexpool->Ui_t, 
         solid->vertexpool->Ui_tminusdt, floorYPosition,
         solid->vertexpool->size);
}

void calculateInternalForces(Solid* solid)  {
    Body* mesh = solid->body;
    TetrahedralTLEDState *state = solid->state;
	
	// bind state as 1d texture with 4 channels, to enable cache on lookups
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();	
	cudaBindTexture( 0,  Ui_t_1d_tex, solid->vertexpool->Ui_t, channelDesc );
	
	// bind mesh as 1d texture with 1 channel, to enable texture cache on lookups
	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float>();	
	cudaBindTexture( 0,  V0_1d_tex, mesh->volume, channelDesc2 );

	// run kernel (BLOCKSIZE=128)
	int tetSize = (int)ceil(((float)mesh->numTetrahedra)/BLOCKSIZE);
	calculateForces_k<<<make_uint3(tetSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>(
		(Matrix4x3 *)mesh->shape_function_deriv,
		mesh->tetrahedra,
		solid->vertexpool->Ui_t,
		mesh->volume,
		mesh->writeIndices,
		solid->vertexpool->pointForces,
		solid->vertexpool->maxNumForces,
		state->mu,
		state->lambda,
		mesh->numTetrahedra);

	// free textures
	cudaUnbindTexture( V0_1d_tex );
	cudaUnbindTexture( Ui_t_1d_tex );
}

void doTimeStep(Solid* solid) {
	calculateInternalForces(solid);

	int pointSize = (int)ceil(((float)solid->vertexpool->size)/BLOCKSIZE);
	updateDisplacements_k
        <<<make_uint3(pointSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>
        (solid->vertexpool->Ui_t, solid->vertexpool->Ui_tminusdt, 
         solid->vertexpool->mass, solid->vertexpool->externalForces, 
         solid->vertexpool->pointForces, solid->vertexpool->maxNumForces,
         solid->vertexpool->ABC, solid->vertexpool->size);

	float4 *temp = solid->vertexpool->Ui_t;
	solid->vertexpool->Ui_t = solid->vertexpool->Ui_tminusdt;
	solid->vertexpool->Ui_tminusdt = temp;
}

void precompute(Solid* solid, 
                float density, float smallestAllowedVolume, 
                float smallestAllowedLength, float mu,
                float lambda, float timeStepFactor, float damping) {

    Body* mesh = solid->body;
    TetrahedralTLEDState *state = solid->state;
    //CUT_DEVICE_INIT(1, "");
	float smallestLength =
        CPUPrecalculation(solid, BLOCKSIZE, solid->vertexpool->maxNumForces,
                          density, smallestAllowedVolume,
                          smallestAllowedLength);


    // mu and lambda is Lame's constants used for calculating
    // Young's modulus and Poisson's ratio.

    // E is Young's modulus, it is a material constant defining axial
    // deformations within the material. 
    // It simply tells how much a material will stretch or compress
    // when forces are applied.
    // [Ref: FEM, Smith, page 661, C.8a]
    // [Ref: Fysik-bog, Erik, page. 168];
	float E = mu*(3.0f*lambda+2.0f*mu)/(lambda+mu);

    // mu is Poisson's ratio. 
    // (From wiki) When a sample of material is stretched in one direction, it
    // tends to contract (or rarely, expand) in the other two
    // directions. Conversely, when a sample of material is compressed
    // in one direction, it tends to expand in the other two
    // directions. Poisson's ratio (ν) is a measure of this tendency.
    // [Ref: FEM, Smith, page 661, C.8b]
    // [Ref: Fysik-bog, Erik, page. 171]
	float nu = lambda/(2.0f*(lambda+mu));
 
    // c is the dilatational wave speed of the material. This constant
    // says something about how sound travels through solid materials.
    // We use it for defining the critical delta time step. 
    // Since explicit time integration is conditional stable, we must
    // keep out time step below the critical delta time step.
    // [Ref: TLED-article formula 17]
    // [Ref: Fysik-bog, Erik, page. 198]
	float c = sqrt((E*(1.0f-nu))/(density*(1.0f-nu)*(1.0f-2.0f*nu)));

	// the factor is to account for changes i c during deformation
    // [ref: TLED-article formula 16]
	float timeStep = timeStepFactor*smallestLength/c;

	//float timeStep = 0.0001f;
	state->mu = mu;
	state->lambda = lambda;

	int tetSize = (int)ceil(((float)mesh->numTetrahedra)/BLOCKSIZE);
	int pointSize = (int)ceil(((float)solid->vertexpool->size)/BLOCKSIZE);

	cudaMalloc((void**)&(solid->vertexpool->pointForces), 
               solid->vertexpool->maxNumForces * sizeof(float4) * solid->vertexpool->size);

    CHECK_FOR_CUDA_ERROR();

	cudaMemset(solid->vertexpool->pointForces, 0, 
               sizeof(float4) * solid->vertexpool->maxNumForces * solid->vertexpool->size);

    // [Ref: TLED-article, formula 11,12,13]
	precalculateABC
        <<<make_uint3(pointSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>
        (solid->vertexpool->ABC, solid->vertexpool->mass,
         timeStep, damping, solid->vertexpool->size);
	
	precalculateShapeFunctionDerivatives_k
        <<<make_uint3(tetSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>
        (mesh->shape_function_deriv, 
         mesh->tetrahedra, 
         solid->vertexpool->data,
         mesh->numTetrahedra);

    CHECK_FOR_CUDA_ERROR();

	state->timeStep = timeStep;
}
