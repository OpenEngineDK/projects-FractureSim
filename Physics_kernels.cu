#include "CUDA.h"
#include "Solid.h"
#include "VboManager.h"
#include "eig3.h"
#include "ColorRamp.h"


#define BLOCKSIZE 128

__global__ void calculateDrivingForces_k
(Point *points, float *masses, float4 *externalForces, unsigned int numPoints) {
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=numPoints)
		return;

	externalForces[me_idx] =
        make_float4(0, -9820*masses[me_idx], 0, 0); // for mm.
}

void calculateGravityForces(Solid* solid) {
	int pointSize = (int)ceil(((float)solid->vertexpool->size)/BLOCKSIZE);
	calculateDrivingForces_k
        <<<make_uint3(pointSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>
        (solid->vertexpool->data, solid->vertexpool->mass,
         solid->vertexpool->externalForces, solid->vertexpool->size);
    CHECK_FOR_CUDA_ERROR();
}

__global__ void applyGroundConstraint_k
(Point *points, float4 *displacements, float4 *oldDisplacements,
 float lowestYValue, unsigned int numPoints) {

	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (me_idx>=numPoints)
		return;

	Point me = points[me_idx];
	float4 displacement = displacements[me_idx];

//	printf("%f, %f, %f \n", me.x, me.y, me.z);
//	printf("%f, %f, %f \n", displacement.x, displacement.y, displacement.z);

	if ((me.y+displacement.y)<lowestYValue) {
		displacements[me_idx].y = lowestYValue - me.y;
		//oldDisplacements[me_idx] = displacements[me_idx];
	}
}

void applyFloorConstraint(Solid* solid, float floorYPosition) {
	int pointSize = (int)ceil(((float)solid->vertexpool->size)/BLOCKSIZE);
	applyGroundConstraint_k
        <<<make_uint3(pointSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>
        (solid->vertexpool->data, solid->vertexpool->Ui_t, 
         solid->vertexpool->Ui_tminusdt, floorYPosition,
         solid->vertexpool->size);
    CHECK_FOR_CUDA_ERROR();
}

//note: supposed to be castable to a ShapeFunctionDerivatives object
struct Matrix4x3 { float e[12]; };
struct Matrix3x3 { 
    float e[9]; 

    // Matrix3x3 *Must* be symmetric. Returns eigenvectors in columns of V
    // and corresponding eigenvalues in d.
    __device__
    void calcEigenDecomposition(double V[3][3], double d[3]) {
        double A[3][3];
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                A[i][j] = e[j+i*3];
        // decompose
        eigen_decomposition(A,V,d);
    }
};
struct Matrix6x3 { float e[6*3]; };

texture<float4,  1, cudaReadModeElementType> Ui_t_1d_tex;
texture<float,  1, cudaReadModeElementType> V0_1d_tex;
texture<float4,  1, cudaReadModeElementType> _tex;

#define h(i,j) (sfdm.e[(i-1)*3+(j-1)])
#define u(i,j) (displacements.e[(i-1)*3+(j-1)])
#define X(i,j) (deformation_gradients.e[(i-1)*3+(j-1)])
#define B(i,j) (b_tensor.e[(i-1)*3+(j-1)])
#define C(i,j) (cauchy_green_deformation.e[(i-1)*3+(j-1)])
#define CI(i,j) (c_inverted.e[(i-1)*3+(j-1)])
#define S(i,j) (s_tensor.e[(i-1)*3+(j-1)])


__global__ void
calculateForces_k(Matrix4x3 *shape_function_derivatives, Tetrahedron *tetrahedra, float4 *Ui_t, float *V_0, 
                  int4 *writeIndices, float4 *pointForces, int maxPointForces, float mu, float lambda, 
                  unsigned int numTets, float4* colrBuf, float4* eigBuf)
{
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=numTets)
		return;

	Tetrahedron e = tetrahedra[me_idx];

	if (e.x < 0) 
		return;
	
	Matrix4x3 sfdm = shape_function_derivatives[me_idx];
	Matrix4x3 displacements;

	//fill in displacement values in u (displacements)
	
    //crop_last_dim := make_float3(float4);
	float3 U1 = make_float3(tex1Dfetch( Ui_t_1d_tex, e.x ));
	float3 U2 = make_float3(tex1Dfetch( Ui_t_1d_tex, e.y ));
	float3 U3 = make_float3(tex1Dfetch( Ui_t_1d_tex, e.z ));
	float3 U4 = make_float3(tex1Dfetch( Ui_t_1d_tex, e.w ));

	displacements.e[0] = U1.x;
	displacements.e[1] = U1.y;
	displacements.e[2] = U1.z;

	displacements.e[3] = U2.x;
	displacements.e[4] = U2.y;
	displacements.e[5] = U2.z;

	displacements.e[6] = U3.x;
	displacements.e[7] = U3.y;
	displacements.e[8] = U3.z;

	displacements.e[9] = U4.x;
	displacements.e[10] = U4.y;
	displacements.e[11] = U4.z;

	/*
	displacements.e[0] = Ui_t[e.x].x;
	displacements.e[1] = Ui_t[e.x].y;
	displacements.e[2] = Ui_t[e.x].z;

	displacements.e[3] = Ui_t[e.y].x;
	displacements.e[4] = Ui_t[e.y].y;
	displacements.e[5] = Ui_t[e.y].z;

	displacements.e[6] = Ui_t[e.z].x;
	displacements.e[7] = Ui_t[e.z].y;
	displacements.e[8] = Ui_t[e.z].z;

	displacements.e[9] = Ui_t[e.w].x;
	displacements.e[10] = Ui_t[e.w].y;
	displacements.e[11] = Ui_t[e.w].z;
	*/

	Matrix3x3 deformation_gradients;

    // [Ref: TLED-article, formel 23]
	//Calculate deformation gradients
	X(1,1) = (u(1,1)*h(1,1)+u(2,1)*h(2,1)+u(3,1)*h(3,1)+u(4,1)*h(4,1)+1.0f); 
	X(1,2) = (u(1,1)*h(1,2)+u(2,1)*h(2,2)+u(3,1)*h(3,2)+u(4,1)*h(4,2));
	X(1,3) = (u(1,1)*h(1,3)+u(2,1)*h(2,3)+u(3,1)*h(3,3)+u(4,1)*h(4,3));

	X(2,1) = (u(1,2)*h(1,1)+u(2,2)*h(2,1)+u(3,2)*h(3,1)+u(4,2)*h(4,1));
	X(2,2) = (u(1,2)*h(1,2)+u(2,2)*h(2,2)+u(3,2)*h(3,2)+u(4,2)*h(4,2)+1.0f);
	X(2,3) = (u(1,2)*h(1,3)+u(2,2)*h(2,3)+u(3,2)*h(3,3)+u(4,2)*h(4,3));

	X(3,1) = (u(1,3)*h(1,1)+u(2,3)*h(2,1)+u(3,3)*h(3,1)+u(4,3)*h(4,1));
	X(3,2) = (u(1,3)*h(1,2)+u(2,3)*h(2,2)+u(3,3)*h(3,2)+u(4,3)*h(4,2));
	X(3,3) = (u(1,3)*h(1,3)+u(2,3)*h(2,3)+u(3,3)*h(3,3)+u(4,3)*h(4,3)+1.0f);

/*	printf("\nDeformation gradient tensor for tetrahedron %i: \n", me_idx);
	printf("%f, %f, %f \n", X(1,1), X(1,2), X(1,3));
	printf("%f, %f, %f \n", X(2,1), X(2,2), X(2,3));
	printf("%f, %f, %f \n", X(3,1), X(3,2), X(3,3));
*/
    // [Ref: TLED-article, formel 2]
    // X transposed multiplied with self, to obtain tensor without rotation.
    // calculate Right Cauchy-Green deformation tensor C
    Matrix3x3 cauchy_green_deformation;

	C(1,1) = X(1, 1)*X(1, 1) + X(2, 1)*X(2, 1) + X(3, 1)*X(3, 1); 
	C(1,2) = X(1, 1)*X(1, 2) + X(2, 1)*X(2, 2) + X(3, 1)*X(3, 2); 
	C(1,3) = X(1, 1)*X(1, 3) + X(2, 1)*X(2, 3) + X(3, 1)*X(3, 3); 

	C(2,1) = X(1, 1)*X(1, 2) + X(2, 1)*X(2, 2) + X(3, 1)*X(3, 2); 
	C(2,2) = X(1, 2)*X(1, 2) + X(2, 2)*X(2, 2) + X(3, 2)*X(3, 2); 
	C(2,3) = X(1, 2)*X(1, 3) + X(2, 2)*X(2, 3) + X(3, 2)*X(3, 3);

	C(3,1) = X(1, 1)*X(1, 3) + X(2, 1)*X(2, 3) + X(3, 1)*X(3, 3); 
	C(3,2) = X(1, 2)*X(1, 3) + X(2, 2)*X(2, 3) + X(3, 2)*X(3, 3); 
	C(3,3) = X(1, 3)*X(1, 3) + X(2, 3)*X(2, 3) + X(3, 3)*X(3, 3);
/*
	printf("\nRight Cauchy-Green deformation tensor for tetrahedron %i: \n", me_idx);
	printf("%f, %f, %f \n", C(1,1), C(1,2), C(1,3));
	printf("%f, %f, %f \n", C(2,1), C(2,2), C(2,3));
	printf("%f, %f, %f \n", C(3,1), C(3,2), C(3,3));
*/

	//Invert C
    // [Ref. TLED-article] calculated for use in stress tensor
	Matrix3x3 c_inverted;

	float denominator = (C(3, 1)*C(1, 2)*C(2, 3) - C(3, 1)*C(1, 3)*C(2, 2) - C(2, 1)*C(1, 2)*C(3, 3) 
		+ C(2, 1)*C(1, 3)*C(3, 2) + C(1, 1)*C(2, 2)*C(3, 3) - C(1, 1)*C(2, 3)*C(3, 2));

	CI(1,1) = (C(2, 2)*C(3, 3) - C(2, 3)*C(3, 2))/denominator; 
	CI(1,2) = (-C(1, 2)*C(3, 3) + C(1, 3)*C(3, 2))/denominator; 
	CI(1,3) = (C(1, 2)*C(2, 3) - C(1, 3)*C(2, 2))/denominator; 

	CI(2,1) = (-C(2, 1)*C(3, 3) + C(3, 1)*C(2, 3))/denominator; 
	CI(2,2) = (-C(3, 1)*C(1, 3) + C(1, 1)*C(3, 3))/denominator; 
	CI(2,3) = (-C(1, 1)*C(2, 3) + C(2, 1)*C(1, 3))/denominator; 

	CI(3,1) = (-C(3, 1)*C(2, 2) + C(2, 1)*C(3, 2))/denominator; 
	CI(3,2) = (-C(1, 1)*C(3, 2) + C(3, 1)*C(1, 2))/denominator; 
	CI(3,3) = (-C(2, 1)*C(1, 2) + C(1, 1)*C(2, 2))/denominator;

/*	printf("\nInverted right Cauchy-Green deformation tensor for tetrahedron %i: \n", me_idx);
	printf("%f, %f, %f \n", CI(1,1), CI(1,2), CI(1,3));
	printf("%f, %f, %f \n", CI(2,1), CI(2,2), CI(2,3));
	printf("%f, %f, %f \n", CI(3,1), CI(3,2), CI(3,3));
*/
	//Find the determinant of the deformation gradient
    // [Ref: TLED-article, formel 5]
	float J = X(1, 1)*X(2, 2)*X(3, 3)-X(1, 1)*X(2, 3)*X(3, 2)+X(2, 1)*X(3, 2)*X(1, 3)-
		X(2, 1)*X(1, 2)*X(3, 3)+X(3, 1)*X(1, 2)*X(2, 3)-X(3, 1)*X(2, 2)*X(1, 3);

//	printf("\nDeterminant of the deformation gradient for tetrahedron %i: %f\n", me_idx, J);

	//Calculate stress tensor S from Neo-Hookean Model
	// [Ref: TLED-article, formel 22]
    //  S(ij) = mu(delta(ij)-(C(ij)^(-1))^)+lambda^J(J-1)((C^(-1))(ij))

//	float mu = 1007.0f;
//	float lambda = 49329.0f;
	Matrix3x3 s_tensor;
    
	S(1,1) = mu*(1.0f-CI(1,1)) + lambda*J*(J-1.0f)*CI(1,1);
	S(2,2) = mu*(1.0f-CI(2,2)) + lambda*J*(J-1.0f)*CI(2,2); 
	S(3,3) = mu*(1.0f-CI(3,3)) + lambda*J*(J-1.0f)*CI(3,3);
	S(1,2) = mu*(-CI(1,2)) + lambda*J*(J-1.0f)*CI(1,2);
	S(2,3) = mu*(-CI(2,3)) + lambda*J*(J-1.0f)*CI(2,3);
	S(1,3) = mu*(-CI(1,3)) + lambda*J*(J-1.0f)*CI(1,3); // IS THIS RIGHT?? (3,1) instead?
//	S(1,3) = mu*(-CI(3,1)) + lambda*J*(J-1.0f)*CI(3,1); // IS THIS RIGHT?? (1,3) instead?

    // Calculate eigen vectors and values and map to colors
    double eigenVector[3][3];
    double eigenValue[3];
    s_tensor.calcEigenDecomposition(eigenVector, eigenValue);
    
    eigBuf[me_idx] = make_float4(eigenValue[0], eigenValue[1], eigenValue[2], 0);

    int maxSign = 1;
    int minSign = 1;

    double maxEv = max( max( eigenValue[0], eigenValue[1]), eigenValue[2] );
    double minEv = min( min( eigenValue[0], eigenValue[1]), eigenValue[2] );

    if( maxEv < 0 ){
        maxSign = -1;
        maxEv *= -1;
    }
    if( minEv < 0 ) {
        minSign = -1;
        minEv *= -1;
    }
    double longestEv = maxEv > minEv ? maxEv : minEv;

    //longestEv *= longestEv;
    //longestEv /= 4;

    longestEv *= maxEv > minEv ? maxSign : minSign;
 
    //float4 col = make_float4(0.2, 0.5, 0.1, 1.0);
    float4 col = GetColor(-longestEv, -1000.0, 500.0);
    int colr_idx = me_idx*12;

    // ---------- COLORS -------------------
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
    
 
/*	printf("\nHyper-elastic stresses for tetrahedron %i: \n", me_idx);
	printf("%f, %f, %f \n", S(1,1), S(1,2), S(1,3));
	printf("%f, %f, %f \n", S(2,1), S(2,2), S(2,3));
	printf("%f, %f, %f \n", S(3,1), S(3,2), S(3,3));
*/
	float4 forces[4];

//	float V = V_0[me_idx];//look up volume
	float V = tex1Dfetch( V0_1d_tex, me_idx );

	//	printf("\nVolume for tetrahedron %i: %f\n", me_idx, V);

	for (int a=1; a<=4; a++) // all 4 nodes
	{
		//Calculate B_L from B_L0 and deformation gradients (a is the node number)

		Matrix6x3 b_tensor;

		B(1,1) = h(a, 1)*X(1, 1);  
		B(1,2) = h(a, 1)*X(2, 1);  
		B(1,3) = h(a, 1)*X(3, 1);  

		B(2,1) = h(a, 2)*X(1, 2);
		B(2,2) = h(a, 2)*X(2, 2);
		B(2,3) = h(a, 2)*X(3, 2);

		B(3,1) = h(a, 3)*X(1, 3);  
		B(3,2) = h(a, 3)*X(2, 3);  
		B(3,3) = h(a, 3)*X(3, 3);  

		B(4,1) = h(a, 2)*X(1, 1) + h(a, 1)*X(1, 2);  
		B(4,2) = h(a, 2)*X(2, 1) + h(a, 1)*X(2, 2);  
		B(4,3) = h(a, 2)*X(3, 1) + h(a, 1)*X(3, 2);  

		B(5,1) = h(a, 3)*X(1, 2) + h(a, 2)*X(1, 3);  
		B(5,2) = h(a, 3)*X(2, 2) + h(a, 2)*X(2, 3);  
		B(5,3) = h(a, 3)*X(3, 2) + h(a, 2)*X(3, 3);

		B(6,1) = h(a, 3)*X(1, 1) + h(a, 1)*X(1, 3);  
		B(6,2) = h(a, 3)*X(2, 1) + h(a, 1)*X(2, 3);  
		B(6,3) = h(a, 3)*X(3, 1) + h(a, 1)*X(3, 3);

/*		printf("\nSubmatrix for a=%i of the stationary strain-displacement matrix for tetrahedron %i: \n", a, me_idx);
		printf("%f, %f, %f \n", B(1,1), B(1,2), B(1,3));
		printf("%f, %f, %f \n", B(2,1), B(2,2), B(2,3));
		printf("%f, %f, %f \n", B(3,1), B(3,2), B(3,3));
		printf("%f, %f, %f \n", B(4,1), B(4,2), B(4,3));
		printf("%f, %f, %f \n", B(5,1), B(5,2), B(5,3));
		printf("%f, %f, %f \n", B(6,1), B(6,2), B(6,3));
*/
		//calculate forces
		float4 force;
		force.x = V*(B(1, 1)*S(1, 1)+B(2, 1)*S(2, 2)+B(3, 1)*S(3, 3)+B(4, 1)*S(1, 2)+B(5, 1)*S(2, 3)+B(6, 1)*S(1, 3));
		force.y = V*(B(1, 2)*S(1, 1)+B(2, 2)*S(2, 2)+B(3, 2)*S(3, 3)+B(4, 2)*S(1, 2)+B(5, 2)*S(2, 3)+B(6, 2)*S(1, 3));
		force.z = V*(B(1, 3)*S(1, 1)+B(2, 3)*S(2, 2)+B(3, 3)*S(3, 3)+B(4, 3)*S(1, 2)+B(5, 3)*S(2, 3)+B(6, 3)*S(1, 3));
		force.w = 0;

		if (length(make_float3(force))<100000 && J>0)
			forces[a-1] = force;
		else
			forces[a-1] = make_float4(0,0,0,0);

	}

/*	printf("\nFor tetrahedron %i: \n", me_idx);
	printf("node1 (%i) force: %f, %f, %f \n", e.x, forces[0].x, forces[0].y, forces[0].z);
	printf("node2 (%i) force: %f, %f, %f \n", e.y, forces[1].x, forces[1].y, forces[1].z);
	printf("node3 (%i) force: %f, %f, %f \n", e.z, forces[2].x, forces[2].y, forces[2].z);
	printf("node4 (%i) force: %f, %f, %f \n", e.w, forces[3].x, forces[3].y, forces[3].z);
*/
/*	forces[0].x = 0;
	forces[0].y = -0.000001;
	forces[0].z = 0;
	forces[1].x = 0;
	forces[1].y = -0.000001;
	forces[1].z = 0;
	forces[2].x = 0;
	forces[2].y = -0.000001;
	forces[2].z = 0;
	forces[3].x = 0;
	forces[3].y = -0.000001;
	forces[3].z = 0;*/

	// look up where this tetrahedron is allowed to store its force contribution to a node
	// store force-vector
	pointForces[maxPointForces * e.x + writeIndices[me_idx].x] = forces[0];
	pointForces[maxPointForces * e.y + writeIndices[me_idx].y] = forces[1];
	pointForces[maxPointForces * e.z + writeIndices[me_idx].z] = forces[2];
	pointForces[maxPointForces * e.w + writeIndices[me_idx].w] = forces[3];

//	printf("Max num forces: %i\n", maxPointForces);

//	printf("%i, %i, %i, %i \n", writeIndices[me_idx].x, writeIndices[me_idx].y, writeIndices[me_idx].z, writeIndices[me_idx].w );
}

void calculateInternalForces(Solid* solid, VboManager* vbom)  {
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
		mesh->numTetrahedra,
        vbom->GetBuf(BODY_COLORS).buf,
        vbom->GetBuf(EIGEN_VALUES).buf);

	// free textures
	cudaUnbindTexture( V0_1d_tex );
	cudaUnbindTexture( Ui_t_1d_tex );
    CHECK_FOR_CUDA_ERROR();
}


__global__ void
updateDisplacements_k(float4 *Ui_t, float4 *Ui_tminusdt, float *M, float4 *Ri, float4 *Fi, int maxNumForces, float4 *ABC, unsigned int numPoints)
{
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=numPoints)
		return;

	float4 F = make_float4(0,0,0,0);

//	printf("Max num forces: %i\n", maxNumForces);

	for (int i=0; i<maxNumForces; i++)
	{
		float4 force_to_add = Fi[me_idx*maxNumForces+i];
		//float4 force_to_add = Fi[me_idx+maxNumForces]; //test - to coarless
		F.x += force_to_add.x;
		F.y += force_to_add.y;
		F.z += force_to_add.z;
	}
//	printf("Accumulated node %i force: %f, %f, %f \n", me_idx, F.x, F.y, F.z);

	float4 ABCi = ABC[me_idx];
	float4 Uit = Ui_t[me_idx];
	float4 Uitminusdt = Ui_tminusdt[me_idx];

	float4 R = Ri[me_idx];
	float x = ABCi.x * (R.x - F.x) + ABCi.y * Uit.x + ABCi.z * Uitminusdt.x;
	float y = ABCi.x * (R.y - F.y) + ABCi.y * Uit.y + ABCi.z * Uitminusdt.y;
	float z = ABCi.x * (R.z - F.z) + ABCi.y * Uit.z + ABCi.z * Uitminusdt.z;

/*	float x = ABCi.x * (-F.x) + ABCi.y * Ui_t[me_idx].x + ABCi.z * Ui_tminusdt[me_idx].x;
	float y = ABCi.x * (-F.x) + ABCi.y * Ui_t[me_idx].y + ABCi.z * Ui_tminusdt[me_idx].y;
	float z = ABCi.x * (-F.x ) + ABCi.y * Ui_t[me_idx].z + ABCi.z * Ui_tminusdt[me_idx].z;
*/
	Ui_tminusdt[me_idx] = make_float4(x,y,z,0);//XXXXXXXXXXXXXXXXXXXXX

}

void updateDisplacement(Solid* solid) {
	int pointSize = (int)ceil(((float)solid->vertexpool->size)/BLOCKSIZE);
	updateDisplacements_k
        <<<make_uint3(pointSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>
        (solid->vertexpool->Ui_t, solid->vertexpool->Ui_tminusdt, 
         solid->vertexpool->mass, solid->vertexpool->externalForces, 
         solid->vertexpool->pointForces, solid->vertexpool->maxNumForces,
         solid->vertexpool->ABC, solid->vertexpool->size);
    CHECK_FOR_CUDA_ERROR();

	float4 *temp = solid->vertexpool->Ui_t;
	solid->vertexpool->Ui_t = solid->vertexpool->Ui_tminusdt;
	solid->vertexpool->Ui_tminusdt = temp;
}
