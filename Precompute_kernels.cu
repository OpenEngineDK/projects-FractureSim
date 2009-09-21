#include <Meta/CUDA.h>

#include "Solid.h"

#define BLOCKSIZE 128

__global__ void precalculateABC_kernel
(float4* ABCm, float* M, float timestep, float alpha, unsigned int numPoints) {
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (me_idx>=numPoints)
		return;

    //timestep *= 1000;

	float twodelta = timestep*2.0f;
	float deltasqr = timestep*timestep;

	float Mii = M[me_idx];
	float Dii = alpha*Mii; // mass-proportional damping is applied
	
    // printf("M: %f\n",Mii);

	float Ai = 1.0f/(Dii/twodelta + Mii/deltasqr);
	float Bi = ((2.0f*Mii)/deltasqr)*Ai;
	float Ci = (Dii/twodelta)*Ai - 0.5f*Bi;

    //printf("ABC for node %i: %e, %e, %e \n", me_idx, Ai, Bi, Ci);

	ABCm[me_idx] = make_float4(Ai,Bi,Ci,Mii);
}

void precalculateABC(float timeStep, float damping, VertexPool* vertexpool) {
	int pointSize = (int)ceil(((float)vertexpool->size)/BLOCKSIZE);

	precalculateABC_kernel
        <<<make_uint3(pointSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>
        (vertexpool->ABC, vertexpool->mass,
         timeStep, damping, vertexpool->size);
    CHECK_FOR_CUDA_ERROR();
}

double total = 0;
int counter = 0;

__global__ void precalculateShapeFunctionDerivatives_kernel
(ShapeFunctionDerivatives *shape_function_derivatives, 
 Tetrahedron *tetrahedra, Point *points, unsigned int numTets) {
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=numTets)
		return;

	Tetrahedron tet = tetrahedra[me_idx];

	if (tet.x<0) return; // illegal tetrahedron

	float4 a = points[tet.x];
	float4 b = points[tet.y];
	float4 c = points[tet.z];
	float4 d = points[tet.w];

    double A = c.y*d.x*b.z;
    double B = a.x*c.y*d.z;
    double C = a.y*d.x*c.z;
    double D = c.x*b.y*a.z;
    double E = c.x*d.y*a.z;
    double F = a.x*b.y*d.z;
    double G = a.x*b.y*c.z;
    double H = c.x*b.y*d.z;
    double I = c.x*a.y*d.z;
    double J = a.y*c.x*b.z;
    double K = a.y*d.x*b.z;
    double L = d.x*b.y*c.z;
    double M = b.x*c.y*a.z;
    double N = b.x*d.y*c.z;
    double O = b.x*a.y*c.z;
    double P = b.x*c.y*d.z;
    double Q = b.x*d.y*a.z;
    double R = b.x*a.y*d.z;
    double S = d.y*c.x*b.z;
    double T = d.y*a.x*b.z;
    double U = d.y*a.x*c.z;
    double V = c.y*a.x*b.z;
    double X = d.x*c.y*a.z;
    double Y = d.x*b.y*a.z;

    double denominator =
        A + B + C -
        D + E - F +
        G + H - I +
        J - K - L +
        M + N - O -
        P - Q + R -
        S + T - U - 
        V - X + Y;
    /*
	double denominator = 
        c.y*d.x*b.z + a.x*c.y*d.z + a.y*d.x*c.z - 
        c.x*b.y*a.z + c.x*d.y*a.z - a.x*b.y*d.z +
		a.x*b.y*c.z + c.x*b.y*d.z - c.x*a.y*d.z +
        a.y*c.x*b.z - a.y*d.x*b.z - d.x*b.y*c.z + 
        b.x*c.y*a.z + b.x*d.y*c.z - b.x*a.y*c.z -
        b.x*c.y*d.z - b.x*d.y*a.z + b.x*a.y*d.z - 
        d.y*c.x*b.z + d.y*a.x*b.z - d.y*a.x*c.z -
        c.y*a.x*b.z - d.x*c.y*a.z + d.x*b.y*a.z;
    */
    //    printf("denominator %f \n", denominator);

	ShapeFunctionDerivatives sfd;	
    // shape function (x,y,z) = c1 + c2*x + c3*y + c4*z
    // C-a2
    double h1_A = c.y*d.z;
    double h1_B = b.y*d.z;
    double h1_C = b.y*c.z;
    double h1_D = d.y*b.z;
    double h1_E = d.y*c.z;
    double h1_F = c.y*b.z;
 
    //printf("a(x,y,z) = (%E,%E,%E)\n", a.x, a.y, a.z);
    //printf("b(x,y,z) = (%E,%E,%E)\n", b.x, b.y, b.z);
    //printf("c(x,y,z) = (%E,%E,%E)\n", c.x, c.y, c.z);
    //printf("d(x,y,z) = (%E,%E,%E)\n", d.x, d.y, d.z); 

    double tmp = h1_A - h1_B + h1_C + h1_D - h1_E - h1_F;

	sfd.h1.x = (double)tmp / (double)denominator;

    //printf("h1.x = %E - %E, %E, %E, %E, %E, %E\n", sfd.h1.x, h1_A, h1_B, h1_C, h1_D, h1_E, h1_F);

    /*	sfd.h1.x = (c.y*d.z - b.y*d.z + b.y*c.z +
                d.y*b.z - d.y*c.z - c.y*b.z)/denominator;
    */


    // C-a3
    sfd.h1.y = -(-c.x*b.z + d.x*b.z + c.x*d.z -
                 b.x*d.z + b.x*c.z - d.x*c.z)/denominator;
	// C-a4
    sfd.h1.z = (-c.x*b.y + c.x*d.y - b.x*d.y + 
                b.x*c.y - d.x*c.y + d.x*b.y)/denominator;

	sfd.h2.x = -(c.y*d.z - c.y*a.z + d.y*a.z - 
                 a.y*d.z + c.z*a.y - d.y*c.z)/denominator;
	sfd.h2.y = (c.x*d.z - a.x*d.z - c.x*a.z + 
                d.x*a.z - d.x*c.z + a.x*c.z)/denominator;
	sfd.h2.z = -(-a.x*d.y + a.x*c.y + d.x*a.y -
                 c.x*a.y + c.x*d.y - d.x*c.y)/denominator;

	sfd.h3.x = (-d.y*b.z + a.y*b.z - a.y*d.z + 
                b.y*d.z + d.y*a.z - b.y*a.z)/denominator;
	sfd.h3.y = -(d.x*a.z - b.x*a.z - d.x*b.z +
                 a.x*b.z - a.x*d.z + b.x*d.z)/denominator;
	sfd.h3.z = (-a.x*d.y + d.x*a.y - b.x*a.y - 
                d.x*b.y + a.x*b.y + b.x*d.y)/denominator;

	sfd.h4.x = -(-c.z*a.y + a.y*b.z + b.y*c.z +
                 c.y*a.z - b.y*a.z - c.y*b.z)/denominator;
	sfd.h4.y = (-a.x*c.z + c.x*a.z - b.x*a.z + 
                b.x*c.z + a.x*b.z - c.x*b.z)/denominator;
	sfd.h4.z = -(-a.x*c.y - b.x*a.y + b.x*c.y +
                 a.x*b.y - c.x*b.y + c.x*a.y)/denominator;
    
    /*  printf("\nFor tetrahedron %i: \n", me_idx);
	printf("h1 derivatives: %f, %f, %f \n", sfd.h1.x, sfd.h1.y, sfd.h1.z);
	printf("h2 derivatives: %f, %f, %f \n", sfd.h2.x, sfd.h2.y, sfd.h2.z);
	printf("h3 derivatives: %f, %f, %f \n", sfd.h3.x, sfd.h3.y, sfd.h3.z);
	printf("h4 derivatives: %f, %f, %f \n", sfd.h4.x, sfd.h4.y, sfd.h4.z);
    float l1 = length(sfd.h1);
    float l2 = length(sfd.h2);
    float l3 = length(sfd.h3);
    float l4 = length(sfd.h4);
    total += l1+l2+l3+l4;
    counter++;
    printf("L1 = %f, L2 = %f, L3 = %f, L4 = %f, total = %f, agv = %f", l1,l2,l3,l4, total, total/counter);
*/
	shape_function_derivatives[me_idx] = sfd;
}

void precalculateShapeFunctionDerivatives(Solid* solid) {
	int tetSize = (int)ceil(((float)solid->body->numTetrahedra)/BLOCKSIZE);
	precalculateShapeFunctionDerivatives_kernel
        <<<make_uint3(tetSize,1,1), make_uint3(BLOCKSIZE,1,1)>>>
        (solid->body->shape_function_deriv, 
         solid->body->tetrahedra, 
         solid->vertexpool->data,
         solid->body->numTetrahedra);
    CHECK_FOR_CUDA_ERROR();
}
