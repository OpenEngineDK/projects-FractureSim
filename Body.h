#ifndef _BODY_
#define _BODY_

typedef int4 Tetrahedron; 

struct ShapeFunctionDerivatives {
	float3 h1; // derivatives at node 1 w.r.t. (x,y,z)
	float3 h2; // derivatives at node 2 w.r.t. (x,y,z)
	float3 h3; // derivatives at node 3 w.r.t. (x,y,z)
	float3 h4; // derivatives at node 4 w.r.t. (x,y,z)
};

class Body {
 public:
	unsigned int numTetrahedra;
	int numWriteIndices;

	Tetrahedron* tetrahedra;
	float* volume;
	ShapeFunctionDerivatives* shape_function_deriv;
	int4 *writeIndices;

    Body();
    Body(unsigned int size);
    void ConvertToCuda();
    void DeAlloc();
    void Print();
};

#endif // _BODY_
