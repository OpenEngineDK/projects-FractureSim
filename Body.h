#ifndef _BODY_
#define _BODY_


//typedef int4 Tetrahedron; 

// Edge convention as explained below
const static int EDGE_CONVENTION[] = {0,1,0,2,0,3,1,2,1,3,2,3};

// Helper functions for the tetrahedron edge convention
static int GetEdgeStartIndex(int edgeNumber) { return EDGE_CONVENTION[edgeNumber*2]; }
static int GetEdgeEndIndex(int edgeNumber)   { return EDGE_CONVENTION[(edgeNumber*2)+1]; }

/*static float dot(float4 v0, float4 v1) {
    return v0.x*v1.x + v0.y*v1.y + v0.z*v1.z; 
    }*/
/**
 * Tetrahedron has 4 points defining 6 edges. 
 * Edge convention is as follows: Edge# : [start, end] (startIdx, endIdx)
 * 0 : [x,y] (0,1)
 * 1 : [x,z] (0,2)
 * 2 : [x,w] (0,3)
 * 3 : [y,z] (1,2)
 * 4 : [y,w] (1,3)
 * 5 : [z,w] (2,3)
 */
struct Tetrahedron {
    int x;
    int y;
    int z;
    int w;

    Tetrahedron& operator=(const int4 rh) {
        x=rh.x;
        y=rh.y;
        z=rh.z;
        w=rh.w;
        return *this;
    }

    // Maps edge indices to node indices
    int GetNodeIndex(int index){
        switch(index) {
        case 0: return x;
        case 1: return y;
        case 2: return z;
        case 3: return w;
        }
        return -1;
    }    
};



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

    // List of tetrahedron indices
	Tetrahedron* tetrahedra;
 
    // List of tetrahedron indices in main memory
    Tetrahedron* tetrahedraMainMem;

    // Each tetrahedron has 6 edges, each edge is shared by max 2 additional tetrahedrons 
    // because only tetrahedrons with a face in common are considered a neighbour.
    // Each edge has 2 entries in the edgeSharing list. Each entry is the tetra index
    // to the one sharing the edge.
    // edgeSharing = {B,D, B,F, D,F, B,E, D,E, E,F} means edge 0 is shared by tetra B and D,
    // edge 1 is shared by B and F, edge 2 is shared by D and F etc....
    int* edgeSharing;

    // Tetrahedron index list of neighbours. First 4 entries are neighbours to tetra 1, etc..
    int* neighbour;

    // Volume for each tetrahedron
	float* volume;

    // Largest stress with direction and size
    // Encoded as a float4: [s_x, s_y, s_z, size]
    float4* principalStress;

    // Crack plane normal or [0,0,0,0] if not yet cracked
    float4* crackPlaneNorm;

    // True if the maximum stress in a tetrahedron has been exceeded
    bool* maxStressExceeded;

	ShapeFunctionDerivatives* shape_function_deriv;
	int4* writeIndices;

    // List of crack points, represented as ratios of distance on each of
    // the 6 edges defining a tetrahedron. 
    float* crackPoints;

    Body();
    Body(unsigned int size);
   
    bool IsMaxStressExceeded();

    void GetPrincipalStress(float4* principalStress);
    void GetTetrahedrons(Tetrahedron* pTetras);


    void AddCrackPoint(int tetraIdx, int edgeIdx, float crackPoint);
    bool AddCrackPoint(int tetraIdx, int nodeIdx1, int nodeIdx2, float crackPoint);

    int*   GetNeighbours(int tetraIdx);
    float4 GetPrincipalStressNorm(int tetraIdx);
    bool   HasCrackPoints(int tetraIdx);
    int    NumCrackPoints(int tetraIdx);
    float* GetCrackPoints(int tetraIdx);
    

    //void CopyFromDeviceToHost();

    //bool CopyToGPU();
    //bool CopyToCPU();

    void ConvertToCuda();
    void DeAlloc();
    void Print();

};



#endif // _BODY_
