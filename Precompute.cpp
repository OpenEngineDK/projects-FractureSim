#include "Precompute.h"

#include "CUDA.h"
#include "Precompute_kernels.h"

#include <Math/Vector.h>
#include <Geometry/Face.h>
#include <Logging/Logger.h>

using namespace Math;

float squaredLength(Math::Vector<3,float> a) {
    return a * a;
}

bool checkTriangle(Math::Vector<3,float> a, Math::Vector<3,float> b,
                   Math::Vector<3,float> c, Math::Vector<3,float> com) {
    //    Geometry::Face face(a,b,c);
    //int result = face.ComparePointPlane(com, Math::EPS);

    // Calculate face normal - we need normals perpendicular to the face
    // so we can't use the loaded face normals because they might be soft.
    Math::Vector<3,double> v1 = b.ToDouble() - a.ToDouble();
    Math::Vector<3,double> v2 = c.ToDouble() - a.ToDouble();
    Math::Vector<3,float> h = (v1 % v2).ToFloat();

	if (h.GetLength() == 0.0f) {
		logger.warning << "hardNorm is 0.0f: " << a << ","
                       << b << "," << c << logger.end;
    }

    // Calculate the distance from constraint p1 to plane.
    float distance = (h * (com - a));

    // If the distance is behind the plane correct p1
    if (distance > Math::EPS) 
        return true;
    else 
        if (distance < -Math::EPS) 
            return false;
	else 
        throw Core::Exception("center of mass is located on one of the triangles");
}

bool checkPositiveVolume(Math::Vector<3,float> a, Math::Vector<3,float> b,
                         Math::Vector<3,float> c, Math::Vector<3,float> d) {
    Math::Vector<3,float> ab = b-a;
    Math::Vector<3,float> ac = c-a;
    Math::Vector<3,float> ad = d-a;

    Math::Vector<3,float> abxac = ab % ac; // cross product
    float projection = abxac * ad; // dot product
    if (projection < 0) {
        logger.info << "volume is negative" << logger.end;
        return false;
    }
    return true;
}

bool checkTetrahedron(Tetrahedron tet, Point* hpoints) {
    //the x,y,z and w points are called a,b,c and d
    float4 pa = hpoints[tet.x];
    float4 pb = hpoints[tet.y];
    float4 pc = hpoints[tet.z];
    float4 pd = hpoints[tet.w];

    Math::Vector<3,float> a(pa.x,pa.y,pa.z);
    Math::Vector<3,float> b(pb.x,pb.y,pb.z);
    Math::Vector<3,float> c(pc.x,pc.y,pc.z);
    Math::Vector<3,float> d(pd.x,pd.y,pd.z);

    // center of mass
    Math::Vector<3,float> com = (a+b+c+d)/4;

    //logger.info << "checking tri: abc" << logger.end;
    if ( !checkTriangle(a,b,c,com) ) return false;
    //logger.info << "checking tri: acd" << logger.end;
    if ( !checkTriangle(a,c,d,com) ) return false;
    //logger.info << "checking tri: adc " << logger.end;
    if ( !checkTriangle(b,d,c,com) ) return false;
    //logger.info << "checking tri: adb " << logger.end;
    if ( !checkTriangle(a,d,b,com) ) return false;
    //logger.info << "checking volume " << logger.end;
    if ( !checkPositiveVolume(a,b,c,d) ) return false;
    return true;
}

Tetrahedron fixTetrahedronOrientation(Tetrahedron tet, Point* hpoints) {
    int a = tet.x;
    int b = tet.y;
    int c = tet.z;
    int d = tet.w;
    Tetrahedron res;

    static int index = -1;
    index++;
    /*
    char chr = getchar();
    if (chr == 'q')
        exit(-1);
    */
    //logger.info << "------------ tetra ---------" << logger.end;
    //logger.info << "checking tretra: abcd" << logger.end;
    res = make_int4(a,b,c,d);
    if ( checkTetrahedron(res,hpoints) )
        return res;
    logger.info << "tetra with index:" << index << logger.end;
    logger.info << "checking tretra: abdc" << logger.end;
    res = make_int4(a,b,d,c);
    if ( checkTetrahedron(res,hpoints) )
        return res;

    logger.info << "checking tretra: acbd" << logger.end;
    res = make_int4(a,c,b,d);
    if ( checkTetrahedron(res,hpoints) )
        return res;
    logger.info << "checking tretra: acdb" << logger.end;
    res = make_int4(a,c,d,b);
    if ( checkTetrahedron(res,hpoints) )
        return res;

    logger.info << "checking tretra: adbc" << logger.end;
    res = make_int4(a,d,b,c);
    if ( checkTetrahedron(res,hpoints) )
        return res;
    logger.info << "checking tretra: adcb" << logger.end;
    res = make_int4(a,d,c,b);
    if ( checkTetrahedron(res,hpoints) )
        return res;

    throw Core::Exception("invalid tetrahedron");
}

//must return smallest length encountered
float CPUPrecalculation
(Solid* solid, unsigned int& return_maxNumForces, 
 float density, float smallestAllowedVolume, float smallestAllowedLength) {

    float totalSmallestLengthSquared = 9e9; //float::max
    double totalVolume = 0;
    Tetrahedron *htetrahedra = solid->body->tetrahedra;
    Point* hpoints = solid->vertexpool->data;
    /*for (unsigned int i = 0; i < solid->vertexpool->size; i++) {
        printf("vertex[%i] = (%f,%f,%f)\n", i, 
               hpoints[i].x,hpoints[i].y,hpoints[i].z);
    }*/

    unsigned int tmpPointCount = solid->vertexpool->size;
    float* mass = solid->vertexpool->mass;
    for (unsigned int i = 0; i < solid->body->numTetrahedra; i++) {
        if (htetrahedra[i].x >= 0) {
            htetrahedra[i] = fixTetrahedronOrientation(htetrahedra[i],hpoints);
        } else
            printf("error\n");
    }

    unsigned int tmpTetCount = solid->body->numTetrahedra;
    solid->body->numWriteIndices = tmpTetCount;
    int4* writeIndices = solid->body->writeIndices;
    Tetrahedron* tets = solid->body->tetrahedra;
    float* initialVolume = solid->body->volume;
    int maxNumForces = 0;

    int* numForces = (int*)malloc(sizeof(int) * tmpPointCount);
    memset(numForces, 0, sizeof(float) * tmpPointCount);
		
    unsigned int counter = 0;
    for (unsigned int i = 0; i < solid->body->numTetrahedra; i++) {
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

            //printf(" %i ", writeIndices[counter].w);

            // calculate volume and smallest length
            float4 pa = hpoints[htetrahedra[i].x];
            float4 pb = hpoints[htetrahedra[i].y];
            float4 pc = hpoints[htetrahedra[i].z];
            float4 pd = hpoints[htetrahedra[i].w];

            Math::Vector<3,float> a(pa.x,pa.y,pa.z);
            Math::Vector<3,float> b(pb.x,pb.y,pb.z);
            Math::Vector<3,float> c(pc.x,pc.y,pc.z);
            Math::Vector<3,float> d(pd.x,pd.y,pd.z);

            Math::Vector<3,float> ab = b-a; // these 3 are used for volume calc
            Math::Vector<3,float> ac = c-a;
            Math::Vector<3,float> ad = d-a;

            Math::Vector<3,float> bc = c-b;
            Math::Vector<3,float> cd = d-c;
            Math::Vector<3,float> bd = d-a;

            float smallestLengthSquared = squaredLength(ab);
            float sql = squaredLength(ac);

            if (sql<smallestLengthSquared) smallestLengthSquared = sql;
            sql = squaredLength(ad);
            if (sql<smallestLengthSquared) smallestLengthSquared = sql;
            sql = squaredLength(bc);
            if (sql<smallestLengthSquared) smallestLengthSquared = sql;
            sql = squaredLength(cd);
            if (sql<smallestLengthSquared) smallestLengthSquared = sql;
            sql = squaredLength(bd);
            if (sql<smallestLengthSquared) smallestLengthSquared = sql;
            
            if (smallestLengthSquared < 
                smallestAllowedLength*smallestAllowedLength) {
                continue;
            }

            if (smallestLengthSquared<totalSmallestLengthSquared) 
                totalSmallestLengthSquared = smallestLengthSquared;

            Math::Vector<3,float> cross_product = ab % ac; // cross product
            float cross_length = cross_product.GetLength();
            //Length of vector ad projected onto cross product normal
            float projected_length = ad * cross_product.GetNormalize();
            float volume = (1.0f/6.0f) * projected_length*cross_length;
            //printf("calc-volume[%i]=%f\n", i, volume);
            if (volume<smallestAllowedVolume) {
                printf("skiping tetrahedron with index: %i\n", i);
                continue;
            }


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
            totalVolume += volume;
            initialVolume[counter] = volume;

            if (volume<0.1) {
                /*static unsigned int index = 0;
                index++;
                printf("volume[%i]: %f \n",index,volume);*/
                printf("volume on tetrahedron with index is too small: %i\n", i);
                //continue;
            }

/*				if	(dot(ad, cross_product)<0)
				{
					printf("volume problem ");
				//	continue;
				}
*/

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

    printf("original number of tetrahedron: %i, number after padding: %i\n",
           tmpTetCount, counter);
    // these are the padded ones
    for (unsigned int i = counter; i < tmpTetCount; i++) {
        tets[i].x = -1;
        tets[i].y = -1;
        tets[i].z = -1;
        tets[i].w = -1;
    }
    printf("Total volume: %f\n", totalVolume);
    // ?!? mesh->numTetrahedra = counter;
    solid->body->numTetrahedra = tmpTetCount;
    solid->vertexpool->size = tmpPointCount;

//		for (int i=0; i<solid->vertexpool->size; i++) {
//			printf("Vertex %i: %f, %f, %f\n", i, 
//                 (points[i].x), (points[i].y), (points[i].z));
//		}

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
    return_maxNumForces = maxNumForces;
    return sqrtf(totalSmallestLengthSquared);
}

void precompute(Solid* solid, 
                float density, float smallestAllowedVolume, 
                float smallestAllowedLength, float mu,
                float lambda, float timeStepFactor, float damping) {

    //TetrahedralTLEDState *state = solid->state;
    //CUT_DEVICE_INIT(1, "");

    //solid->Print();

	float smallestLength =
        CPUPrecalculation(solid, solid->vertexpool->maxNumForces,
                          density, smallestAllowedVolume,
                          smallestAllowedLength);

    //solid->Print();

    CHECK_FOR_CUDA_ERROR();
    solid->vertexpool->ConvertToCuda();
    solid->body->ConvertToCuda();
    solid->surface->ConvertToCuda();
    CHECK_FOR_CUDA_ERROR();

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

	solid->state->timeStep = timeStep;
	solid->state->mu = mu;
	solid->state->lambda = lambda;

    // [Ref: TLED-article, formula 11,12,13]
    precalculateABC(timeStep, damping, solid->vertexpool);

    precalculateShapeFunctionDerivatives(solid);
}
