#include "Precompute.h"

#include <Meta/CUDA.h>
#include "Precompute_kernels.h"

#include <Math/Vector.h>
#include <Geometry/Face.h>
#include <Logging/Logger.h>
#include <cstring>

using namespace Math;

bool checkTriangle(Math::Vector<3,float> a, Math::Vector<3,float> b,
                   Math::Vector<3,float> c, Math::Vector<3,float> com) {
    //    Geometry::Face face(a,b,c);
    //int result = face.ComparePointPlane(com, Math::EPS);

    // Calculate face normal - we need normals perpendicular to the face
    // so we can't use the loaded face normals because they might be soft.
    Math::Vector<3,double> v1 = b.ToDouble() - a.ToDouble();
    Math::Vector<3,double> v2 = c.ToDouble() - a.ToDouble();
    Math::Vector<3,float> h = (v1 % v2).ToFloat();

    /*	if (h.GetLength() == 0.0f) {
		logger.warning << "hardNorm is 0.0f: " << a << ","
                       << b << "," << c << logger.end;
                       }*/

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
    float totalLargestLengthSquared = 0; //float::min
    float totalSmallestVolume = 9e9;
    float totalLargestVolume = 0;

    double totalMass = 0;
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

            float smallestLengthSquared = ab*ab;
            if(ac*ac < smallestLengthSquared) smallestLengthSquared = ac*ac;
            if(ad*ad < smallestLengthSquared) smallestLengthSquared = ad*ad;
            if(bc*bc < smallestLengthSquared) smallestLengthSquared = bc*bc;
            if(cd*cd < smallestLengthSquared) smallestLengthSquared = cd*cd;
            if(bd*bd < smallestLengthSquared) smallestLengthSquared = bd*bd;
            
            float largestLengthSquared = ab*ab;
            if(ac*ac > largestLengthSquared) largestLengthSquared = ac*ac;
            if(ad*ad > largestLengthSquared) largestLengthSquared = ad*ad;
            if(bc*bc > largestLengthSquared) largestLengthSquared = bc*bc;
            if(cd*cd > largestLengthSquared) largestLengthSquared = cd*cd;
            if(bd*bd > largestLengthSquared) largestLengthSquared = bd*bd;
            
            if (smallestLengthSquared < 
                smallestAllowedLength*smallestAllowedLength) {
                printf("SKIPPING: smallest length in tetra is to small, index: %i, length: %f\n", i, sqrt(smallestLengthSquared));
                continue;
            }

            if (smallestLengthSquared<totalSmallestLengthSquared) 
                totalSmallestLengthSquared = smallestLengthSquared;

            if (largestLengthSquared>totalLargestLengthSquared) 
                totalLargestLengthSquared = largestLengthSquared;

            Math::Vector<3,float> cross_product = ab % ac; // cross product
            float cross_length = cross_product.GetLength();
            //Length of vector ad projected onto cross product normal
            float projected_length = ad * cross_product.GetNormalize();

            /*				
            if (ad*cross_product < 0)
                printf("volume problem ");
            */

            float volume = (1.0f/6.0f) * projected_length*cross_length;
            //printf("calc-volume[%i]=%f\n", i, volume);

            if (volume<smallestAllowedVolume) {
                printf("SKIPPING: volume too small skiping tetrahedron with index: %i, volume: %f\n", i, volume);
                continue;
            }


            if (volume < totalSmallestVolume) totalSmallestVolume = volume;
            if (volume > totalLargestVolume) totalLargestVolume = volume;

            totalVolume += volume;
            initialVolume[counter] = volume;

            ///printf("volume[%i]: %f, ",i,volume);

            //if (volume<0.1) {
                /*static unsigned int index = 0;
                index++;
                printf("volume[%i]: %f \n",index,volume);*/
            //  printf("volume on tetrahedron is too small. volume: %f, index: %i\n", volume, i);
                //continue;
            //}


            // center of mass
            Math::Vector<3,float> com = (a+b+c+d)/4.0;

            // lengths
            float al = (com - a).GetLength();
            float bl = (com - b).GetLength();
            float cl = (com - c).GetLength();
            float dl = (com - d).GetLength();
            float tLength = al + bl + cl + dl;

            // weights
            float av = al/tLength;
            float bv = bl/tLength;
            float cv = cl/tLength;
            float dv = dl/tLength;
            float tv = av + bv + cv + dv;

            float sum = av/tv + bv/tv + cv/tv + dv/tv;

            if (abs(sum - 1.0) > Math::EPS)
                logger.info << "ERROR sum not 100% on: "
                            << " index: " << i 
                            << " tv:" << sum
                            << " av:" << av/tv
                            << " bv:" << bv/tv
                            << " cv:" << cv/tv
                            << " dv:" << dv/tv
                            << logger.end;

            // x=a, y=b, z=c, w=d
            float tMass = volume * density;
            totalMass += tMass;
            ///printf("mass[%i]: %f \n",i,tMass);
            mass[htetrahedra[i].x] += tMass * av/tv;
            mass[htetrahedra[i].y] += tMass * bv/tv;
            mass[htetrahedra[i].z] += tMass * cv/tv;
            mass[htetrahedra[i].w] += tMass * dv/tv;


            /* old calcs
            mass[htetrahedra[i].x] += volume * 0.25 * density;
            mass[htetrahedra[i].y] += volume * 0.25 * density;
            mass[htetrahedra[i].z] += volume * 0.25 * density;
            mass[htetrahedra[i].w] += volume * 0.25 * density;
            */      
            /*
            float mMass = 200000;
            mass[htetrahedra[i].x] = mMass;
            mass[htetrahedra[i].y] = mMass;
            mass[htetrahedra[i].z] = mMass;
            mass[htetrahedra[i].w] = mMass;
            */
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

    // override mass distrubution by constant mass
    for(unsigned int i=0; i< solid->vertexpool->size;i++)
        mass[i] = totalMass / (solid->vertexpool->size);


    // ?!? mesh->numTetrahedra = counter;
    solid->body->numTetrahedra = tmpTetCount;
    solid->vertexpool->size = tmpPointCount;

    float minEdge = sqrtf(totalSmallestLengthSquared);
    float maxEdge = sqrtf(totalLargestLengthSquared);
    printf("Smalest edge: %f\n", minEdge);
    printf("Largest edge: %f\n", maxEdge);
    printf("Min/Max edge ratio: %f\n", maxEdge/minEdge);
    printf("Smalest volume: %f M^3\n", totalSmallestVolume);
    printf("Largest volume: %f M^3\n", totalLargestVolume);
    printf("Min/Max volume ratio: %f\n",
           totalLargestVolume/totalSmallestVolume);
    printf("Total volume: %f M^3\n", totalVolume);
    printf("Average volume: %f M^3\n", totalVolume/solid->body->numTetrahedra);

    printf("Total mass: %f kg\n", totalMass);
    printf("Average mass: %f kg\n", totalMass/solid->body->numTetrahedra);

//		for (int i=0; i<solid->vertexpool->size; i++) {
//			printf("Vertex %i: %f, %f, %f\n", i, 
//                 (points[i].x), (points[i].y), (points[i].z));
//		}

    for (unsigned int i = 0; i < tmpPointCount; i++) {
        if (mass[i] == 0) {
            printf("warning: point without mass detected\n");
        }
        ///printf("point mass[%i]: %f \n",i,mass[i]);
    }

	//	for (int i = 0; i < mesh->numWriteIndices; i++) {
	//		printf("%i, %i, %i, %i \n",
    //             writeIndices[i].x, writeIndices[i].y,
    //             writeIndices[i].z, writeIndices[i].w );
	//	}

    return_maxNumForces = maxNumForces;
    return sqrtf(totalSmallestLengthSquared);
}

void moveAccordingToBoundingBox(Solid* solid) {
    float4 a = solid->vertexpool->data
        [solid->surface->faces[0].x ];

    Math::Vector<3,float> max(a.x,a.y,a.z);
    Math::Vector<3,float> min(max);

    // find the boundary values
    for (unsigned int k=0; k<solid->surface->numFaces; k++) {
            float4 a = solid->vertexpool->data
                [solid->surface->faces[k].x ];
            Math::Vector<3,float> v(a.x,a.y,a.z);
            for (int j=0; j<3; j++)
                if (v[j] < min[j]) min[j] = v[j];
                else if (v[j] > max[j]) max[j] = v[j];

            float4 b = solid->vertexpool->data
                [solid->surface->faces[k].y ];
            v = Math::Vector<3,float>(b.x,b.y,b.z);
            for (int j=0; j<3; j++)
                if (v[j] < min[j]) min[j] = v[j];
                else if (v[j] > max[j]) max[j] = v[j];

            float4 c = solid->vertexpool->data
                [solid->surface->faces[k].z ];
            v = Math::Vector<3,float>(c.x,c.y,c.z);
            for (int j=0; j<3; j++)
                if (v[j] < min[j]) min[j] = v[j];
                else if (v[j] > max[j]) max[j] = v[j];
    }

    Math::Vector<3,float> center = (max - min) / 2 + min;
    logger.info << "bounding box: min=" << min 
                << " max=" << max 
                << " center=" << center 
                << logger.end;

    solid->vertexpool->Move(-center[0], -min[1], -center[2]);
}

float precompute(Solid* solid, 
                float smallestAllowedVolume, 
                float smallestAllowedLength,
                float timeStepFactor, float damping) {

    float density = solid->mp->density;
	float smallestLength =
        CPUPrecalculation(solid, solid->vertexpool->maxNumForces,
                          density, smallestAllowedVolume,
                          smallestAllowedLength);


    CHECK_FOR_CUDA_ERROR();
    solid->vertexpool->ConvertToCuda();
    solid->body->ConvertToCuda();
    solid->surface->ConvertToCuda();
    CHECK_FOR_CUDA_ERROR();


    float E = solid->mp->E;
    float nu = solid->mp->nu;

    float M = (E*(1.0f-nu))/((1.0f+nu)*(1.0f-2.0f*nu)); //P-wave modulus

    // c is the dilatational wave speed of the material. This constant
    // says something about how sound travels through solid materials.
    // We use it for defining the critical delta time step. 
    // Since explicit time integration is conditional stable, we must
    // keep out time step below the critical delta time step.
    // [Ref: TLED-article formula 17]
    // [Ref: Fysik-bog, Erik, page. 198]
	float c = sqrt(M/density); // speed of pressure wave (P-wave)

	// the factor is to account for changes i c during deformation
    // [ref: TLED-article formula 16]
	float timeStep = timeStepFactor * smallestLength/c;


    logger.info << "time step: " << timeStep 
                << "sec = " << timeStep*1000.0f << "milli Sec" << logger.end;
    logger.info << "time step squared: " << timeStep*timeStep 
                << "sec" << logger.end;

	solid->state->timeStep = timeStep;
	solid->state->mu = E/(2*(1+nu)); // (G)
	solid->state->lambda = (E*nu)/((1+nu)*(1-2*nu));

    // [Ref: TLED-article, formula 11,12,13]
    precalculateABC(timeStep, damping, solid->vertexpool);

    precalculateShapeFunctionDerivatives(solid);

    return timeStep;
}

// Returns true if the two tetrahedrons share a face, equal to
// having 3 nodes in common.
bool isNeighbour(Tetrahedron a, Tetrahedron b) {
    //    logger.info << "Comparing: " << a.x << "," << a.y << "," << a.z << "," << a.w << " == " <<
    //     b.x << "," << b.y << "," << b.z << "," << b.w << logger.end;  
    int nodesInCommon = 0;
    for( int i=0; i<4; i++ ) {
        int aIdx = a.GetNodeIndex(i);
        for( int j=0; j<4; j++ ){
            int bIdx = b.GetNodeIndex(j);
            if( aIdx == bIdx ) 
                nodesInCommon++;   
        }
    }
    //logger.info << "NodesInCommon: " << nodesInCommon << logger.end;
    return nodesInCommon == 3;
}


void createNeighbourList(Solid* solid) {
    // Get body
    Body* body = solid->body;
    // Get all tetrahedron indices from device
    body->tetrahedraMainMem = (Tetrahedron*)malloc(sizeof(Tetrahedron) * solid->body->numTetrahedra);
    body->GetTetrahedrons(body->tetrahedraMainMem);

    Tetrahedron* tetrahedra = body->tetrahedraMainMem;

    // TEMP
    int tetraWithFourNeighbours = 0;

    int numPrints = 0;
    // For each tetrahedron 
    for( unsigned int i=0; i<body->numTetrahedra; i++ ) {
        float percentDone = ((float)(i+1) / (float)body->numTetrahedra) * 100.0f;
        if( ((int)percentDone / 10) > numPrints ) {
            numPrints++;
            logger.info << "Neighbour List Processing " << numPrints * 10 << "% done" << logger.end;
        }

        // Find neighbours
        std::list<int> neighbours;
        for( unsigned int j=0; j<body->numTetrahedra; j++ )
            if( i != j && isNeighbour(tetrahedra[i], tetrahedra[j]) ){
                body->neighbour[(i*4) + neighbours.size()] = j;
                neighbours.push_back(j);
            }

        //logger.info << "Tetra #" << i << " has " << neighbours.size() << " neighbours" << logger.end;

        // Just for statistical use 
        if( neighbours.size() == 4 )  tetraWithFourNeighbours++;

        // For each edge in tetrahedron
        for( int edge=0; edge<6; edge++ ) {
            // Get node index 1
            int idx1 = tetrahedra[i].GetNodeIndex(GetEdgeStartIndex(edge));
            // Get node index 2
            int idx2 = tetrahedra[i].GetNodeIndex(GetEdgeEndIndex(edge));
            
            int numTetraSharingEdge = 0;
            std::list<int>::iterator itr;
            for( itr=neighbours.begin(); itr!=neighbours.end(); itr++ ){
                if( numTetraSharingEdge == 2) break;
                // Neighbour tetra index
                int tIdx = *itr; 
                //logger.info << "Edge Comparing tetra with index " << i << " and " << tIdx << logger.end; 
                // Check if edge is part of neighbour tetrahedron
                for( int nEdge=0; nEdge<6; nEdge++ ) {
                    // Get neighbour node index 1
                    int nIdx1 = tetrahedra[tIdx].GetNodeIndex(GetEdgeStartIndex(nEdge));
                    // Get neighbour node index 2
                    int nIdx2 = tetrahedra[tIdx].GetNodeIndex(GetEdgeEndIndex(nEdge));
                    
                    //logger.info << "comparing: " << idx1 << "," << idx2 << " == " << nIdx1 << "," << nIdx2;
                    // If edge is the same, tetra i and j shares an edge
                    if( (idx1 == nIdx1 && idx2 == nIdx2) || 
                        (idx1 == nIdx2 && idx2 == nIdx1) ) {
                        // Add neighbour tetrahedron index to edge sharing list
                        body->edgeSharing[(i*12)+(edge*2)+numTetraSharingEdge] = tIdx;
                        numTetraSharingEdge++;
                        break;
                    }
                }
            }
        }
    }
    /*
    for( unsigned int i=0; i<body->numTetrahedra; i++ ){
        logger.info << "----- Tetra #" << i << "------" << logger.end;
        for( int edge=0; edge<6; edge++ ) {
            logger.info << "Edge #" << edge << ": ";
            logger.info << body->edgeSharing[(i*12)+edge*2] << ",";
            logger.info << body->edgeSharing[(i*12)+edge*2+1] << logger.end; 
        }
    }

    logger.info << "Number of tetrahedrons: " << body->numTetrahedra << logger.end;
    logger.info << "Number of tetras with four neighbours: " << tetraWithFourNeighbours << logger.end;
    logger.info << "Number of faces in surface: " << solid->surface->numFaces << logger.end;
    */
}

