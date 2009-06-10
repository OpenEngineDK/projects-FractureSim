
#include "CrackStrategyOne.h"
#include <Logging/Logger.h>
#include <Math/Math.h>
#include <algorithm>

inline bool operator==(const float4 l, const float4 r) {
    return l.x==r.x && l.y==r.y && l.z==r.z && l.w==r.w;
}

inline bool operator!=(const float4 l, const float4 r) {
    return !(l==r);
}

CrackStrategyOne::CrackStrategyOne() : crackInitialized(false) {
}

CrackStrategyOne::~CrackStrategyOne() { 
}


bool CrackStrategyOne::CrackInitialized(Solid* solid) {
    //    this->solid = solid;

    // Check if max stress has been exceeded
    if( !crackInitialized && solid->body->IsMaxStressExceeded() ) {
        // Crack first tetrahedron
        logger.info << "-------- CRACK INITIALIZATION ---------" << logger.end;
        InitializeCrack(solid);
        logger.info << "Press 'c' to apply crack tracking" << logger.end;
        logger.info << "----------------------------" << logger.end;
    }
    return crackInitialized;
}


// Crack first tetrahedron 
void CrackStrategyOne::InitializeCrack(Solid* solid) {
     // Alloc buffer
    float4* principalStress = (float4*)malloc(sizeof(float4) * solid->body->numTetrahedra);    
    // Get principal stress for all tetrahedrons 
    solid->body->GetPrincipalStress(principalStress);
 
    // Find tetrahedron with highest stress
    int maxStressTetraIndex = 0;
    float maxStressValue = 0;
    for( unsigned int i=0; i<solid->body->numTetrahedra; i++) {
        if( abs(principalStress[i].w) > maxStressValue ) {
            maxStressValue = abs(principalStress[i].w);
            maxStressTetraIndex = i;
        }
    }
    //logger.info << "TetraId: " << maxStressTetraIndex << " MaxStress: " << maxStressValue << logger.end;

    // Crack has now been initialized and will propagate from here.
    crackInitialized = true;
    // Add initial tetrahedron to crack front in order to add crack points to neighbours.
    crackFront.push_back(maxStressTetraIndex);
    // Free principal stress array
    free(principalStress);
}


void CrackStrategyOne::ApplyCrackTracking(Solid* solid) {
    Body* b = solid->body;
    // List of new crack front elements
    std::list<int> tetraToBeCracked;

    //logger.info << "Crack Front Set Size = " << crackFront.size() << logger.end; 
    // Print crack front set
    std::list<int>::iterator itr;
    logger.info << "CrackFront = {";
    for( itr = crackFront.begin(); itr!=crackFront.end(); itr++ )
        logger.info << *itr  << ",";
    logger.info << "}" << logger.end;

    // For each tetrahedron in crack front set
    if( !crackFront.empty() ) {
        itr = crackFront.begin();
        crackFront.pop_front();
    }else return;
    //    for( itr = crackFront.begin(); itr!=crackFront.end(); itr++ ) 
    {
        int tetraIdx = *itr;
        logger.info << "ApplyCrackTracking to tetra: " << tetraIdx << logger.end;

        // ------------ Calculate Own Crack Plane -------------------- //
        // Get tetra indices
        Tetrahedron tetra = b->tetrahedraMainMem[tetraIdx];
        // Get crack points for tetrahedron
        float* crackPoint = b->GetCrackPoints(tetraIdx);

        // Tetrahedron absolute coordinates
        float4 node[4];
        solid->vertexpool->GetTetrahedronAbsPosition(tetra, &node[0]);

        int numCrackPoints = 0;
        float4 cp[4];
        for( int i=0; i<6; i++ )
            if( crackPoint[i] > 0 ) { 
                float4 lp0 = node[GetEdgeStartIndex(i)]; // Line Point 0
                float4 lp1 = node[GetEdgeEndIndex(i)];   // Line Point 1
                float4 dir = lp1 - lp0;
                cp[numCrackPoints] = lp0 + (dir * crackPoint[i]);
                logger.info << "CP[" << numCrackPoints << "] " << cp[numCrackPoints].x << ", " << cp[numCrackPoints].y << ", " << cp[numCrackPoints].z << logger.end;
                numCrackPoints++;
            }
        logger.info << "Number of existing CrackPoints: " << numCrackPoints << logger.end;

        // Get normal to principal stress plane.
        float3 principalStressNorm = make_float3(b->GetPrincipalStressNorm(tetraIdx));
        principalStressNorm = normalize(principalStressNorm);
        logger.info << "PrincipalStressNorm: " << principalStressNorm.x << ", " << principalStressNorm.y << ", " << principalStressNorm.z << logger.end;
        
        float3 planeNorm = make_float3(0);
        float3 pointOnPlane = make_float3(0);
        static float maxAngle = 0;
 
       // Handle each of the five cases
        if( numCrackPoints == 0 ) {
            // Save initial plane normal for boundary condition
            initTetraIdx = tetraIdx;
            initPlaneNorm = principalStressNorm;
            // Save plane normal
            b->crackPlaneNorm[tetraIdx] = make_float4(initPlaneNorm);
            // Tetrahedron center of mass
            float4 node[4];
            solid->vertexpool->GetTetrahedronAbsPosition(tetra, &node[0]);
            // Set point on plane equal to center of mass 
            pointOnPlane = make_float3((node[0] + node[1] + node[2] + node[3]) / 4.0);
            // Set plane norm equal to initial principal stress norm
            planeNorm = initPlaneNorm;
        }
        else if( numCrackPoints == 1 ){
            logger.info << "UN HANDLED CASE NUMBER OF CRACK POINTS == 1" << logger.end;
        }
        else if( numCrackPoints == 2 ) {
            // Find neighbour tetra index
            int neighbourIdx = -1;
            int* nIndices = b->GetNeighbours(tetraIdx);
            for( int i=0; i<4; i++ ) {
                if( std::find(crackedTetrahedrons.begin(), 
                              crackedTetrahedrons.end(), nIndices[i]) != crackedTetrahedrons.end() ) {
                    // Found cracked neighbour
                    neighbourIdx = nIndices[i];
                    break;
                }
            }
            // Check that neighbour is found
            if( neighbourIdx == -1 )
                logger.info << "Warning: Cracked neighbour NOT found" << logger.end;
      
            // Get neighbour absolute node coordinates
            float4 neighbourNode[4];
            solid->vertexpool->GetTetrahedronAbsPosition(b->tetrahedraMainMem[neighbourIdx], &neighbourNode[0]);
            // Get neighbour crack points
            float* neighbourCrackPoint = b->GetCrackPoints(neighbourIdx);
            
            // Find absolute coordinate of a third point in neighbour (not in crack edge)
            float4 ncp[4];
            float4 thirdAbsNeighbourCrackPoint = make_float4(0);
            int numNeighbourCrackPoint = 0;
            for( int i=0; i<6; i++ )
                if( neighbourCrackPoint[i] > 0 ) { 
                    float4 lp0 = neighbourNode[GetEdgeStartIndex(i)]; // Line Point 0
                    float4 lp1 = neighbourNode[GetEdgeEndIndex(i)];   // Line Point 1
                    float4 dir = lp1 - lp0;
                    ncp[numNeighbourCrackPoint] = lp0 + (dir * neighbourCrackPoint[i]);
                    numNeighbourCrackPoint++;
                }

            for( int i=0; i<4; i++ ) {
                if( length(ncp[i]-cp[0])>Math::EPS && length(ncp[i]-cp[1])>Math::EPS ){
                    thirdAbsNeighbourCrackPoint = ncp[i];
                    logger.info << "Third crack point found: " << ncp[i].x << ", " << ncp[i].y << ", " << ncp[i].z << logger.end;
                    break;
                }
            }
            if( thirdAbsNeighbourCrackPoint == make_float4(0) )
                logger.info << "Warning: third neighbour crack point NOT found - cannot determine plane tangent" << logger.end;

            // Get neighbour crack plane normal
            float3 nPlaneNorm = make_float3(b->crackPlaneNorm[neighbourIdx]);
            if( length(nPlaneNorm) == 0 )
                logger.info << "Warning: Neighbour has no crack plane normal" << logger.end;

            // Calculate neighbour crack plane tangent
            float3 nTangent = make_float3(thirdAbsNeighbourCrackPoint - cp[0]);

            // Calculate crack edge
            float3 edge = make_float3(cp[1] - cp[0]);

            // Calculate crack plane normal
            float3 calcNeighbourNorm = cross(nTangent, edge);

            // Check calculate neighbour normal to its actual normal to determine the orientation 
            // of the crack edge
            float nDotp = dot(nPlaneNorm, calcNeighbourNorm);
            float nAngle = acos( nDotp / (length(nPlaneNorm)*length(calcNeighbourNorm))) * (180 / Math::PI);
            // If angle between the two normals are 180 degrees - flip the orientation of edge
            if( nAngle > 5.0 ) {
                float4 tmp = cp[0];
                cp[0] = cp[1]; cp[1] = tmp;
                edge *= -1;
            }
            // Re-Calculate perpendicular neighbour tangent
            nTangent = make_float3(thirdAbsNeighbourCrackPoint - cp[0]);
            calcNeighbourNorm = cross(nTangent, edge);
            nTangent = cross(calcNeighbourNorm, edge);

            // Calculate modified normal according to article (38)
            float3 AB = make_float3(cp[1]-cp[0]);
            float ABLengthSqr = pow(length(AB),2);
            planeNorm = principalStressNorm - ((dot(principalStressNorm, AB) / ABLengthSqr) * AB); 
            planeNorm = normalize(planeNorm); 
            logger.info << "ResultingStressNorm: " << planeNorm.x << ", " << planeNorm.y << ", " << planeNorm.z << logger.end;

            // Note which two crack points that defines the edge
            int cpIdx1 = -1;
            int cpIdx2 = -1;
            for( int i=0; i<6; i++ )
                if( crackPoint[i] > 0 ) { 
                    if( cpIdx1 < 0 ) cpIdx1 = i;
                    else cpIdx2 = i;
                }

            // Set point on plane equal to one of the edge points
            pointOnPlane = make_float3(cp[0]);

            // Crack the tetra to Find third crack point (not in crack edge)
            int newCpIdx = -1;
            if( CrackTetrahedron(solid, tetraIdx, make_float4(planeNorm), make_float4(pointOnPlane)) ){
                // Find new crack point
                for( int i=0; i<6; i++ )
                    if( crackPoint[i] > 0 && i!=cpIdx1 && i!=cpIdx2) 
                        newCpIdx = i;
            
            }

            // Create vector from crack edge to new crack point
            float4 lp0 = node[GetEdgeStartIndex(newCpIdx)]; // Line Point 0
            float4 lp1 = node[GetEdgeEndIndex(newCpIdx)];   // Line Point 1
            float4 dir = lp1 - lp0;
            float4 absCrackPoint = lp0 + (dir * crackPoint[newCpIdx]);
            // Calculate crack plane tangent   
            float3 tangent = make_float3(absCrackPoint - cp[0]);
            float3 calcNorm = cross(edge, tangent);
            tangent = cross(calcNorm, edge);

            // Calculate angel between this plane tangent and the neighbour plane tangent
            float dotp = dot(tangent, nTangent);
            float angle = acos( dotp / (length(tangent)*length(nTangent))) * (180 / Math::PI);

            // While the angle is 
            float angleLimit = 45.0f;
            static int MAX_ITR = 100;
            int iterations = 0;
            // If angle exceeds limit - recalculate plane normal
            while( angle > angleLimit ) {
                if( iterations++ > MAX_ITR ) break;
                logger.info << "Angle Between tangents: " << angle << " exceeds boundary of " << angleLimit << " recalc" << logger.end;
                //planeNorm += nPlaneNorm;
                principalStressNorm += nPlaneNorm;
                principalStressNorm = normalize(principalStressNorm);
                planeNorm = principalStressNorm - ((dot(principalStressNorm, AB) / ABLengthSqr) * AB); 
                planeNorm = normalize(planeNorm); 
                logger.info << "RecalculatedStressNorm: " << planeNorm.x << ", " << planeNorm.y << ", " << planeNorm.z << logger.end;
                //planeNorm = normalize(planeNorm);
                //planeNorm = planeNorm - ((dot(planeNorm, AB) / ABLengthSqr) * AB); 
                //planeNorm = normalize(planeNorm);
                
                // Crack the tetra to Find third crack point (not in crack edge)
                newCpIdx = -1;
                if( CrackTetrahedron(solid, tetraIdx, make_float4(planeNorm), make_float4(pointOnPlane)) ){
                    // Find new crack point
                    for( int i=0; i<6; i++ )
                        if( crackPoint[i] > 0 && i!=cpIdx1 && i!=cpIdx2) 
                            newCpIdx = i;    
                }
                if( newCpIdx == -1 )
                    logger.info << "Warning: could not find third crack point - cannot calculate tangent" << logger.end;

                // Create vector from crack edge to new crack point
                float4 lp0 = node[GetEdgeStartIndex(newCpIdx)]; // Line Point 0
                float4 lp1 = node[GetEdgeEndIndex(newCpIdx)];   // Line Point 1
                float4 dir = lp1 - lp0;
                float4 absCrackPoint = lp0 + (dir * crackPoint[newCpIdx]);
                // Calculate crack plane tangent   
                tangent = make_float3(absCrackPoint - cp[0]);
                calcNorm = cross(edge, tangent);
                tangent = cross(calcNorm, edge);
                
                // Calculate angel between this plane tangent and the neighbour plane tangent
                dotp = dot(tangent, nTangent);
                angle = acos( dotp / (length(tangent)*length(nTangent))) * (180 / Math::PI);
             
            }
    
            // DEBUG
            debugVector.clear();
            AddDebugVector(pointOnPlane, nTangent, make_float3(0,1,0));
            logger.info << "Adding nTangent: " << nTangent.x <<", " << nTangent.y<<", " <<nTangent.z << logger.end;
            AddDebugVector(pointOnPlane, tangent,  make_float3(0,0,1));
            logger.info << "Adding tangent: " << tangent.x <<", " << tangent.y<<", " <<tangent.z << logger.end;
            
            logger.info << "Angle between tangents: " << angle << logger.end;
        }
        else if( numCrackPoints == 3 ) {
            float3 v1 = normalize( make_float3(cp[1]-cp[0]) );
            float3 v2 = normalize( make_float3(cp[2]-cp[0]) );
            planeNorm = normalize( cross(v1, v2) );
      
            // Angle between new plane and initial crack plane
            float dotp = dot( planeNorm, initPlaneNorm );
            float angle = acos( dotp / (length(planeNorm)*length(initPlaneNorm))) * (180 / Math::PI);
            
            //      logger.info << "Angle Between Planes: " << angle << logger.end;
              
            if( angle > 90.0f ) {
                planeNorm *= -1;
                dotp = dot( planeNorm, initPlaneNorm);
                angle = acos( dotp / (length(planeNorm)*length(initPlaneNorm))) * (180 / Math::PI);
            }
            if( angle > maxAngle ) maxAngle = angle;
            logger.info << "Angle Between Planes: " << angle << " (max:" << maxAngle << ")" << logger.end;
      
            pointOnPlane = make_float3(cp[0]);
        }else if( numCrackPoints == 4 ) {
            logger.info << "ERROR: case with 4 crack points NOT YET IMPLEMENTED" << logger.end;
        }
        else
            logger.info << "Warning: tetra " << tetraIdx << " has less than 1 or more than 4 crack points" << logger.end;

        // If tetrahedron-plane intersections is successful, write crack point into neighbours
        if( CrackTetrahedron(solid, tetraIdx, make_float4(planeNorm), make_float4(pointOnPlane)) ) {
            // Cracking tetrahedron was successful - add to set of cracked tetras 
            crackedTetrahedrons.push_back(tetraIdx);
            // Save crack plane normal
            b->crackPlaneNorm[tetraIdx] = make_float4(planeNorm);

            debugPlaneNorm.push_back(planeNorm);
            debugPointOnPlane.push_back(pointOnPlane);

            // ------ Write crack points in neighbour tetrahedrons ------- //
            // For each crack point in tetrahedron
            for( int edgeIdx=0; edgeIdx<6; edgeIdx++ ){
                // Get crack point
                float cp = crackPoint[edgeIdx];
                // If edge has a crack..
                if( cp > 0 ){
                    
                    // Get the two node indices defining the edge that has a crack point
                    int nIdx1 = tetra.GetNodeIndex(GetEdgeStartIndex(edgeIdx));
                    int nIdx2 = tetra.GetNodeIndex(GetEdgeEndIndex(edgeIdx));

                    // Look up neighbours sharing cracked edge.
                    int tetraIndexSharingEdge[2];
                    tetraIndexSharingEdge[0] = b->edgeSharing[(tetraIdx*12)+(edgeIdx*2)];
                    tetraIndexSharingEdge[1] = b->edgeSharing[(tetraIdx*12)+(edgeIdx*2)+1];

                    // For each neighbour sharing edge, add crack point.
                    for( int j=0; j<2; j++){
                        // If neighbour exists and not already cracked...
                        if( tetraIndexSharingEdge[j] != -1 && 
                            std::find( crackedTetrahedrons.begin(), 
                                       crackedTetrahedrons.end(), 
                                       tetraIndexSharingEdge[j]) == crackedTetrahedrons.end()) {

                            // Add crack point to neighbour
                            b->AddCrackPoint(tetraIndexSharingEdge[j], nIdx1, nIdx2, cp);
                            
                            // Add tetrahedron index to be cracked next time, if not already added or cracked.
                            if( std::find( tetraToBeCracked.begin(), 
                                           tetraToBeCracked.end(), 
                                           tetraIndexSharingEdge[j]) == tetraToBeCracked.end() &&
                                std::find( crackFront.begin(), 
                                           crackFront.end(), 
                                           tetraIndexSharingEdge[j]) == crackFront.end()) {
                                tetraToBeCracked.push_back(tetraIndexSharingEdge[j]);

                                logger.info << "Adding neighbour tetra to be cracked: " << tetraIndexSharingEdge[j] << logger.end;
                            }
                        }
                    }

                    // Add crack point to all tetrahedrons sharing edge with crack point
                    // For each tetrahedron in body..
                    for( int i=0; i<(int)b->numTetrahedra; i++ ) {
                        if( std::find( crackedTetrahedrons.begin(), 
                                       crackedTetrahedrons.end(), i) == crackedTetrahedrons.end()){
                        
                            // Add crack point to edge if shared
                            if( b->AddCrackPoint(i, nIdx1, nIdx2, cp) )
                                logger.info << "Adding crack point to non-neighbour " << i << " sharing same edge " << 
                                    cp << logger.end;
                        }
                    }
                    
                }
            }
        }
    }
    // Update crack front set
    //crackFront.clear();
    tetraToBeCracked.unique();
    
  
    while( !tetraToBeCracked.empty() ) {
        crackFront.push_back(*tetraToBeCracked.begin());
        tetraToBeCracked.pop_front();
    }
    logger.info << "-----------------" << logger.end;
}

bool CrackStrategyOne::FragmentationDone() {
    return crackFront.empty();
}

// Returns true if plane intersects tetrahedron 
bool CrackStrategyOne::CrackTetrahedron(Solid* solid, int tetraIdx, float4 planeNorm, float4 pointOnPlane){
    // Check if tetrahedron is already cracked (init case)
    if( std::find(crackedTetrahedrons.begin(), 
                  crackedTetrahedrons.end(), 
                  tetraIdx) != crackedTetrahedrons.end() ) {
        // Tetra already cracked -> return
        logger.info << "WARNING: tetra " << tetraIdx << " is alread cracked" << logger.end;
        return false;
    }

    logger.info << "Cracking tetra " << tetraIdx << " with planeNorm " << planeNorm.x << ", " << planeNorm.y << ", " << planeNorm.z;

    // Get tetrahedron with highest stress value
    Tetrahedron tetra = solid->body->tetrahedraMainMem[tetraIdx];

    // Tetrahedron center of mass
    float4 node[4];
    solid->vertexpool->GetTetrahedronAbsPosition(tetra, &node[0]);

    // Iterate through all 6 edges
    int numIntersects = 0;
    for( int i=0; i<6; i++ ){
        float4 lp0 = node[GetEdgeStartIndex(i)]; // Line Point 0
        float4 lp1 = node[GetEdgeEndIndex(i)];   // Line Point 1
     
        // Plane line intersection explained here:
        // http://local.wasp.uwa.edu.au/~pbourke/geometry/planeline/
        // u = (n dot (p0 - l0)) / (n dot (l1 - l0))
        // where n = plane normal, p0 lies on the plane, l0 and l1 is the line.
        // if u is between 0 and 1 the line segment intersects the plane.
        //double nominator = dot( planeNorm, (pointOnPlane - lp0));
        float4 tmp = pointOnPlane - lp0;
        double nominator = planeNorm.x*tmp.x + planeNorm.y*tmp.y + planeNorm.z*tmp.z;
        if( nominator < Math::EPS && nominator > -Math::EPS ) nominator = 0;  

        float4 dir = lp1 - lp0;
        double denomitor = planeNorm.x*dir.x + planeNorm.y*dir.y + planeNorm.z*dir.z;
        if( denomitor < Math::EPS && denomitor > -Math::EPS ) denomitor = 0;  
        
        double u = nominator / denomitor;

        //logger.info << "nominator: " << nominator << ", den: " << denomitor << ", u= " << u << logger.end;
        //double u = dot( planeNorm, (pointOnPlane - lp0)) / dot(planeNorm, (lp1 - lp0));

        // Each tetrahedron has 6 edges and therefore 6 possible edge intersections.
        // If the plane intersects edge 4 [y,w] the ratio (0.0-1.0) from first edge point (y) is 
        // saved at the 4'th position in crackPoints.
        if( u > Math::EPS && u < 1.0-Math::EPS ) {
            //logger.info << "Plane Intersects Edge " << i << " u = " << u << logger.end;
            //float4 cp = lp0 + (lp1 - lp0) * u;
            //           logger.info << "Adding crackpoint to " << tetraIdx << " - " << cp.x << ", " << cp.y << ", " << cp.z << logger.end;
            // Add crack point to tetrahedron
            solid->body->AddCrackPoint(tetraIdx, i, u);
            numIntersects++;
        }
        else // Clear crack point
            solid->body->AddCrackPoint(tetraIdx, i, 0);
    }
    // Add tetrahedron to cracked set.
    if( numIntersects >= 3 ){
        logger.info << " - numIntersects: " << numIntersects << " ..OK" << logger.end;
        return true;
    }
    else
        logger.info << " - numItersects: " << numIntersects << " FAILED!" << logger.end;
    return false;
}


void CrackStrategyOne::RenderDebugInfo(Solid* solid) {
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

    float color = 0.5;
    std::list<int>::iterator itr;
 
    glColor4f(1.0, 0, 0, 1.0);
    glPointSize(5.0);
    glBegin(GL_POINTS);
    for( itr=crackedTetrahedrons.begin(); itr!=crackedTetrahedrons.end(); itr++){
        int tetraIdx = *itr;
        Tetrahedron tetra = solid->body->tetrahedraMainMem[tetraIdx];
        float4 node[4];
        solid->vertexpool->GetTetrahedronAbsPosition(tetra, &node[0]);
        
        for(int i=0; i<6; i++) {
            float crackPoint = solid->body->crackPoints[tetraIdx*6 + i];
            if( crackPoint > 0 ) {
                float4 lp0 = node[GetEdgeStartIndex(i)]; // Line Point 0
                float4 lp1 = node[GetEdgeEndIndex(i)];   // Line Point 1
                float4 dir = lp1 - lp0;
                float4 pos = lp0 + (dir * crackPoint);
                glVertex3f(pos.x, pos.y, pos.z);
            }
        }
    }
    glEnd();


    // ------ Render debug vectors ----------------- //
    glLineWidth(3.0);
    glBegin(GL_LINES);
    std::list<float3>::iterator vItr;
    for( vItr=debugVector.begin(); vItr!=debugVector.end();  ){
        float3 pos = *vItr++;
        float3 dir = *vItr++;
        float3 col = *vItr++;
        glColor4f(col.x, col.y, col.z, 1.0);
        glVertex3f(pos.x, pos.y, pos.z);
        glVertex3f(dir.x+pos.x, dir.y+pos.y, dir.z+pos.z);
    }
    glEnd();


    /*    // ----- Render plane normals -------- //
    
    std::list<float3>::iterator pointItr = debugPointOnPlane.begin();;

    glLineWidth(3.0);
    glBegin(GL_LINES);
    for( ; planeItr!=debugPlaneNorm.end(); planeItr++, pointItr++ ) {
        float3 norm = *planeItr;
        float3 point = *pointItr;
        
        //logger.info << "CP" << point.x << ", " << point.y << ", " << point.z << logger.end;
        glColor4f(0,1,0,1);
        glVertex3f(point.x, point.y, point.z);
        glColor4f(1,0,0,1);
        glVertex3f(norm.x+point.x, norm.y+point.y, norm.z+point.z);
    }
    glEnd();
    */



    // -------- Render crack planes --------------- //
    std::list<float3>::iterator planeItr = debugPlaneNorm.begin();
    for( itr=crackedTetrahedrons.begin(); itr!=crackedTetrahedrons.end(); itr++, planeItr++){
        int tetraIdx = *itr;
        float3 norm = *planeItr;
        Tetrahedron tetra = solid->body->tetrahedraMainMem[tetraIdx];
        float4 node[4];
        solid->vertexpool->GetTetrahedronAbsPosition(tetra, &node[0]);
        
        // Count number of crack points
        int numCrackPoints = 0;
        // Valid crack points
        float4 cp[4];
        // Get crack points for tetrahedron
        float* crackPoint = solid->body->GetCrackPoints(tetraIdx);
        for(int i=0; i<6; i++)
            if( crackPoint[i] > 0 ) {
                float4 lp0 = node[GetEdgeStartIndex(i)]; // Line Point 0
                float4 lp1 = node[GetEdgeEndIndex(i)];   // Line Point 1
                float4 dir = lp1 - lp0;
                cp[numCrackPoints++] = lp0 + (dir * crackPoint[i]);
            }

        // Set color
        //color += 0.05;
        if( color > 0.8 ) color = 0.1;
        glColor4f(color, color, color, 1.0);

        // Draw triangle
        if( numCrackPoints == 3 ){
            glBegin(GL_TRIANGLES);
            glNormal3f(norm.x, norm.y, norm.z);
            glVertex3f(cp[0].x, cp[0].y, cp[0].z);
            glVertex3f(cp[1].x, cp[1].y, cp[1].z);
            glVertex3f(cp[2].x, cp[2].y, cp[2].z);
            glEnd();
       
        } else
        // Draw quad
        if( numCrackPoints == 4 ){
            glBegin(GL_QUADS);
            glNormal3f(norm.x, norm.y, norm.z);
            glVertex3f(cp[0].x, cp[0].y, cp[0].z);
            glVertex3f(cp[1].x, cp[1].y, cp[1].z);
            glVertex3f(cp[3].x, cp[3].y, cp[3].z);
            glVertex3f(cp[2].x, cp[2].y, cp[2].z);            
            glEnd();
        }    
    }
    




    /*
    glColor4f(0.4, 0.6, 0.5 ,0.5);
    glBegin(GL_TRIANGLES);
    for( itr = cpTri.begin(); itr!=cpTri.end(); itr++ ){
        float4 p = *itr;
        glVertex3f(p.x, p.y, p.z);
    }
    glEnd();
    */
    /*   logger.info << "[A] Principal Stress Plane Normal: " << 
        planeNorm.x << ", " << planeNorm.y << ", " << planeNorm.z << logger.end;
    */
    /*
    glLineWidth(4.0);
    glColor4f(0, 1.0, 0, 1.0);
    glBegin(GL_LINES);
    glVertex3f(planeNorm.x*10.0, planeNorm.y*10.0, planeNorm.z*10.0);
    glEnd();
    */
 
    /*    glColor4f(0, 1.0, 0, 1.0);
    glLineWidth(4.0);
 
    // Render lines from center of mass to crack points for all tetras in crack front
    std::list<int>::iterator cfItr;
    for( cfItr=crackFront.begin(); cfItr!=crackFront.end(); cfItr++){
        int tetraIdx = *cfItr;
        // Get tetrahedron with highest stress value
        Tetrahedron tetra = solid->body->tetrahedraMainMem[tetraIdx];
 
        // Tetrahedron center of mass
        float4 node[4];
        solid->vertexpool->GetTetrahedronAbsPosition(tetra, &node[0]);
        float4 com = (node[0] + node[1] + node[2] + node[3]) / 4.0;

        glBegin(GL_LINES);
        for(int i=0; i<6; i++) {
            float crackPoint = solid->body->crackPoints[tetraIdx*6 + i];
            if(  crackPoint > 0 ){
                float4 lp0 = node[GetEdgeStartIndex(i)]; // Line Point 0
                float4 lp1 = node[GetEdgeEndIndex(i)];   // Line Point 1

                float4 dir = lp1 - lp0;
                float4 pos = lp0 + (dir * crackPoint);
                glVertex3f(pos.x, pos.y, pos.z);
                glVertex3f(com.x, com.y, com.z);
            }
        }
        glEnd();

    }
    */
}

void CrackStrategyOne::AddDebugVector(float3 pos, float3 dir, float3 color) {
    debugVector.push_back(pos);
    debugVector.push_back(dir);
    debugVector.push_back(color);
}
