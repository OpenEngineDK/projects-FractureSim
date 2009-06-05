
#include "CrackStrategyOne.h"
#include <Logging/Logger.h>
#include <Math/Math.h>
#include <algorithm>

CrackStrategyOne::CrackStrategyOne() : crackInitialized(false) {
}

CrackStrategyOne::~CrackStrategyOne() { 
}


bool CrackStrategyOne::CrackInitialized(Solid* solid) {
    this->solid = solid;

    // Check if max stress has been exceeded
    if( !crackInitialized && solid->body->IsMaxStressExceeded() ) {
        // Crack first tetrahedron
        InitializeCrack(solid);
        logger.info << "CRACK HAS BEEN INITIALIZED" << logger.end;
    }
    return crackInitialized;
}


// Crack first tetrahedron 
void CrackStrategyOne::InitializeCrack(Solid* solid) {
     // Alloc buffer
    float4* principalStress = (float4*)malloc(sizeof(float4) * solid->body->numTetrahedra);    
    // Get principal stress for all tetrahedrons 
    solid->body->GetPrincipalStress(principalStress);
    normalize(principalStress[0]);
    logger.info << "[A] Principal Stress Plane Normal: " << 
        principalStress[0].x << ", " << principalStress[0].y << ", " << principalStress[0].z << logger.end;
 
    // Find tetrahedron with highest stress
    int maxStressTetraIndex;
    float maxStressValue = 0;
    for( unsigned int i=0; i<solid->body->numTetrahedra; i++) {
        if( abs(principalStress[i].w) > maxStressValue ) {
            maxStressValue = abs(principalStress[i].w);
            maxStressTetraIndex = i;
        }
    }
    //logger.info << "TetraId: " << maxStressTetraIndex << " MaxStress: " << maxStressValue << logger.end;

    // Get tetrahedron with highest stress value
    Tetrahedron tetra = solid->body->tetrahedraMainMem[maxStressTetraIndex];
    // Get the normal to the plane defining the principal stress
    float4 stressPlane = principalStress[maxStressTetraIndex];
    stressPlane.w = 0;
    stressPlane = normalize(stressPlane);

    // Save initial plane normal for boundary condition
    initTetraIdx = maxStressTetraIndex;
    initPlaneNorm = stressPlane;

    // Tetrahedron center of mass
    float4 node[4];
    solid->vertexpool->GetTetrahedronAbsPosition(tetra, &node[0]);
    float4 com = (node[0] + node[1] + node[2] + node[3]) / 4.0;

    // Plane-Tetrahedron intersection to determine crack points
    CrackTetrahedron(maxStressTetraIndex, stressPlane, com);
    // Remove initial tetrahedron from cracked set so it will be processed in first iteration.
    crackedTetrahedrons.clear();
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
    for( itr = crackFront.begin(); itr!=crackFront.end(); itr++ ) {
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
                numCrackPoints++;
            }
        logger.info << "Number of existing CrackPoints: " << numCrackPoints << logger.end;

        // Get normal to principal stress plane.
        float4 principalStressNorm = b->GetPrincipalStressNorm(tetraIdx);
        double invLen = 1.0f / sqrt( dot(make_float3(principalStressNorm),make_float3(principalStressNorm)) );
        principalStressNorm *= invLen;
        logger.info << "PrincipalStressNorm: " << principalStressNorm.x << ", " << principalStressNorm.y << ", " << principalStressNorm.z << logger.end;
        
        float4 planeNorm;
        float4 pointOnPlane;
        // Handle each of the four cases
        if( numCrackPoints == 2 ) {
            //float3 axis = normalize(make_float3(cp[1] - cp[0]));
            //float3 v = normalize( cross(axis, make_float3(principalStressNorm)) );
            //planeNorm = normalize( make_float4(cross(v, axis)));
            
            // Get neighbour plane normal
            float4 neighbourPlaneNorm;
            int* neighbour = b->GetNeighbours(tetraIdx);
            for( int i=0; i<4; i++ ){
                if( b->HasCrackPoints(neighbour[i]) ) {
                    neighbourPlaneNorm = b->GetPrincipalStressNorm(neighbour[i]);
                    neighbourPlaneNorm.w = 0.0f;
                }
            }
            logger.info << "NeighbourStressNorm: " << neighbourPlaneNorm.x << ", " << neighbourPlaneNorm.y << ", " << neighbourPlaneNorm.z << logger.end;
            
            // Calculate modified normal according to article (38)
            float4 AB = cp[1]-cp[0];
            float ABLengthSqr = pow(length(AB),2);
            planeNorm = neighbourPlaneNorm - ((dot(neighbourPlaneNorm, AB) / ABLengthSqr) * AB); 
            logger.info << "ResultingStressNorm: " << planeNorm.x << ", " << planeNorm.y << ", " << planeNorm.z << logger.end;

            // Angle between new plane and initial crack plane
            float dotp = dot(make_float3(planeNorm),make_float3(initPlaneNorm));
            float angle = acos( dotp / (length(planeNorm)*length(initPlaneNorm))) * (180 / Math::PI);
            if( angle > 90.0f ) {
                planeNorm *= -1;
                dotp = dot(make_float3(planeNorm),make_float3(initPlaneNorm));
                angle = acos( dotp / (length(planeNorm)*length(initPlaneNorm))) * (180 / Math::PI);
            }

            float angleLimit = 30.0f;
            static int MAX_ITR = 100;
            int iterations = 0;
            // If angle exceeds limit - recalculate plane normal
            while( angle > angleLimit ) {
                if( iterations++ > MAX_ITR ) break;

                logger.info << "Angle Between Planes: " << angle << " exceeds boundary of " << angleLimit << " recalc" << logger.end;
                float4 AB = cp[1]-cp[0];
                float ABLengthSqr = pow(length(AB),2);
                planeNorm += initPlaneNorm;
                //planeNorm = normalize(planeNorm);
                planeNorm = planeNorm - ((dot(planeNorm, AB) / ABLengthSqr) * AB); 
                //planeNorm = normalize(planeNorm);
                // Angle between new plane and initial crack plane
                float dotp = dot(make_float3(planeNorm),make_float3(initPlaneNorm));
                angle = acos( dotp / (length(planeNorm)*length(initPlaneNorm))) * (180 / Math::PI);
                if( angle > 90.0f ) {
                    planeNorm *= -1;
                    dotp = dot(make_float3(planeNorm),make_float3(initPlaneNorm));
                    angle = acos( dotp / (length(planeNorm)*length(initPlaneNorm))) * (180 / Math::PI);
                }
            }

            static float maxAngle = 0;
            if( angle > maxAngle ) maxAngle = angle;
            logger.info << "Angle Between Planes: " << angle << " (max:" << maxAngle << ")" << logger.end;
            
            pointOnPlane = cp[0];
        }
        else if( numCrackPoints == 3 || numCrackPoints == 4 ) {
            float3 v1 = normalize( make_float3(cp[1]-cp[0]) );
            float3 v2 = normalize( make_float3(cp[2]-cp[0]) );
            planeNorm = normalize( make_float4(cross(v1, v2)));
            pointOnPlane = cp[0];
        }else 
            logger.info << "Warning: tetra " << tetraIdx << " has less than 1 or more than 4 crack points" << logger.end;

        // If tetrahedron-plane intersections is successful, write crack point into neighbours
        if( CrackTetrahedron(tetraIdx, planeNorm, pointOnPlane) ) {
            // ------ Write crack points in neighbour tetrahedrons ------- //
            // For each crack point in tetrahedron
            for( int edgeIdx=0; edgeIdx<6; edgeIdx++ ){
                // Get crack point
                float cp = crackPoint[edgeIdx];
                // If edge has a crack..
                if( cp > 0 ){
                    // Get the two node indices
                    int nodeIdx0 = tetra.GetNodeIndex(GetEdgeStartIndex(edgeIdx));
                    int nodeIdx1 = tetra.GetNodeIndex(GetEdgeEndIndex(edgeIdx));
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
                            b->AddCrackPoint(tetraIndexSharingEdge[j], nodeIdx0, nodeIdx1, cp);
                            
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
                }
            }
        }
    }
    // Update crack front set
    crackFront.clear();
    tetraToBeCracked.unique();
    crackFront.merge(tetraToBeCracked);
    logger.info << "-----------------" << logger.end;
}

bool CrackStrategyOne::FragmentationDone() {
    return false;
}

// Returns true if plane intersects tetrahedron 
bool CrackStrategyOne::CrackTetrahedron(int tetraIdx, float4 planeNorm, float4 pointOnPlane){
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
            float4 cp = lp0 + (lp1 - lp0) * u;
            // Add crack point to tetrahedron
            solid->body->AddCrackPoint(tetraIdx, i, u);
            numIntersects++;
        }
    }
    // Add tetrahedron to cracked set.
    if( numIntersects >= 3 ){
        crackedTetrahedrons.push_back(tetraIdx);
        logger.info << " - numIntersects: " << numIntersects << " ..OK" << logger.end;
        return true;
    }
    else
        logger.info << " - numItersects: " << numIntersects << " FAILED!" << logger.end;
    return false;
}


void CrackStrategyOne::RenderDebugInfo() {
    float color = 0.0;
    std::list<int>::iterator itr;
 
    /*    glColor4f(1.0, 0, 0, 1.0);
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
    */
    for( itr=crackedTetrahedrons.begin(); itr!=crackedTetrahedrons.end(); itr++){
        int tetraIdx = *itr;
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
            if( crackPoint[i] > 0 ){
                float4 lp0 = node[GetEdgeStartIndex(i)]; // Line Point 0
                float4 lp1 = node[GetEdgeEndIndex(i)];   // Line Point 1
                float4 dir = lp1 - lp0;
                cp[numCrackPoints++] = lp0 + (dir * crackPoint[i]);
            }

        // Set color
        color += 0.05;
        if( color > 0.8 ) color = 0.1;
        glColor4f(color, color, color, 1.0);

        // Draw triangle
        if( numCrackPoints == 3 ){
            glBegin(GL_TRIANGLES);
            glVertex3f(cp[0].x, cp[0].y, cp[0].z);
            glVertex3f(cp[1].x, cp[1].y, cp[1].z);
            glVertex3f(cp[2].x, cp[2].y, cp[2].z);
            glEnd();
       
        } else
        // Draw quad
        if( numCrackPoints == 4 ){
            glBegin(GL_QUADS);
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
