#include "CoordSystemNode.h"

#include <Math/Vector.h>
#include <Math/Matrix.h>
#include <Geometry/Line.h>
#include <Meta/OpenGL.h>
#include <Logging/Logger.h>
#include "Math/eig3.h"

using namespace OpenEngine::Math;

CoordSystemNode::CoordSystemNode() {
}

void CoordSystemNode::Apply(Renderers::IRenderingView* view) {
    // draw coordinate system axis
    view->GetRenderer()->
        DrawLine( Geometry::Line(Math::Vector<3,float>(0.0),
                                 Math::Vector<3,float>(1000.0,0.0,0.0) ),
                  Math::Vector<3,float>(1.0,0.0,0.0) );
    view->GetRenderer()->
        DrawLine( Geometry::Line(Math::Vector<3,float>(0.0),
                                 Math::Vector<3,float>(0.0,1000.0,0.0) ),
                  Math::Vector<3,float>(0.0,1.0,0.0) );
    view->GetRenderer()->
        DrawLine( Geometry::Line(Math::Vector<3,float>(0.0),
                                 Math::Vector<3,float>(0.0,0.0,1000.0) ),
                  Math::Vector<3,float>(0.0,0.0,1.0) );

    return;
    // --------- test eigenvectors ---------------- //
    //glMatrixMode(GL_PROJECTION);
    // Reset The Projection Matrix
    //glLoadIdentity();

    //glMatrixMode(GL_PROJECTION);
      // Select the modelview matrix
    //glMatrixMode(GL_MODELVIEW);

    /*
    glTranslatef(1.0, 1.0, 1.0);
    glRotatef(45.0,0,0,1);

    float mView[ 16 ];
    //    glGetFloatv( GL_MODELVIEW_MATRIX, mView );
    glGetFloatv( GL_PROJECTION_MATRIX, mView );

    logger.info << "[ "<<mView[0]<<","<<mView[1]<<","<<mView[2]<<","<<mView[3]
                << "  "<<mView[4]<<","<<mView[5]<<","<<mView[6]<<","<<mView[7]
                << "  "<<mView[8]<<","<<mView[9]<<","<<mView[10]<<","<<mView[11]
                << "  "<<mView[12]<<","<<mView[13]<<","<<mView[14]<<","<<mView[15]<<"]"<<logger.end;

    
    glBegin(GL_LINES);
    glColor3f(1,0,0);
    glVertex3f(0,0,0);
    glVertex3f(10,0,0);
    glColor3f(0,1,0);
    glVertex3f(0,0,0);
    glVertex3f(0,10,0);
    glColor3f(0,0,1);
    glVertex3f(0,0,0);
    glVertex3f(0,0,10);
    glEnd();
    */

    glPushMatrix();

    Matrix<3,3,float> rot(0.866, -0.5,   0,
                          0.5  , 0.866, 0,
                          0,     0,     1);

   Matrix<3,3,float> rotT(0.866, 0.5,   0,
                          0.5  , 0.866, 0,
                          0,     0,     1);

    Vector<3,float> vec(1,0,0);

    Vector<3,float> res = rot * vec;
    

    glPushMatrix();
    glRotatef(45.0, 0,0,1);
    glLineWidth(5.0);
    glBegin(GL_LINES);
    glColor3f(1,1,0); // yellow
    glVertex3f(0,0,0);
    glVertex3f(10,0,0);

    
    //    glVertex3f(res[0],res[1],res[2]);
    /*   glVertex3f(V[0][0], V[1][0], V[2][0]);
    glColor3f(0,1,0); // green
    glVertex3f(0,0,0);
    glVertex3f(V[0][1], V[1][1], V[2][1]);
    glColor3f(0,0,1); // blue
    glVertex3f(0,0,0);
    glVertex3f(V[0][2], V[1][2], V[2][2]);
    */   glEnd();

    glPopMatrix();

    // Calculate eigen vectors
    double A[3][3];
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            A[i][j] = rotT(i,j);

    double V[3][3];
    double d[3];

    eigen_decomposition(A,V,d);

    glLineWidth(2.0);
    glBegin(GL_LINES);
    glColor3f(1,0,0); // red
    glVertex3f(0,0,0);
    glVertex3f(V[0][0], V[1][0], V[2][0]);
    glColor3f(0,1,0); // green
    glVertex3f(0,0,0);
    glVertex3f(V[0][1], V[1][1], V[2][1]);
    glColor3f(0,0,1); // blue
    glVertex3f(0,0,0);
    glVertex3f(V[0][2], V[1][2], V[2][2]);
    glEnd();

    glPopMatrix();   
}

