#include "TLEDNode.h"

#include <string>
#include <iostream>

#include <Math/Vector.h>
#include <Geometry/Line.h>

using OpenEngine::Math::Vector;
using OpenEngine::Geometry::Line;

TLEDNode::TLEDNode() {
    state = NULL;
    mesh = NULL;
    surface = NULL;
}

TLEDNode::~TLEDNode() {

}

void TLEDNode::Handle(Core::InitializeEventArg arg) {
	state = new TetrahedralTLEDState(); 

    std::string dataDir = "projects/TLED/data/RegistrationShapes/";
    std::string meshFile = dataDir + "PROSTATE.msh";
    std::string objFile = dataDir + "PROSTATE.obj";

    std::cout << "loading mesh" << std::endl;
    mesh = loadMesh(meshFile.c_str());
    std::cout << "loading obj" << std::endl;
    surface = loadSurfaceOBJ(objFile.c_str());
    std::cout << "pre computing" << std::endl;
	precompute(mesh, state, 0.001f, 0.0f, 0.0f, 1007.0f, 49329.0f, 0.5f, 10.0f);
    std::cout << "done" << std::endl;
}

void TLEDNode::Handle(Core::ProcessEventArg arg) {
    if (state == NULL || mesh == NULL || surface == NULL) return;

	for (int i=0; i<25; i++) {
		calculateGravityForces(mesh, state); 
		doTimeStep(mesh, state);
		applyFloorConstraint(mesh, state, -10); 
	}

    /*
    static unsigned int counter = 0;
    counter++;
    std::cout << "display:" << counter << std::endl;
    */
}

void TLEDNode::Handle(Core::DeinitializeEventArg arg) {
}

void TLEDNode::Apply(Renderers::IRenderingView* view) {
    if (state == NULL || mesh == NULL || surface == NULL) return;
    display(0,mesh, state, surface);

    view->GetRenderer()->DrawLine( Line(Vector<3,float>(0.0),
                                        Vector<3,float>(1000.0,0.0,0.0) ),
                                   Vector<3,float>(1.0,0.0,0.0) );
    view->GetRenderer()->DrawLine( Line(Vector<3,float>(0.0),
                                        Vector<3,float>(0.0,1000.0,0.0) ),
                                   Vector<3,float>(0.0,1.0,0.0) );
    view->GetRenderer()->DrawLine( Line(Vector<3,float>(0.0),
                                        Vector<3,float>(0.0,0.0,-1000.0) ),
                                   Vector<3,float>(0.0,0.0,1.0) );

    //view->GetViewport()->GetViewingVolume()->LookAt();
}
