#include "TLEDNode.h"

#include <string>
#include <iostream>

#include <Math/Vector.h>
#include <Geometry/Line.h>

using OpenEngine::Math::Vector;
using OpenEngine::Geometry::Line;

TLEDNode::TLEDNode(std::string meshFile, std::string surfaceFile) {
    this->meshFile = meshFile;
    this->surfaceFile = surfaceFile;
    state = NULL;
    mesh = NULL;
    surface = NULL;
}

TLEDNode::~TLEDNode() {

}

void TLEDNode::Handle(Core::InitializeEventArg arg) {
    std::cout << "TLEDNode initialization start" << std::endl;
	state = new TetrahedralTLEDState(); 
    std::cout << "loading mesh" << std::endl;
    mesh = loadMesh(meshFile.c_str());
    std::cout << "loading obj" << std::endl;
    surface = loadSurfaceOBJ(surfaceFile.c_str());
    std::cout << "pre computing" << std::endl;
	precompute(mesh, state, 0.001f, 0.0f, 0.0f, 1007.0f, 49329.0f, 0.5f, 10.0f);
    std::cout << "TLEDNode initialization done" << std::endl;
}

void TLEDNode::Handle(Core::ProcessEventArg arg) {
    if (state == NULL || mesh == NULL || surface == NULL) return;
	for (int i=0; i<25; i++) {
		calculateGravityForces(mesh, state); 
		doTimeStep(mesh, state);
		applyFloorConstraint(mesh, state, -10); 
	}
}

void TLEDNode::Handle(Core::DeinitializeEventArg arg) {
}

void TLEDNode::Apply(Renderers::IRenderingView* view) {
    if (state == NULL || mesh == NULL || surface == NULL) return;
    display(0,mesh, state, surface);
}
