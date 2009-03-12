#include "TLEDNode.h"
#include "MshObjLoader.h"
#include "TetGenLoader.h"
#include "TypeConverter.h"
#include "CUDA.h"

#include <string>
#include <iostream>

#include <Math/Vector.h>
#include <Geometry/Line.h>
#include <Logging/Logger.h>

using OpenEngine::Math::Vector;
using OpenEngine::Geometry::Line;

TLEDNode::TLEDNode(std::string meshFile, std::string surfaceFile) {
    this->meshFile = meshFile;
    this->surfaceFile = surfaceFile;
    state = NULL;
    body = NULL;
    surface = NULL;
    vertexpool = NULL;
}

TLEDNode::~TLEDNode() {}

void TLEDNode::Handle(Core::InitializeEventArg arg) {
    INITIALIZE_CUDA();
    logger.info << "CUDA info:" << PRINT_CUDA_DEVICE_INFO() << logger.end;
    logger.info << "TLEDNode initialization start" << logger.end;

    ISolidLoader* loader = NULL;
    loader = new MshObjLoader(meshFile, surfaceFile);
    //loader = new TetGenLoader("./projects/TLED/data/RegistrationShapes/tetrahedron.ascii.1.node", "./projects/TLED/data/RegistrationShapes/tetrahedron.ascii.1.ele", "./projects/TLED/data/RegistrationShapes/tetrahedron.ascii.1.smesh");
    //loader = new TetGenLoader("./projects/TLED/data/RegistrationShapes/box.ascii.1.node", "./projects/TLED/data/RegistrationShapes/box.ascii.1.ele", "./projects/TLED/data/RegistrationShapes/box.ascii.1.smesh");
    loader->Load();

    logger.info << "number of vertices: "
                << loader->GetVertexPool().size() << logger.end;
    logger.info << "number of body tetrahedra: " 
                << loader->GetBody().size() << logger.end;
    logger.info << "number of surface triangles: " 
                << loader->GetSurface().size() << logger.end;

    vertexpool = TypeConverter
        ::ConvertToVertexPool(loader->GetVertexPool());
    //vertexpool->Scale(1.1);
    //vertexpool->Move(10,10,10);
    vertexpool->ConvertToCuda();

    body = TypeConverter
        ::ConvertToBody(loader->GetBody());

    surface = TypeConverter
        ::ConvertToSurface(loader->GetSurface());

	state = new TetrahedralTLEDState(); 

    solid = new Solid();
    solid->state = state;
    solid->body = body;
    solid->surface = surface;
    solid->vertexpool = vertexpool;

    logger.info << "pre computing" << logger.end;
	precompute(solid, 0.001f, 0.0f, 0.0f, 1007.0f, 49329.0f, 0.5f, 10.0f);
    logger.info << "TLEDNode initialization done" << logger.end;

    PolyShape ps("box.obj");

    // Initialize the Visualizer
    visualizer = new Visualizer();
    //visualizer->AllocBuffer(ELM_CENTER_OF_MASS, solid->body->numTetrahedra, LINES);
    visualizer->AllocPolyBuffer(STRESS_TENSORS, solid->body->numTetrahedra, ps);
}

void TLEDNode::Handle(Core::ProcessEventArg arg) {
    if (!solid->IsInitialized()) return;
	for (unsigned int i=0; i<25; i++) {
		calculateGravityForces(solid);
		doTimeStep(solid);
		applyFloorConstraint(solid, -10); 
	}
}

void TLEDNode::Handle(Core::DeinitializeEventArg arg) {
    if (solid != NULL)
        delete solid;
}

void TLEDNode::Apply(Renderers::IRenderingView* view) {
    if (!solid->IsInitialized()) return;
    
	// Map all visual buffers
    visualizer->MapAllBufferObjects();
    //
    display(0, solid, visualizer->GetBuffer(STRESS_TENSORS));
    //display(0, solid, visualizer->GetBuffer(ELM_CENTER_OF_MASS));
	
    // Unmap all visual buffers
    visualizer->UnmapAllBufferObjects();
    // Render all debug information
    visualizer->Render();
}
