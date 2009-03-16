#include "TLEDNode.h"
#include "MshObjLoader.h"
#include "TetGenLoader.h"
#include "TypeConverter.h"
#include "Precompute.h"

#include "Physics_kernels.h"

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

TLEDNode::~TLEDNode() {
    // Delete VBO manager
    delete vbom;
}

void TLEDNode::Handle(Core::InitializeEventArg arg) {
    INITIALIZE_CUDA();
    logger.info << "CUDA info:" << PRINT_CUDA_DEVICE_INFO() << logger.end;
    logger.info << "TLEDNode initialization start" << logger.end;

    ISolidLoader* loader = NULL;
    loader = new MshObjLoader(meshFile, surfaceFile);
    /*
    std::string dataDir = "projects/TLED/data/RegistrationShapes/";
    loader = new MshObjLoader( dataDir + "sphere.msh", 
                               dataDir + "sphere.obj");
    */
    /*
    loader = new TetGenLoader
    ("./projects/TLED/data/RegistrationShapes/tetrahedron.ascii.1.node",
     "./projects/TLED/data/RegistrationShapes/tetrahedron.ascii.1.ele",
     "./projects/TLED/data/RegistrationShapes/tetrahedron.ascii.1.smesh");

    loader = new TetGenLoader
        ("./projects/TLED/data/RegistrationShapes/box.ascii.1.node", 
         "./projects/TLED/data/RegistrationShapes/box.ascii.1.ele", 
         "./projects/TLED/data/RegistrationShapes/box.ascii.1.smesh");
*/
    /*
    loader = new TetGenLoader
        ("/Users/cpvc/bunny.ascii.1.node", 
         "/Users/cpvc/bunny.ascii.1.ele", 
         "/Users/cpvc/bunny.ascii.1.smesh");
    */

    loader->Load();

    logger.info << "number of vertices: "
                << loader->GetVertexPool().size() << logger.end;
    logger.info << "number of body tetrahedra: " 
                << loader->GetBody().size() << logger.end;
    logger.info << "number of surface triangles: " 
                << loader->GetSurface().size() << logger.end;

    vertexpool = TypeConverter
        ::ConvertToVertexPool(loader->GetVertexPool());
    //vertexpool->Scale(30);
    //vertexpool->Move(0,10,0);

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
	//precompute(solid, 0.001f, 0.0f, 0.0f, 1007.0f, 49329.0f, 0.5f, 10.0f);
    precompute(solid, 0.001f, 0.0f, 0.0f, 10007.0f, 5500.0f, 0.5f, 10.0f);
    logger.info << "TLEDNode initialization done" << logger.end;

    // Load polygon model for visualization
    PolyShape ps("box.obj");

    // Initialize the Visualizer
    vbom = new VboManager();

    // Alloc buffers
    vbom->AllocBuffer(SURFACE_VERTICES, solid->surface->numFaces, GL_TRIANGLES);
    vbom->AllocBuffer(SURFACE_NORMALS,  solid->surface->numFaces, GL_POINTS);

    vbom->AllocBuffer(CENTER_OF_MASS, solid->body->numTetrahedra, GL_POINTS);
    vbom->AllocBuffer(BODY_MESH, solid->body->numTetrahedra*4, GL_TRIANGLES);
    vbom->AllocBuffer(BODY_COLORS, solid->body->numTetrahedra*4, GL_TRIANGLES);
    vbom->AllocBuffer(STRESS_TENSORS, solid->body->numTetrahedra, ps);
    

    // Disabled to bypass normal rendering
    vbom->Disable(BODY_MESH);
    vbom->Disable(BODY_COLORS);

    /*vbom->Disable(SURFACE_NORMALS);
    vbom->Disable(CENTER_OF_MASS);
    */
 
    printf("[VboManager] Total Bytes Allocated: %i\n", totalByteAlloc);
    // Buffer setup
    vbom->GetBuf(CENTER_OF_MASS).SetColor(0.0, 0.0, 1.0, 1.0);
}

void TLEDNode::Handle(Core::ProcessEventArg arg) {
    if (!solid->IsInitialized()) return;
	for (unsigned int i=0; i<25; i++) {
        calculateGravityForces(solid);
        calculateInternalForces(solid);
        updateDisplacement(solid);
        applyFloorConstraint(solid, 0); 
	}

	// Update all visualization data
    vbom->MapAllBufferObjects();    
    //updateSurface(solid, vbom);
    updateBodyMesh(solid, vbom, 0.0);
    //updateCenterOfMass(solid, vbom);
    updateStressTensors(solid, vbom);    
    vbom->UnmapAllBufferObjects();
    
    //vbom->dumpBufferToFile("./dump.txt", vbom->GetBuf(BODY_COLORS));
    //exit(-1);
}

void TLEDNode::Handle(Core::DeinitializeEventArg arg) {
    if (solid != NULL)
        solid->DeAlloc();
    //cleanupDisplay();
}

void TLEDNode::Apply(Renderers::IRenderingView* view) {
    if (!solid->IsInitialized()) return;

    vbom->Render();

    vbom->Render(vbom->GetBuf(BODY_MESH), 
                 vbom->GetBuf(BODY_COLORS));
}
