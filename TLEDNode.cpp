#include "TLEDNode.h"
#include "MshObjLoader.h"
#include "TetGenLoader.h"
#include "TypeConverter.h"
#include "Precompute.h"
#include "Physics_kernels.h"
#include "HelperFunctions.h"

#include <string>
#include <iostream>

#include <Math/Vector.h>
#include <Logging/Logger.h>

TLEDNode::TLEDNode(std::string meshFile, std::string surfaceFile) {
    this->meshFile = meshFile;
    this->surfaceFile = surfaceFile;
    state = NULL;
    body = NULL;
    surface = NULL;
    vertexpool = NULL;
    numIterations = 25;
    paused = false;
    renderPlane = true;
    minX = 220.0;
    plane = Create(10,40,10, Vector<4,float>(0.5,0.5,0.0,0.2));
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

    // Surface
    vbom->AllocBuffer(SURFACE_VERTICES, solid->surface->numFaces, GL_TRIANGLES);
    vbom->AllocBuffer(SURFACE_NORMALS,  solid->surface->numFaces, GL_TRIANGLES);
    // Center of mass points
    vbom->AllocBuffer(CENTER_OF_MASS, solid->body->numTetrahedra, GL_POINTS);
    // Body mesh is all tetrahedron faces with colors and normals
    vbom->AllocBuffer(BODY_MESH, solid->body->numTetrahedra*4, GL_TRIANGLES);
    vbom->AllocBuffer(BODY_COLORS, solid->body->numTetrahedra*4, GL_TRIANGLES);
    vbom->AllocBuffer(BODY_NORMALS, solid->body->numTetrahedra*4, GL_TRIANGLES);
    vbom->AllocBuffer(EIGEN_VALUES, solid->body->numTetrahedra, GL_POINTS);

    // Stress tensors visualizes stress planes.
    vbom->AllocBuffer(STRESS_TENSORS, solid->body->numTetrahedra, ps);
    

    // Disabled to bypass rendering
    vbom->Disable(CENTER_OF_MASS);
    vbom->Disable(STRESS_TENSORS);
    
    printf("[VboManager] Total Bytes Allocated: %i\n", totalByteAlloc);

    // Buffer setup
    vbom->GetBuf(CENTER_OF_MASS).SetColor(0.0, 0.0, 1.0, 1.0);
}

void TLEDNode::Handle(Core::ProcessEventArg arg) {
    if (!solid->IsInitialized()) return;

	// Update all visualization data
    vbom->MapAllBufferObjects();   

    if( !paused )
        for (unsigned int i=0; i<numIterations; i++) {
            calculateGravityForces(solid);
            calculateInternalForces(solid, vbom);
            updateDisplacement(solid);
            applyFloorConstraint(solid, 0); 
        }

    plane->SetPosition(Vector<3,float>(minX,0.0,0.0));
 
    if( vbom->IsEnabled(SURFACE_VERTICES) )
        updateSurface(solid, vbom);

    if( vbom->IsEnabled(BODY_MESH) )
        updateBodyMesh(solid, vbom, minX);

    if( vbom->IsEnabled(CENTER_OF_MASS) )
        updateCenterOfMass(solid, vbom);
    
    if( vbom->IsEnabled(STRESS_TENSORS) )
        updateStressTensors(solid, vbom);

    vbom->UnmapAllBufferObjects();
    
    //    if( paused ) {
    //    vbom->dumpBufferToFile("./dump.txt", vbom->GetBuf(EIGEN_VALUES));
    //    vbom->dumpBufferToFile("./dump.txt", vbom->GetBuf(BODY_COLORS));
    //    exit(-1);
    // }
}

void TLEDNode::Handle(Core::DeinitializeEventArg arg) {
    if (solid != NULL)
        solid->DeAlloc();
    //cleanupDisplay();
}

void TLEDNode::Apply(Renderers::IRenderingView* view) {
    VisitSubNodes(*view);

    if (!solid->IsInitialized()) return;

    // These buffers will only be rendered if they are enabled.
    //vbom->Render(SURFACE_VERTICES);
    vbom->Render(CENTER_OF_MASS);
    vbom->Render(STRESS_TENSORS);

    vbom->Render(vbom->GetBuf(SURFACE_VERTICES),
                 vbom->GetBuf(BODY_COLORS),
                 vbom->GetBuf(SURFACE_NORMALS));
    
    vbom->Render(vbom->GetBuf(BODY_MESH), 
                 vbom->GetBuf(BODY_COLORS),
                 vbom->GetBuf(BODY_NORMALS));
    
    // needs to be last, because it is transparent
    if (renderPlane) 
        plane->Accept(*view);
}
