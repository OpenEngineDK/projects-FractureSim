#include "TLEDNode.h"
#include "MshObjLoader.h"
#include "TetGenLoader.h"
#include "TypeConverter.h"
#include "Precompute.h"
#include "Physics_kernels.h"
#include "HelperFunctions.h"
#include "CudaMem.h"
#include "CrackStrategyOne.h"

#include <string>
#include <iostream>

#include <Math/Vector.h>
#include <Logging/Logger.h>

TLEDNode::TLEDNode() {
    solid = NULL;
    numIterations = 25;
    paused = true;
    renderPlane = false;
    useAlphaBlending = false;
    minX = -100.0;
    plane = Create(10,40,10, Vector<4,float>(0.5,0.5,0.0,0.2));
    dump = false;
    timer.Start();
    crackTrackAllWay = false;
    exception = false;
    crackTrackingItrCount = 0;
}

TLEDNode::~TLEDNode() {
    // Delete VBO manager
    delete vbom;
}

void TLEDNode::Handle(Core::InitializeEventArg arg) {
    INITIALIZE_CUDA();
    CHECK_FOR_CUDA_ERROR();
    logger.info << "CUDA info:" << PRINT_CUDA_DEVICE_INFO() << logger.end;
    logger.info << "TLEDNode initialization start" << logger.end;

    ISolidLoader* loader = NULL;
    std::string dataDir = "projects/TLED/data/RegistrationShapes/";

    
    //PROSTATE: vpool: 3386, body tetrahedra: 16068, surface triangles: 2470
    /*loader = new MshObjLoader(dataDir + "PROSTATE.msh",
                              dataDir + "PROSTATE.obj");
    */
    //tand2: vpool: 865, body tetrahedra: 3545, surface triangles: 946
    /*loader = new MshObjLoader(dataDir + "tand2.msh",
                              dataDir + "tand2.obj");
      
    */
    //tetrahedra: vpool: 4, body tetrahedra: 1, surface triangles: 4
    /*loader = new TetGenLoader
    (dataDir + "tetrahedron.ascii.1.node",
     dataDir + "tetrahedron.ascii.1.ele",
     dataDir + "tetrahedron.ascii.1.smesh");
    */

    /*    //box: vpool: 14, body tetrahedra: 17, surface triangles: 24
    loader = new TetGenLoader
    (dataDir + "box.ascii.1.node",
     dataDir + "box.ascii.1.ele",
     dataDir + "box.ascii.1.smesh");
    */
    //tetrahedra: vpool: 119, body tetrahedra: 328, surface triangles: 212
    /*    loader = new TetGenLoader
        (dataDir + "sphere.ascii.1.node", 
         dataDir + "sphere.ascii.1.ele", 
         dataDir + "sphere.ascii.1.smesh");
    */

    //Bar: vpool: 119, body tetrahedra: 328, surface triangles: 212
    loader = new TetGenLoader
        (dataDir + "bar.ascii.1.node", 
         dataDir + "bar.ascii.1.ele", 
         dataDir + "bar.ascii.1.smesh");
    

    /*
    //bunny: vpool: , body tetrahedra: , surface triangles: 
    loader = new TetGenLoader
    (dataDir + "bunny.ascii.1.node",
     dataDir + "bunny.ascii.1.ele",
     dataDir + "bunny.ascii.1.smesh");
    */

    loader->Load();

    logger.info << "number of vertices: "
                << loader->GetVertexPool().size() << logger.end;
    logger.info << "number of body tetrahedra: " 
                << loader->GetBody().size() << logger.end;
    logger.info << "number of surface triangles: " 
                << loader->GetSurface().size() << logger.end;

    CHECK_FOR_CUDA_ERROR();
    solid = new Solid();
	solid->state = new TetrahedralTLEDState(); 
    solid->vertexpool = TypeConverter
        ::ConvertToVertexPool(loader->GetVertexPool());
    solid->body = TypeConverter
        ::ConvertToBody(loader->GetBody());
    solid->surface = TypeConverter
        ::ConvertToSurface(loader->GetSurface());

    // scaling factors for the different models
    //solid->vertexpool->Scale(1.1); // blob
    //solid->vertexpool->Scale(5); // tand2, tetrahedra and box
    //solid->vertexpool->Scale(10);
    //solid->vertexpool->Scale(30); // bunny
    //solid->vertexpool->Scale(0.3); // sphere
    solid->vertexpool->Scale(10.0); // bar

    logger.info << "pre computing" << logger.end;
    moveAccordingToBoundingBox(solid);
    solid->vertexpool->Move(0,25,0);
    
    //precompute(solid, density, smallestAllowedVolume, smallestAllowedLength,
    //           mu, lambda, timeStepFactor, damping);
    precompute(solid, 0.001f, 0.0f, 0.0f, 1000007.0f, 0.005f, 0.4f, 50.0f); //stiff
    //precompute(solid, 24000.0f, 0.0f, 0.0f,
    //       75000000000.0f, 2045000000.0f, 0.4f, 100.0f); //concrete
    //precompute(solid, 2.4f, 0.0f, 0.0f,
    //       136360000000.0f, 8334000000.0f, 0.4f, 100.0f); //concrete moded
    //precompute(solid, 0.001f, 0.0f, 0.0f, 207.0f, 2500.0f, 0.3f, 10.0f); //yelly
	//precompute(solid, 0.001f, 0.0f, 0.0f, 80007.0f, 49329.0f, 0.5f, 50.0f); //soft
    //precompute(solid, 0.001f, 0.0f, 0.0f, 207.0f, 2500.0f, 0.2f, 150.0f); //yelly
    
    // Initialize crack strategy
    crackStrategy = new CrackStrategyOne();
    
    // Initializing tetrahedron neighbouring lists
    createNeighbourList(solid);
    
    logger.info << "TLEDNode initialization done" << logger.end;

    // Load polygon model for visualization
    PolyShape ps("FlightArrow7.obj", 0.5f);
    //PolyShape ps("Box12.obj");
    //PolyShape ps("Sphere80.obj");

    // Add node constraints
    AddConstraint(new PolyShape("Box12.obj"), &solidCollisionConstraint);


    // Initialize the Visualizer
    vbom = new VboManager();

    // Surface
    vbom->AllocBuffer(SURFACE_VERTICES, solid->surface->numFaces, GL_TRIANGLES);
    vbom->AllocBuffer(SURFACE_NORMALS,  solid->surface->numFaces, GL_TRIANGLES);

    // Center of mass points
    vbom->AllocBuffer(CENTER_OF_MASS, solid->body->numTetrahedra, GL_POINTS);
    
    // Body mesh is all tetrahedron faces with colors and normals
    vbom->AllocBuffer(BODY_MESH, solid->body->numTetrahedra*4, GL_TRIANGLES);
    vbom->AllocBuffer(BODY_COLORS, solid->body->numTetrahedra*12, GL_POINTS);
    vbom->AllocBuffer(BODY_NORMALS, solid->body->numTetrahedra*12, GL_POINTS);
    vbom->AllocBuffer(EIGEN_VECTORS, solid->body->numTetrahedra*3, GL_POINTS);
    vbom->AllocBuffer(EIGEN_VALUES, solid->body->numTetrahedra, GL_POINTS);

    // Stress tensors visualizes stress planes.
    //vbom->AllocBuffer(STRESS_TENSOR_COLORS, solid->body->numTetrahedra, GL_POINTS);
    vbom->AllocBuffer(STRESS_TENSOR_VERTICES, solid->body->numTetrahedra, ps);
    //vbom->AllocBuffer(STRESS_TENSOR_COLORS,   solid->body->numTetrahedra*ps.numVertices, GL_POINTS);
    vbom->AllocBuffer(STRESS_TENSOR_NORMALS,  solid->body->numTetrahedra*ps.numVertices, GL_POINTS);
    
    // Disabled to bypass rendering
    vbom->Disable(SURFACE_VERTICES);
    vbom->Disable(SURFACE_NORMALS);
    vbom->Disable(CENTER_OF_MASS);
    vbom->Disable(STRESS_TENSOR_VERTICES);
    
    // Buffer setup
    vbom->GetBuf(CENTER_OF_MASS).SetColor(0.0, 0.0, 1.0, 1.0);

    PrintAllocedMemory();
}

void TLEDNode::StepPhysics() {
	// Update all visualization data
    vbom->MapAllBufferObjects();   

    for (unsigned int i=0; i<numIterations; i++) {
        calculateGravityForces(solid);
        calculateInternalForces(solid, vbom);
        updateDisplacement(solid);
        applyFloorConstraint(solid, 0);
    }

    vbom->UnmapAllBufferObjects();
}

void TLEDNode::Handle(Core::ProcessEventArg arg) {
    if (!solid->IsInitialized()) return;

    // TEST
    //constraintStrategyPtr = &solidCollisionConstraint;

    plane->SetPosition(Vector<3,float>(minX,0.0,0.0));

    // skip physics if delta time is to low
    const unsigned int deltaTime = 50000;

	// Update all visualization data
    vbom->MapAllBufferObjects();   

    if( !paused &&
        timer.GetElapsedTime() > Utils::Time(deltaTime)) {
        for (unsigned int i=0; i<numIterations; i++) {
            calculateGravityForces(solid);
            calculateInternalForces(solid, vbom);
            updateDisplacement(solid);
            //            ApplyConstraints(solid);
            applyFloorConstraint(solid, 0);

        }
        timer.Reset();
    }
 
    // Crack Tracking
    try {
        if( crackStrategy->CrackInitialized(solid) && !exception ) {
            
            while(crackTrackAllWay && !crackStrategy->FragmentationDone()){
                crackStrategy->ApplyCrackTracking(solid);
                
                if( crackTrackingItrCount++ > 100 ) break;
            } 
            if( !crackTrackAllWay )
                paused = true;
        }
    }catch(Core::Exception ex) { 
        paused = true; 
        exception = true;
        logger.info << "EXCEPTION: " << ex.what() << logger.end;
    }
    

    /*
    float3 n1 = make_float3(0,1,0);
    for( float r=Math::PI/2.0; r<2*Math::PI+Math::PI/2.0; r+=0.2f ){
        float3 n2 = make_float3(cos(r),sin(r),0);
        logger.info << "Angle: " << acos(dot(n1,n2)) * (180.0f / Math::PI) << logger.end;
    }
    exit(0);
    */
    if( vbom->IsEnabled(SURFACE_VERTICES) )
        updateSurface(solid, vbom);
    
    if( vbom->IsEnabled(CENTER_OF_MASS) ||
        vbom->IsEnabled(STRESS_TENSOR_VERTICES) )
        updateCenterOfMass(solid, vbom);
    
    if( vbom->IsEnabled(BODY_MESH) )
        updateBodyMesh(solid, vbom, minX);

    if (renderPlane) 
        planeClipping(solid, vbom, minX);

    if( vbom->IsEnabled(STRESS_TENSOR_VERTICES) ) {
        updateStressTensors(solid, vbom);
        applyTransformation(vbom->GetBuf(STRESS_TENSOR_VERTICES),
                            vbom->GetBuf(STRESS_TENSOR_NORMALS));    
    }
       
    vbom->UnmapAllBufferObjects();
    
    // press x to dump
    if( dump ) {
        //float* data;
        vbom->CopyBufferDeviceToHost(vbom->GetBuf(EIGEN_VALUES), "./eigValues.dump"); 
        vbom->CopyBufferDeviceToHost(vbom->GetBuf(EIGEN_VECTORS), "./eigVectors.dump");   
        //vbom->CopyBufferDeviceToHost(vbom->GetBuf(STRESS_TENSORS), "./matrixBuffer.dump");   
       
        //vbom->dumpBufferToFile("./dump.txt", vbom->GetBuf(STRESS_TENSOR_VERTICES));
        dump = false;
    }
}

void TLEDNode::Handle(Core::DeinitializeEventArg arg) {
    DEINITIALIZE_CUDA();
    if (solid != NULL)
        solid->DeAlloc();
    //cleanupDisplay();
}

void TLEDNode::Apply(Renderers::IRenderingView* view) {
    VisitSubNodes(*view);

    if (!solid->IsInitialized()) return;

    // Visualize constraints
    VisualizeConstraints();

    // These buffers will only be rendered if they are enabled.
    vbom->Render(CENTER_OF_MASS);
    vbom->RenderWithNormals(vbom->GetBuf(STRESS_TENSOR_VERTICES),
                            vbom->GetBuf(STRESS_TENSOR_NORMALS)); 
    //vbom->Render(STRESS_TENSOR_VERTICES);
    //vbom->RenderWithColors(vbom->GetBuf(STRESS_TENSORS), vbom->GetBuf(STRESS_TENSOR_COLORS));

    vbom->RenderWithNormals(vbom->GetBuf(SURFACE_VERTICES),
                            vbom->GetBuf(SURFACE_NORMALS));
    
    vbom->Render(vbom->GetBuf(BODY_MESH), 
                 vbom->GetBuf(BODY_COLORS),
                 vbom->GetBuf(BODY_NORMALS), useAlphaBlending);
    
    // needs to be last, because it is transparent
    if (renderPlane) 
        plane->Accept(*view);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    crackStrategy->RenderDebugInfo(solid);
}

void TLEDNode::AddConstraint(PolyShape* constrainArea, constraintFuncPtr constrainFunction) {
    std::pair<PolyShape*, constraintFuncPtr> constraint(constrainArea, constrainFunction);
    nodeConstraint.push_back(constraint);
}

void TLEDNode::ApplyConstraints(Solid* solid) {
    // For each constrain
    std::list< std::pair<PolyShape*, constraintFuncPtr> >::iterator itr;
    for( itr=nodeConstraint.begin(); itr!=nodeConstraint.end(); itr++ ){
        PolyShape* area = (*itr).first;
        constraintFuncPtr cons = (*itr).second;
        // Apply constrain on solid
        (*cons)(solid, area);  
    }
}

void TLEDNode::VisualizeConstraints() {

}

