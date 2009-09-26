#include "TLEDNode.h"
#include "Precompute.h"
#include "Physics_kernels.h"
#include "HelperFunctions.h"
#include "CudaMem.h"
#include "CrackStrategyOne.h"
#include "FixedModifier.h"
#include "ForceModifier.h"
#include "DisplacementModifier.h"
#include "SimpleCollisionModifier.h"
#include "MovableCollisionModifier.h"

#include "CudaMem.h"
#include <string>
#include <iostream>

#include <Math/Vector.h>
#include <Logging/Logger.h>

static const float POS_X = 0.0;
static const float POS_Y = 0.0;
static const float POS_Z = 0.0;

TLEDNode::TLEDNode(Solid* solid) {
    this->solid = solid;
    numIterations = 50;
    paused = true;
    renderPlane = false;
    useAlphaBlending = false;
    minX = -100.0;
    plane = Create(10,40,10, Vector<4,float>(0.5,0.5,0.0,0.2));
    dump = false;
    timer.Start();
    crackTrackAllWay = false;
    crackTrackingEnabled = true;
    exception = false;
    crackTrackingItrCount = 0;
    timestep = 0.0;
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

    logger.info << "pre computing" << logger.end;
    moveAccordingToBoundingBox(solid);
    //    solid->vertexpool->Move(0,1.5,0);
    //solid->vertexpool->Move(20,0,0);
    solid->vertexpool->Move(POS_X, POS_Y, POS_Z);
    //    solid->vertexpool->Scale(0.1, 0.1, 0.1);
    //    solid->vertexpool->Scale(0.8, 2.0, 2.0);

    // Debug
    displacement = (float4*)malloc(sizeof(float4)*solid->vertexpool->size);

    //precompute(solid, smallestAllowedVolume, smallestAllowedLength,
    //           timeStepFactor, damping);
	timestep = precompute(solid, 0.0f, 0.0f, 0.5f, 0.5f);
     
    // Initialize crack strategy
    crackStrategy = new CrackStrategyOne();
    
    // Initializing tetrahedron neighbouring lists
    createNeighbourList(solid);
    
    logger.info << "TLEDNode initialization done" << logger.end;

    // Load polygon model for visualization
    PolyShape ps("FlightArrow7.obj", 0.25, 0.25, 0.1);
    //printf("numVertrices: %i - Size:%i\n ", ps.numVertices, sizeof(float4));
    //PolyShape ps("Box12.obj");
    //PolyShape ps("Sphere80.obj");

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
    vbom->AllocBuffer(STRESS_TENSOR_VERTICES, solid->body->numTetrahedra, ps);
    vbom->AllocBuffer(STRESS_TENSOR_NORMALS,  solid->body->numTetrahedra*ps.numVertices, GL_POINTS);
    // Disabled to bypass rendering
    vbom->Disable(SURFACE_VERTICES);
    vbom->Disable(SURFACE_NORMALS);
    vbom->Disable(CENTER_OF_MASS);
    vbom->Disable(STRESS_TENSOR_VERTICES);
    // Buffer setup
    vbom->GetBuf(CENTER_OF_MASS).SetColor(0.0, 0.0, 1.0, 1.0);




    // ----- Basic beam bending setup ------ //
    /*    
    float scale = 3;
    float3 force = make_float3(0, -(float)(1.0 * pow(10,7)), 0);
    ForceModifier* addForce = new ForceModifier(solid, new PolyShape("Box12.obj", scale, scale*2, scale*2), force);
    //addForce->Move(89, POS_Y, POS_Z);
    addForce->Move(8.9, POS_Y, POS_Z);
    addForce->SetColorBufferForSelection(&vbom->GetBuf(BODY_COLORS));
    modifier.push_back(addForce);
    
    FixedModifier* fixedBox2 = new FixedModifier(new PolyShape("Box12.obj", scale, scale*2, scale*2));
    //fixedBox2->Move(-70,POS_Y,POS_Z);
    fixedBox2->Move(-5.0,POS_Y,POS_Z);
    modifier.push_back(fixedBox2);
    */  

    // ---  Stress-strain curve setup ---- //
    float scale = 30;
    float3 force = make_float3((float)(1.0 * pow(10,6))/9.0f, 0, 0);
    //float3 force = make_float3(0, 0, 0);
    addForce = new ForceModifier(solid, new PolyShape("Box12.obj", scale, scale*2, scale*2), force);
    addForce->Move(90, POS_Y, POS_Z);
    addForce->SetColorBufferForSelection(&vbom->GetBuf(BODY_COLORS));
    modifier.push_back(addForce);
    
    /*             
    float3 disp = make_float3(0, 0, 0);
    addDisp = new DisplacementModifier(solid, new PolyShape("Box12.obj", scale,scale*2,scale*2), disp);
    addDisp->Move(90, POS_Y, POS_Z);
    //addDisp->Move(5, POS_Y, POS_Z);
    addDisp->SetColorBufferForSelection(&vbom->GetBuf(BODY_COLORS));
    modifier.push_back(addDisp);
    */
    FixedModifier* fixedBox2 = new FixedModifier(new PolyShape("Box12.obj", scale, scale*2, scale*2));
    fixedBox2->Move(-90,POS_Y,POS_Z);
    //fixedBox2->Move(-5,POS_Y,POS_Z);
    modifier.push_back(fixedBox2);
    

    // ----- Mesh independence setup ----- //
    /*
    float scale = 25;
    MovableCollisionModifier* centerBox = new MovableCollisionModifier(new PolyShape("Box12.obj", scale/5,scale,scale*2));
    centerBox->Move(0,45,0);
    modifier.push_back(centerBox);

    MovableCollisionModifier* leftBox = new MovableCollisionModifier(new PolyShape("Box12.obj", scale,scale,scale*2));
    leftBox->Move(-70.0, 0, 0);
    modifier.push_back(leftBox);

    MovableCollisionModifier* rightBox = new MovableCollisionModifier(new PolyShape("Box12.obj", scale,scale,scale*2));
    rightBox->Move(70.0, 0, 0);
    modifier.push_back(rightBox);
    */


    // ----- Crack beam in half setup ---- //
    /*    
    float scale = 3;
    
    MovableCollisionModifier* centerBox = new MovableCollisionModifier(new PolyShape("Box12.obj", scale/4,scale,scale));
    centerBox->Move(0,5,0);
    modifier.push_back(centerBox);
    
    SimpleCollisionModifier* leftBox = new SimpleCollisionModifier(new PolyShape("Box12.obj", scale,scale,scale));
    leftBox->Move(-5.0, 0, 0);
    modifier.push_back(leftBox);

    SimpleCollisionModifier* rightBox = new SimpleCollisionModifier(new PolyShape("Box12.obj", scale,scale,scale));
    rightBox->Move(5.0, 0, 0);
    modifier.push_back(rightBox);
    */


    // ----- Crack supported beam setup ----- //
    /*
    float scale = 5;
    float3 force = make_float3(0, -(float)(5.125 * pow(10,8)), 0);
    ForceModifier* addForce = new ForceModifier(solid, new PolyShape("Box12.obj", scale, scale, scale), force);
    addForce->Move(5, POS_Y, POS_Z);
    addForce->SetColorBufferForSelection(&vbom->GetBuf(BODY_COLORS));
    modifier.push_back(addForce);
    
    FixedModifier* fixedBox2 = new FixedModifier(new PolyShape("Box12.obj", scale, scale, scale));
    fixedBox2->Move(-5,POS_Y,POS_Z);
    modifier.push_back(fixedBox2);
    */


    // ------ Crack tooth dummy by force ------ //
    /*    float scale = 15;
        
    MovableCollisionModifier* box = new MovableCollisionModifier(new PolyShape("Box12.obj", scale,scale/5,scale));
    box->Move(-13,30,0);
    modifier.push_back(box);
    
    FixedModifier* fixedBox2 = new FixedModifier(new PolyShape("Box12.obj", scale, scale, scale));
    fixedBox2->Move(0, 0, POS_Z);
    modifier.push_back(fixedBox2);
    */
    /*  FixedModifier* fixedBox3 = new FixedModifier(new PolyShape("Box12.obj", scale, scale, scale));
    fixedBox3->Move(0, 20, POS_Z);
    modifier.push_back(fixedBox3);
    */
    // ------ Crack fake tooth with elevator tool ---------- //
    /*float scale = 15;
    PolyShape* toolShape = new PolyShape("Box12.obj", 20.0, 0.8, 20.0);
    tool = new MovableCollisionModifier(toolShape);
    tool->Move(14, 17, 0);
    modifier.push_back(tool);

    FixedModifier* fixedBox2 = new FixedModifier(new PolyShape("Box12.obj", scale, scale, scale));
    fixedBox2->Move(0, POS_Y, POS_Z);
    modifier.push_back(fixedBox2);
    */

    // ------ Crack real tooth model with elevator tool ---------- //
    /*
    float scale = 35;
    PolyShape* toolShape = new PolyShape("Box12.obj", 20.0, 0.8, 20.0);
    tool = new MovableCollisionModifier(toolShape);
    tool->Move(0, 47, -25);
    modifier.push_back(tool);
    
    FixedModifier* fixedBox2 = new FixedModifier(new PolyShape("Box12.obj", scale, scale+25, scale));
    fixedBox2->Move(0, 10, POS_Z);
    modifier.push_back(fixedBox2);
    */

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

    plane->SetPosition(Vector<3,float>(minX,0.0,0.0));

    // skip physics if delta time is to low
    const unsigned int deltaTime = 50000 * 0;

	// Update all visualization data
    vbom->MapAllBufferObjects();   

    static int numItr = 0;
    static Utils::Time oldtime;
    if( !paused &&
        timer.GetElapsedTime() > Utils::Time(deltaTime)) {
        sim_clock.Start();

        // Use this for stress/strain test to increase stretch
        //        addDisp->addDisplacement += make_float3(0.0001f, 0, 0);

        //        float3 force = make_float3((float)(3.28 * pow(10,10))/33.0f, 0, 0);
        /*        float maxForce = (float)((1.0*pow(10,20)));
        if(addForce->addForce.x*28.0f < maxForce){
            float3 newForce = addForce->addForce + (make_float3(1.0*pow(10,9), 0.0f, 0.0f)/ 28.0f);
            addForce->addForce = newForce;
        }else
            addForce->addForce = make_float3(maxForce/28.0f, 0, 0);
        */

        for (unsigned int i=0; i<numIterations; i++) {
            // calculateGravityForces(solid);
            ApplyModifiers(solid);
            //printf("");
            calculateInternalForces(solid, vbom);
            updateDisplacement(solid);
            //applyFloorConstraint(solid, 0);

     
            //            if( numItr++ > 100 )
            //  exit(0);
            /*            
            static int iterations = 0;
            iterations++;
            if ((iterations % 1000) == 0) {
                Utils::Time time = sim_clock.GetElapsedTime();
                try {
                    logger.info << "iterations: " << iterations
                                << " sim-time: " << Utils
                        ::Time((unsigned long)(iterations * timestep * 1000000))
                                << " time: " << time
                                << " delta: " << time - oldtime
                                << logger.end;
                } catch(Core::Exception) {} //happens on reset
                oldtime = time;
                }*/
            //if (iterations == 12982)
            //  paused = true;
        }
        timer.Reset();
    }
    else
        sim_clock.Stop();

    // Debug
    
    cudaMemcpy(displacement, solid->vertexpool->Ui_t, sizeof(float4)*solid->vertexpool->size, cudaMemcpyDeviceToHost);
    float maxDisp = 0;
    //float minX = 0;
    float maxX = 0;
    for( unsigned int i=0; i<solid->vertexpool->size; i++ ) {
        if( length(displacement[i]) > maxDisp ) 
            maxDisp = length(displacement[i]);
       if( displacement[i].x > maxX ) 
           maxX = displacement[i].x;
    }
    // Print engineering strain and stress 
    //printf("%E \n", addForce->addForce.x/28.0f);
    //    printf("%E \t %E\n", ((160.0f+maxDisp)/160.0f)-1.0, addForce->addForce.x*28.0f);
    //    printf("%E \t %E \t %E\n", ((160.0f+maxDisp)/160.0f)-1.0, addForce->addForce.x*20.0f, maxDisp);
    //    static int p = 0;
    //if( p++ % 10 == 0 )
    logger.info << "MaxDisp(Axial/Total): (" << maxX << "/" << maxDisp << ")" << logger.end;
    
    // Crack Tracking
    if( crackTrackingEnabled ){
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
    }

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

    // The modifiers needs to be applied again here for the selection tool
    ApplyModifiers(solid);

    vbom->UnmapAllBufferObjects();
    
    // press x to dump
    if( dump ) {
        //float* data;
        vbom->CopyBufferDeviceToHost(vbom->GetBuf(EIGEN_VALUES), "./eigValues.dump"); 
        vbom->CopyBufferDeviceToHost(vbom->GetBuf(EIGEN_VECTORS), "./eigVectors.dump");   
        //vbom->CopyBufferDeviceToHost(vbom->GetBuf(STRESS_TENSORS), "./matrixBuffer.dump");   
       
        //vbom->dumpBufferToFile("./com.txt", vbom->GetBuf(CENTER_OF_MASS));
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

    
    // These buffers will only be rendered if they are enabled.
    vbom->Render(CENTER_OF_MASS);

    vbom->RenderWithNormals(vbom->GetBuf(STRESS_TENSOR_VERTICES),
                            vbom->GetBuf(STRESS_TENSOR_NORMALS)); 

    vbom->RenderWithNormals(vbom->GetBuf(SURFACE_VERTICES),
                            vbom->GetBuf(SURFACE_NORMALS));
    
    vbom->Render(vbom->GetBuf(BODY_MESH), 
                 vbom->GetBuf(BODY_COLORS),
                 vbom->GetBuf(BODY_NORMALS), useAlphaBlending);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    crackStrategy->RenderDebugInfo(solid);
    
    // Visualize constraints
    VisualizeModifiers();

     // needs to be last, because it is transparent
    if (renderPlane) 
        plane->Accept(*view);
    
    //    tool->VisualizeNormals();
    
}


void TLEDNode::ApplyModifiers(Solid* solid) {
    std::list<Modifier*>::iterator itr;
    for( itr=modifier.begin(); itr!=modifier.end(); itr++ ){
        (*itr)->Apply(solid);
    }
}

void TLEDNode::VisualizeModifiers() {
    std::list<Modifier*>::iterator itr;
    for( itr=modifier.begin(); itr!=modifier.end(); itr++ ){
        (*itr)->Visualize();
    }
}

