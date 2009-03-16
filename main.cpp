// main
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

// OpenEngine stuff
#include <Meta/Config.h>
#include <Logging/Logger.h>
#include <Logging/StreamLogger.h>
#include <Core/Engine.h>

// Math
#include <Math/Tensor.h>

#include <Display/Camera.h>
#include <Scene/RenderStateNode.h>

// SimpleSetup
#include <Utils/SimpleSetup.h>
#include <Utils/MoveHandler.h>
#include <Utils/RenderStateHandler.h>

#include "TLEDNode.h"
#include "GridNode.h"
#include "CoordSystemNode.h"
#include "KeyHandler.h"

// name spaces that we will be using.
// this combined with the above imports is almost the same as
// fx. import OpenEngine.Logging.*; in Java.
using namespace OpenEngine::Logging;
using namespace OpenEngine::Core;
using namespace OpenEngine::Display;
using namespace OpenEngine::Utils;
using namespace OpenEngine::Math;
using namespace OpenEngine::Resources;

/**
 * Main method for the first quarter project of CGD.
 * Corresponds to the
 *   public static void main(String args[])
 * method in Java.
 */
int main(int argc, char** argv) {
 
    // Create simple setup
    SimpleSetup* setup = new SimpleSetup("TLED", new Viewport(0,0,800,600));

   // Print usage info.
    logger.info << "========= Running OpenEngine Test Project =========";
    logger.info << logger.end;
   
    // Move the camera
    Camera* camera = setup->GetCamera();
    camera->SetPosition(Vector<3,float>(10,0,10));
    camera->LookAt(Vector<3,float>(0,0,0));

    // Register movement handler to be able to move the camera
    MoveHandler* move_h = 
        new MoveHandler(*camera, setup->GetMouse());
    setup->GetKeyboard().KeyEvent().Attach(*move_h);
    setup->GetEngine().InitializeEvent().Attach(*move_h);
    setup->GetEngine().ProcessEvent().Attach(*move_h);
    setup->GetEngine().DeinitializeEvent().Attach(*move_h);

    // Get root node of scene
    ISceneNode* root = setup->GetScene();
    RenderStateNode* rsn = new RenderStateNode();
    rsn->EnableOption(RenderStateNode::WIREFRAME);
    root->AddNode(rsn);
    RenderStateHandler* rsh = new RenderStateHandler(*rsn);
    setup->GetKeyboard().KeyEvent().Attach(*rsh);

    KeyHandler* kh = new KeyHandler(*camera);
    kh->SetEye(Vector<3,float>(0.0,10.0,-10.0f));
    kh->SetPoint(Vector<3,float>(0.0,10.0,0.0));
    setup->GetKeyboard().KeyEvent().Attach(*kh);

    std::string dataDir = "projects/TLED/data/RegistrationShapes/";
    //std::string meshFile = dataDir + "tand2.msh";
    //std::string surfaceFile = dataDir + "tand2.obj";
    std::string meshFile = dataDir + "PROSTATE.msh";
    std::string surfaceFile = dataDir + "PROSTATE.obj";
    TLEDNode* tled = new TLEDNode(meshFile, surfaceFile);
    rsn->AddNode(tled);

    Vector<3,float> color(0.0,0.0,0.0);
    root->AddNode(new GridNode(1000,20,color));
    root->AddNode(new CoordSystemNode());

    setup->GetEngine().InitializeEvent().Attach(*tled);
    setup->GetEngine().ProcessEvent().Attach(*tled);
    setup->GetEngine().DeinitializeEvent().Attach(*tled);

    setup->AddDataDirectory("resources/");
    
    // Start the engine.
    setup->GetEngine().Start();

    // delete the entire scene
    delete setup->GetScene();
    // Return when the engine stops.
    return EXIT_SUCCESS;
}


