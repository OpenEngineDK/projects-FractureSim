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
#include <Scene/BlendingNode.h>
#include <Scene/RenderStateNode.h>
#include <Scene/PointLightNode.h>

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
    SimpleSetup* setup = new SimpleSetup("TLED", new Viewport(0,0,1280,1024));

    // Print usage info.
    logger.info << "========= Running OpenEngine Test Project =========";
    logger.info << logger.end;
   
    //Vector<3,float> position(-100,30,0);
    //Vector<3,float> lookat(0,30,0);

    Vector<3,float> position(10,3,30);
    Vector<3,float> lookat(0,3,0);

    // Move the camera
    Camera* camera = setup->GetCamera();
    camera->SetPosition(position);
    camera->LookAt(lookat);
    
    // Register movement handler to be able to move the camera
    MoveHandler* move_h = 
        new MoveHandler(*camera, setup->GetMouse());
    move_h->SetObjectMove(false);
    setup->GetKeyboard().KeyEvent().Attach(*move_h);
    setup->GetEngine().InitializeEvent().Attach(*move_h);
    setup->GetEngine().ProcessEvent().Attach(*move_h);
    setup->GetEngine().DeinitializeEvent().Attach(*move_h);

    // Get root node of scene
    ISceneNode* root = setup->GetScene();
    RenderStateNode* rsn = new RenderStateNode();
    rsn->DisableOption(RenderStateNode::WIREFRAME);
    root->AddNode(rsn);
    RenderStateHandler* rsh = new RenderStateHandler(*rsn);
    setup->GetKeyboard().KeyEvent().Attach(*rsh);

    PointLightNode* pln = new PointLightNode();
    TransformationNode* lightPos = new TransformationNode();
    lightPos->AddNode(pln);
    lightPos->SetPosition(Vector<3,float>(1000,1000,1000));
    root->AddNode(lightPos);

    Scene::BlendingNode* bn = new Scene::BlendingNode();
    rsn->AddNode(bn);
    Vector<3,float> color(0.0,0.0,0.0);
    bn->AddNode(new GridNode(1000,30,color));
    bn->AddNode(new CoordSystemNode());

    TLEDNode* tled = new TLEDNode();
    bn->AddNode(tled);
    setup->GetEngine().InitializeEvent().Attach(*tled);
    setup->GetEngine().ProcessEvent().Attach(*tled);
    setup->GetEngine().DeinitializeEvent().Attach(*tled);

    KeyHandler* kh = new KeyHandler(*camera, tled, setup->GetEngine());
    kh->SetEye(position);
    kh->SetPoint(lookat);
    setup->GetKeyboard().KeyEvent().Attach(*kh);

    setup->AddDataDirectory("projects/TLED/data/models/");

     // Start the engine.
    setup->GetEngine().Start();

    // delete the entire scene
    delete setup->GetScene();
    // Return when the engine stops.
    return EXIT_SUCCESS;
}


