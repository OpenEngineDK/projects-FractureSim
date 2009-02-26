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

#include <Display/Camera.h>
#include <Scene/RenderStateNode.h>

// SimpleSetup
#include <Utils/SimpleSetup.h>
#include <Utils/MoveHandler.h>

#include "TLEDNode.h"

// name spaces that we will be using.
// this combined with the above imports is almost the same as
// fx. import OpenEngine.Logging.*; in Java.
using namespace OpenEngine::Logging;
using namespace OpenEngine::Core;
using namespace OpenEngine::Display;
using namespace OpenEngine::Utils;

/**
 * Main method for the first quarter project of CGD.
 * Corresponds to the
 *   public static void main(String args[])
 * method in Java.
 */
int main(int argc, char** argv) {
    // Print usage info.
    logger.info << "========= Running OpenEngine Test Project =========";
    logger.info << logger.end;

    // Create simple setup
    SimpleSetup* setup = new SimpleSetup("TLED");

      // Move the camera
    Camera* camera = setup->GetCamera();
    camera->SetPosition(Vector<3,float>(50,0,50));
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
    TLEDNode* tled = new TLEDNode();
    rsn->AddNode(tled);

    setup->GetEngine().InitializeEvent().Attach(*tled);
    setup->GetEngine().ProcessEvent().Attach(*tled);
    setup->GetEngine().DeinitializeEvent().Attach(*tled);

    // Start the engine.
    setup->GetEngine().Start();

    // Return when the engine stops.
    return EXIT_SUCCESS;
}


