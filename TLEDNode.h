#ifndef _TLED_NODE_
#define _TLED_NODE_

#include <Meta/OpenGL.h>

#include <Core/IModule.h>
#include <Core/EngineEvents.h>
#include <Scene/RenderNode.h>
#include <Scene/TransformationNode.h>
#include <Renderers/IRenderingView.h>

#include "VboManager.h"
#include "Solid.h"
#include <string>

using namespace OpenEngine;

class TLEDNode : public Scene::RenderNode, public Core::IModule {
 public:
    VboManager* vbom;
    unsigned int numIterations;
    bool paused, renderPlane;
    float minX;

    TLEDNode();
    virtual ~TLEDNode();

    virtual void Apply(Renderers::IRenderingView* view);
    virtual void Handle(Core::InitializeEventArg arg);
    virtual void Handle(Core::ProcessEventArg arg);
    virtual void Handle(Core::DeinitializeEventArg arg);

    void TLEDNode::StepPhysics();
 private:
    Scene::TransformationNode* plane;
    Solid* solid;
};

#endif // _TLED_NODE_
