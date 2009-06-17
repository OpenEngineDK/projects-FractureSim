#ifndef _TLED_NODE_
#define _TLED_NODE_

#include <Meta/OpenGL.h>

#include <Core/IModule.h>
#include <Core/EngineEvents.h>
#include <Scene/RenderNode.h>
#include <Scene/TransformationNode.h>
#include <Renderers/IRenderingView.h>
#include <Utils/Timer.h>

#include "VboManager.h"
#include "CrackStrategy.h"
#include "Solid.h"
#include <string>

using namespace OpenEngine;


typedef void (*constraintFuncPtr)(Solid* solid, PolyShape* collidableObject);

class TLEDNode : public Scene::RenderNode, public Core::IModule {
 public:
    CrackStrategy* crackStrategy;
    Scene::TransformationNode* plane;
    Solid* solid;
    Utils::Timer timer;
    VboManager* vbom;
    std::list< std::pair<PolyShape*, constraintFuncPtr> > nodeConstraint;
    unsigned int numIterations;
    bool paused, dump, renderPlane, useAlphaBlending;
    float minX;
    bool crackTrackAllWay;
    bool exception;
    int crackTrackingItrCount;
    
    TLEDNode();
    virtual ~TLEDNode();

    virtual void Apply(Renderers::IRenderingView* view);
    virtual void Handle(Core::InitializeEventArg arg);
    virtual void Handle(Core::ProcessEventArg arg);
    virtual void Handle(Core::DeinitializeEventArg arg);

    void StepPhysics();

    void AddConstraint(PolyShape* constrainArea, constraintFuncPtr constrainFunction);
    void ApplyConstraints(Solid* solid);
    void VisualizeConstraints();

};

#endif // _TLED_NODE_
