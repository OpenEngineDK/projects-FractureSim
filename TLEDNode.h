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
#include "Modifier.h"
#include "DisplacementModifier.h"
#include "MovableCollisionModifier.h"
#include "ForceModifier.h"
#include <string>

using namespace OpenEngine;


class TLEDNode : public Scene::RenderNode, public Core::IModule {
 public:
    CrackStrategy* crackStrategy;
    Scene::TransformationNode* plane;
    Solid* solid;
    Utils::Timer timer, sim_clock;
    VboManager* vbom;
    std::list< Modifier* > modifier;
    unsigned int numIterations;
    bool paused, dump, renderPlane, useAlphaBlending, crackTrackingEnabled;
    float minX;
    bool crackTrackAllWay;
    bool exception;
    int crackTrackingItrCount;
    float timestep;

    float4* displacement;
    
    DisplacementModifier* addDisp;
    MovableCollisionModifier* tool;
    ForceModifier* addForce;
       
    TLEDNode(Solid* solid);
    virtual ~TLEDNode();

    virtual void Apply(Renderers::RenderingEventArg arg, Scene::ISceneNodeVisitor& v);
    virtual void Handle(Core::InitializeEventArg arg);
    virtual void Handle(Core::ProcessEventArg arg);
    virtual void Handle(Core::DeinitializeEventArg arg);

    void StepPhysics();

    void ApplyModifiers(Solid* solid);
    void VisualizeModifiers();
};

#endif // _TLED_NODE_
