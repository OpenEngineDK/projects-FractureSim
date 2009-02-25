#ifndef _TLED_NODE_
#define _TLED_NODE_

#include <Core/IModule.h>
#include <Core/EngineEvents.h>
#include <Scene/RenderNode.h>
#include <Renderers/IRenderingView.h>

using namespace OpenEngine;

#include "TLEDSolver.h"

class TLEDNode : public Scene::RenderNode, public Core::IModule {
 public:
    TLEDNode();
    virtual ~TLEDNode();

    virtual void Apply(Renderers::IRenderingView* view);
    virtual void Handle(Core::InitializeEventArg arg);
    virtual void Handle(Core::ProcessEventArg arg);
    virtual void Handle(Core::DeinitializeEventArg arg);

 private:
    TetrahedralTLEDState* state;
    TetrahedralMesh* mesh;
    TriangleSurface* surface;
};

#endif // _TLED_NODE_
