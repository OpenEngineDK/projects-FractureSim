#ifndef _TLED_NODE_
#define _TLED_NODE_

#include <Meta/OpenGL.h>

#include <Core/IModule.h>
#include <Core/EngineEvents.h>
#include <Scene/RenderNode.h>
#include <Renderers/IRenderingView.h>

#include "VboManager.h"
#include "TetrahedralMesh.h"
#include <string>

using namespace OpenEngine;

class TLEDNode : public Scene::RenderNode, public Core::IModule {
 public:
    TLEDNode(std::string meshFile, std::string surfaceFile);
    virtual ~TLEDNode();

    virtual void Apply(Renderers::IRenderingView* view);
    virtual void Handle(Core::InitializeEventArg arg);
    virtual void Handle(Core::ProcessEventArg arg);
    virtual void Handle(Core::DeinitializeEventArg arg);

 private:
    std::string meshFile;
    std::string surfaceFile;
    TetrahedralTLEDState* state;

    VboManager* vbom;
    VertexPool* vertexpool;
    Body* body;
    Surface* surface;
    Solid* solid;
};

#endif // _TLED_NODE_
