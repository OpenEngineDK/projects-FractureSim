#ifndef _TLED_NODE_
#define _TLED_NODE_

#include <Core/IModule.h>
#include <Core/EngineEvents.h>
#include <Scene/RenderNode.h>
#include <Renderers/IRenderingView.h>
#include "Visualizer.h"
#include <string>

using namespace OpenEngine;

#include "TLEDSolver.h"

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
    TetrahedralMesh* mesh;
    TriangleSurface* surface;

    // Test
    Visualizer* visualizer;
};

#endif // _TLED_NODE_
