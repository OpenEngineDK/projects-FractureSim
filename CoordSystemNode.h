#ifndef _COORD_SYSTEM_NODE_
#define _COORD_SYSTEM_NODE_

#include <Scene/RenderNode.h>

using namespace OpenEngine;

class CoordSystemNode : public Scene::RenderNode {
 public:
    CoordSystemNode();
    void Apply(Renderers::IRenderingView* view);
};

#endif // _COORD_SYSTEM_NODE_
