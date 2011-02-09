#ifndef _GRID_NODE_
#define _GRID_NODE_

#include <Math/Vector.h>
#include <Scene/RenderNode.h>

using namespace OpenEngine;

class GridNode : public Scene::RenderNode {
 public:
    GridNode(float numberOfLinesPerAxis,
             float spaceBetweenLines, Math::Vector<3,float> color);
    ~GridNode() {}
    void Apply(Renderers::RenderingEventArg arg, Scene::ISceneNodeVisitor& v);
 private:
    float numberOfLinesPerAxis;
    float spaceBetweenLines;
    Math::Vector<3,float> color;
};

#endif // _GRID_NODE_
