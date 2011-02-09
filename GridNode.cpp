#include "GridNode.h"

GridNode::GridNode(float numberOfLinesPerAxis,
                   float spaceBetweenLines, Math::Vector<3,float> color) {
    this->numberOfLinesPerAxis = numberOfLinesPerAxis;
    this->spaceBetweenLines = spaceBetweenLines;
    this->color = color;
}

void GridNode::Apply(Renderers::RenderingEventArg arg, Scene::ISceneNodeVisitor& v) {
    // draw xz plane as grid
    for (float i= -numberOfLinesPerAxis; i<numberOfLinesPerAxis; 
         i+=spaceBetweenLines) {
        if (i == 0.0) continue;
        arg.renderer.DrawLine( Geometry::Line(Math::Vector<3,float>(-numberOfLinesPerAxis,0.0,i),  Math::Vector<3,float>(numberOfLinesPerAxis,0.0,i) ), color);
        arg.renderer.DrawLine( Geometry::Line(Math::Vector<3,float>(i, 0.0, -numberOfLinesPerAxis), Math::Vector<3,float>(i, 0.0, numberOfLinesPerAxis) ), color);
    }
}
