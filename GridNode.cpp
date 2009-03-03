#include "GridNode.h"

GridNode::GridNode(float numberOfLinesPerAxis,
                   float spaceBetweenLines, Math::Vector<3,float> color) {
    this->numberOfLinesPerAxis = numberOfLinesPerAxis;
    this->spaceBetweenLines = spaceBetweenLines;
    this->color = color;
}

void GridNode::Apply(Renderers::IRenderingView* view) {
    // draw xz plane as grid
    for (float i= -numberOfLinesPerAxis; i<numberOfLinesPerAxis; 
         i+=spaceBetweenLines) {
        //if (i == 0.0) continue;
        view->GetRenderer()->DrawLine( Geometry::Line(Math::Vector<3,float>(-numberOfLinesPerAxis,0.0,i),  Math::Vector<3,float>(numberOfLinesPerAxis,0.0,i) ), color);
        view->GetRenderer()->DrawLine( Geometry::Line(Math::Vector<3,float>(i, 0.0, -numberOfLinesPerAxis), Math::Vector<3,float>(i, 0.0, numberOfLinesPerAxis) ), color);
    }
}
