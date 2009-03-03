#include "CoordSystemNode.h"

#include <Math/Vector.h>
#include <Geometry/Line.h>

CoordSystemNode::CoordSystemNode() {
}

void CoordSystemNode::Apply(Renderers::IRenderingView* view) {
    // draw coordinate system axis
    view->GetRenderer()->
        DrawLine( Geometry::Line(Math::Vector<3,float>(0.0),
                                 Math::Vector<3,float>(1000.0,0.0,0.0) ),
                  Math::Vector<3,float>(1.0,0.0,0.0) );
    view->GetRenderer()->
        DrawLine( Geometry::Line(Math::Vector<3,float>(0.0),
                                 Math::Vector<3,float>(0.0,1000.0,0.0) ),
                  Math::Vector<3,float>(0.0,1.0,0.0) );
    view->GetRenderer()->
        DrawLine( Geometry::Line(Math::Vector<3,float>(0.0),
                                 Math::Vector<3,float>(0.0,0.0,-1000.0) ),
                  Math::Vector<3,float>(0.0,0.0,1.0) );
}
