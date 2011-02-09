#include "CoordSystemNode.h"

#include <Math/Vector.h>
#include <Math/Matrix.h>
#include <Geometry/Line.h>
#include <Meta/OpenGL.h>
#include <Logging/Logger.h>

using namespace OpenEngine::Math;

CoordSystemNode::CoordSystemNode() {
}

void CoordSystemNode::Apply(Renderers::RenderingEventArg arg, Scene::ISceneNodeVisitor& v) {
    // draw coordinate system axis
    arg.renderer.
        DrawLine( Geometry::Line(Math::Vector<3,float>(0.0),
                                 Math::Vector<3,float>(1000.0,0.0,0.0) ),
                  Math::Vector<3,float>(1.0,0.0,0.0) );
    arg.renderer.
        DrawLine( Geometry::Line(Math::Vector<3,float>(0.0),
                                 Math::Vector<3,float>(0.0,1000.0,0.0) ),
                  Math::Vector<3,float>(0.0,1.0,0.0) );
    arg.renderer.
        DrawLine( Geometry::Line(Math::Vector<3,float>(0.0),
                                 Math::Vector<3,float>(0.0,0.0,1000.0) ),
                  Math::Vector<3,float>(0.0,0.0,1.0) );

    
    /*    Vector<3,float> v(1,1,0);
    Vector<3,float> c(0,0,0);
    
    Vector<3,float> color(0.5, 0.5, 0);
    
    view->GetRenderer()->DrawLine( Geometry::Line(c,v), color);
    */
}

