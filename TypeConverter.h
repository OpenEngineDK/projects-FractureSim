#ifndef _TYPE_CONVERTER_
#define _TYPE_CONVERTER_

#include "Solid.h"
#include <Math/Vector.h>
#include <list>

using namespace OpenEngine;

class TypeConverter {
 public:
    static VertexPool* ConvertToVertexPool(std::list< Math::Vector<3,float> > vertexpool) {
        VertexPool* vp = new VertexPool(vertexpool.size());
        Point* points = vp->data;
        std::list< Math::Vector<3,float> >::iterator itr2 = vertexpool.begin();
        for (unsigned int i=0; itr2 != vertexpool.end(); itr2++, i++) {
            Math::Vector<3,float> vert = *itr2;
            points[i].x = vert[0];
            points[i].y = vert[1];
            points[i].z = vert[2];
        }
        return vp;
    }

    static Surface* ConvertToSurface(std::list< Math::Vector<3, unsigned int> > sid) {
        Surface* surface = new Surface(sid.size());
        Triangle* faces = surface->faces;
        std::list< Math::Vector<3,unsigned int> >::iterator itr = sid.begin();
        for (unsigned int i=0; itr != sid.end(); itr++, i++) {
            Math::Vector<3,unsigned int> ids = *itr;
            faces[i].x = ids[0];
            faces[i].y = ids[1];
            faces[i].z = ids[2];
        }
        return surface;
    }

    static Body* ConvertToBody(std::list< Math::Vector<4, unsigned int> > bid) {
        Body* body = new Body(bid.size());

        Tetrahedron* tets = body->tetrahedra;
        std::list< Math::Vector<4,unsigned int> >::iterator itr = bid.begin();
        for (unsigned int i=0; itr != bid.end(); itr++, i++) {
            Math::Vector<4,unsigned int> ids = *itr;
            tets[i].x = ids[0];
            tets[i].y = ids[1];
            tets[i].z = ids[2];
            tets[i].w = ids[3];
        }
        return body;
    }
};

#endif // _TYPE_CONVERTER_
