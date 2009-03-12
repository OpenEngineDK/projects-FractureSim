#ifndef _I_SOLID_LOADER_
#define _I_SOLID_LOADER_

#include <Math/Vector.h>
#include <list>

using namespace OpenEngine;

class ISolidLoader {
 public:
    virtual ~ISolidLoader() {}
    virtual void Load() = 0;
    virtual std::list< Math::Vector<3,float> > GetVertexPool() = 0;
    virtual std::list< Math::Vector<3,unsigned int> > GetSurface() = 0;
    virtual std::list< Math::Vector<4,unsigned int> > GetBody() = 0;
};

#endif // _I_SOLID_LOADER_
