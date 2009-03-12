#ifndef _MSH_OBJ_LOADER_
#define _MSH_OBJ_LOADER_

#include "ISolidLoader.h"

#include <Math/Vector.h>
#include <list>

using namespace OpenEngine;

class MshObjLoader : public ISolidLoader {
 public:
    MshObjLoader(std::string mshfile, std::string objfile);
    ~MshObjLoader();
    void Load();
    std::list< Math::Vector<3,float> > GetVertexPool();
    std::list< Math::Vector<3,unsigned int> > GetSurface();
    std::list< Math::Vector<4,unsigned int> > GetBody();
 private:
    std::string mshfile;
    std::string objfile;

    std::list< Math::Vector<3,float> > vertexPool;
    std::list< Math::Vector<3,unsigned int> > surface;
    std::list< Math::Vector<4,unsigned int> > body;
};

#endif // _MSH_OBJ_LOADER_
