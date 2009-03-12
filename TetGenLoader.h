#ifndef _TET_GEN_LOADER_
#define _TET_GEN_LOADER_

#include "ISolidLoader.h"

#include <Math/Vector.h>
#include <list>

using namespace OpenEngine;

class TetGenLoader : public ISolidLoader {
 public:
    TetGenLoader(std::string vertexfile, std::string bodyfile,
                 std::string surfacefile);
    ~TetGenLoader();
    void Load();
    std::list< Math::Vector<3,float> > GetVertexPool();
    std::list< Math::Vector<3,unsigned int> > GetSurface();
    std::list< Math::Vector<4,unsigned int> > GetBody();
 private:
    std::string vertexfile;
    std::string bodyfile;
    std::string surfacefile;

    std::list< Math::Vector<3,float> > vertexPool;
    std::list< Math::Vector<3,unsigned int> > surface;
    std::list< Math::Vector<4,unsigned int> > body;
};

#endif // _TET_GEN_LOADER_
