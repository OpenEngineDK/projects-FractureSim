#include "TetGenLoader.h"

#include <Core/Exceptions.h>
#include <Logging/Logger.h>

TetGenLoader::TetGenLoader(std::string vertexfile, std::string bodyfile,
                           std::string surfacefile) {
    this->vertexfile = vertexfile;
    this->bodyfile = bodyfile;
    this->surfacefile = surfacefile;
}

TetGenLoader::~TetGenLoader() {
}

void TetGenLoader::Load() {
	unsigned int temp;
    unsigned int index;
    {
    // Load body, surface and vertexpool from file
	FILE* vFile = fopen(vertexfile.c_str(),"r");
	if (!vFile)
        throw Core::Exception("vertex file not found: " + vertexfile);

	unsigned int numVertices;
	fscanf (vFile, "%i\n", &numVertices);
	fscanf (vFile, "%i\n", &temp);//dimension
	fscanf (vFile, "%i\n", &temp);
	fscanf (vFile, "%i\n", &temp);

    logger.info << "---- vertex pool ----"  << logger.end;
	for (unsigned int i=0; i<numVertices && !feof(vFile); i++) {
        Math::Vector<3,float> point;
		fscanf (vFile, "%i %f %f %f\n", 
                &index, &(point[0]), &(point[1]), &(point[2]));
        logger.info << point << logger.end;
        vertexPool.push_back(point);
	}
	fclose (vFile);
}
    {
	FILE* bFile = fopen(bodyfile.c_str(),"r");
	if (!bFile)
        throw Core::Exception("body index file not found: " + bodyfile);

	unsigned int numTetrahedra;
	fscanf (bFile, "%i\n", &numTetrahedra);
	fscanf (bFile, "%i\n", &temp);
	fscanf (bFile, "%i\n", &temp);

    logger.info << "---- body indices ----"  << logger.end;
	for (unsigned int i=0; i<numTetrahedra && !feof(bFile); i++) {
        Math::Vector<4,unsigned int> bid;
		fscanf (bFile, "%i %i %i %i %i %i\n", 
                &index, &(bid[0]), &(bid[1]), &(bid[2]), &(bid[3]), &temp);
        logger.info << bid << logger.end;
        body.push_back(bid);
	}
	fclose (bFile);
    }
    {
	FILE* sFile = fopen(surfacefile.c_str(),"r");
	if (!sFile)
        throw Core::Exception("surface index file not found: " + surfacefile);

	unsigned int numTriangles;
	fscanf (sFile, "%i\n", &numTriangles);
	fscanf (sFile, "%i\n", &temp);

    logger.info << "---- surface indices ----"  << logger.end;
    for (unsigned int i=0; i<numTriangles && !feof(sFile); i++) {
        Math::Vector<3, unsigned int> sid;
        fscanf (sFile, "%i %i %i %i", &temp, &(sid[0]), &(sid[1]), &(sid[2]));
        logger.info << sid << logger.end;
        surface.push_back(sid);
	}
	fclose (sFile);
    }
    logger.info << "---- -------------- ----"  << logger.end;
}

std::list< Math::Vector<3,float> > TetGenLoader::GetVertexPool() {
    return vertexPool;
}

std::list< Math::Vector<3,unsigned int> > TetGenLoader::GetSurface() {
    return surface;
}

std::list< Math::Vector<4,unsigned int> > TetGenLoader::GetBody() {
    return body;
}
