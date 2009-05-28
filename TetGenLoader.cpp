#include "TetGenLoader.h"

#include <Core/Exceptions.h>
#include <Logging/Logger.h>
#include <Resources/File.h>
#include <Utils/Convert.h>
#include <iostream>

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
	unsigned int dim;
    unsigned int index;
    {
        // Load body, surface and vertexpool from file
        FILE* vFile = fopen(vertexfile.c_str(),"r");
        if (!vFile)
            throw Core::Exception("vertex file not found: " + vertexfile);
        
        unsigned int numVertices;
        int rslt = fscanf(vFile, "%i %i %i %i\n",
			  &numVertices, &dim, &temp, &temp);
	if (rslt != 4) throw Core::Exception("vertex pool header error");
        
        //logger.info << "---- vertex pool ----"  << logger.end;
        for (unsigned int i=0; i<numVertices && !feof(vFile); i++) {
            Math::Vector<3,float> point;
            rslt = fscanf(vFile, "%i %f %f %f\n", 
			   &index, &(point[0]), &(point[1]), &(point[2]));
	    if (rslt != 4) throw Core::Exception("not a vertex");
            //logger.info << point << logger.end;
	    vertexPool.push_back(point);
        }
        fclose (vFile);
        if (numVertices != vertexPool.size())
            throw Core::Exception("wrong number of vertex in vertexpool");
    }
    {
        FILE* bFile = fopen(bodyfile.c_str(),"r");
        if (!bFile)
            throw Core::Exception("body index file not found: " + bodyfile);
        
        unsigned int numTetrahedra;
        int rslt = fscanf(bFile, "%i %i %i\n", &numTetrahedra, &dim, &temp);
	if (rslt != 3) throw Core::Exception("body file header error");
        
        //logger.info << "---- body indices ----"  << logger.end;
        for (unsigned int i=0; i<numTetrahedra && !feof(bFile); i++) {
            Math::Vector<4,unsigned int> bid;
            rslt = fscanf(bFile, "%i %i %i %i %i %i\n", &index, 
			  &(bid[0]), &(bid[1]), &(bid[2]), &(bid[3]), &temp);
	    if (rslt != 6) throw Core::Exception("malformed tetra");
            //logger.info << bid << logger.end;
            body.push_back(bid);
        }
        fclose (bFile);
        if (numTetrahedra != body.size())
            throw Core::Exception("wrong number of body indices");
    }
    {
        std::ifstream* in = Resources::File::Open(surfacefile);
        char buffer[255];
        int line = 0;
        bool done = false;
        while (!done) {
            in->getline(buffer, 255);
            line++;
            if (sscanf(buffer, "%i %i %i %i", &temp, &dim, &temp, &temp) == 4)
                done = true;
            if (in->eof()) 
                throw Core::Exception("invalid surface file header on line: " +
                                      Utils::Convert::ToString(line));
        }

        unsigned int numTriangles;
        done = false;
        while (!done) {
            in->getline(buffer, 255);
            line++;
            if (sscanf (buffer, "%i %i\n", &numTriangles, &temp) == 2)
                done = true;
            if (in->eof()) 
                throw Core::Exception("invalid surface file header on line: " +
                                      Utils::Convert::ToString(line));
        }

        //logger.info << "---- surface indices ----"  << logger.end;
        for (unsigned int i=0; i<numTriangles && !in->eof(); i++) {
            Math::Vector<3, unsigned int> sid;
            in->getline(buffer, 255);
            line++;
            if (sscanf (buffer, "%i %i %i %i", 
                        &temp, &(sid[0]), &(sid[1]), &(sid[2])) != 4)
                throw Core::Exception("invalid line in surface file: line " +
                                      Utils::Convert::ToString(line));
            //logger.info << sid << logger.end;
            surface.push_back(sid);
	}
    in->close();
    delete in;
    if (numTriangles != surface.size())
        throw Core::Exception("wrong number of surface indices");
    }
    //logger.info << "---- -------------- ----"  << logger.end;
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
