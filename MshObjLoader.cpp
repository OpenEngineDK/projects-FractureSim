#include "MshObjLoader.h"

#include <Core/Exceptions.h>
#include <stdio.h>

MshObjLoader::MshObjLoader(std::string mshfile, std::string objfile) {
    this->mshfile = mshfile;
    this->objfile = objfile;
}

MshObjLoader::~MshObjLoader() {
}

void MshObjLoader::Load() {
    // Load body and vertexpool from msh file
	FILE* bFile = fopen(mshfile.c_str(),"r");
	if (!bFile)
        throw Core::Exception("mesh file not found: " + mshfile);

	unsigned int numVertices;	
	unsigned int numTetrahedra;
	fscanf (bFile, "%i\n", &numVertices);
	fscanf (bFile, "%i\n", &numTetrahedra);

	for (unsigned int i=0; i<numVertices && !feof(bFile); i++) {
        Math::Vector<3,float> point;
		fscanf (bFile, "%f %f %f\n", 
                &(point[0]), &(point[1]), &(point[2]));
        vertexPool.push_back(point);
	}
	for (unsigned int i=0; i<numTetrahedra && !feof(bFile); i++) {
        Math::Vector<4,unsigned int> bid;
		fscanf (bFile, "%i %i %i %i\n", 
                &(bid[0]), &(bid[1]), &(bid[2]), &(bid[3]));
        body.push_back(bid);
	}
	fclose (bFile);

    // load surface from obj file
	FILE* sFile = fopen(objfile.c_str(),"r");
	if (!sFile)
        throw Core::Exception("obj file not found: " + objfile);

	unsigned int numTriangles = 0;
	while (!feof(sFile)) {
        unsigned char c;
		fscanf (sFile, "%c", &c);

		float tmp;
		switch (c) {
		case 'v': case 'V':
			fscanf (sFile, "%f %f %f\n", &(tmp), &(tmp), &(tmp)); //check vertexpool !?!?!
			break;	
		case 'f': case 'F': {
            Math::Vector<3, unsigned int> sid;
			fscanf (sFile, " %i %i %i", &(sid[0]), &(sid[1]), &(sid[2]));
            surface.push_back(sid - 1); //obj indexes from 1
			numTriangles++; }
            break;
		default:
            //printf("Unknown tag '%c' found in OBJ file\n", c);
            break;
		}
	}
	fclose (sFile);
}

std::list< Math::Vector<3,float> > MshObjLoader::GetVertexPool() {
    return vertexPool;
}

std::list< Math::Vector<3,unsigned int> > MshObjLoader::GetSurface() {
    return surface;
}

std::list< Math::Vector<4,unsigned int> > MshObjLoader::GetBody() {
    return body;
}
