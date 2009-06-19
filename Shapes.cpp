
#include "Shapes.h"
#include <stdio.h>
#include <vector>

#include <Resources/IModelResource.h>
#include <Resources/ResourceManager.h>
#include <Scene/GeometryNode.h>
#include <Geometry/Face.h>

#include <cufft.h>
#include <cutil.h>
#include <driver_types.h> // includes cudaError_t
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h> // includes cudaMalloc and cudaMemset

using namespace OpenEngine::Geometry;
using namespace OpenEngine::Scene;
using namespace OpenEngine::Resources;

PolyShape::PolyShape(std::string name, float scale) {
    // Load the model
    IModelResourcePtr mod_res = ResourceManager<IModelResource>::Create(name);
    mod_res->Load();
    if (mod_res->GetSceneNode() == NULL) 
        logger.info << "Error loading model" << logger.end;
    
    GeometryNode* geo_node = (GeometryNode*)mod_res->GetSceneNode();
    mod_res->Unload();
    FaceSet* face_set = geo_node->GetFaceSet();

    vector<float4> vList;  //Vertices 
    vector<float4> nList;  //Normals
    FaceList::iterator itr;
    for( itr = face_set->begin(); itr!=face_set->end(); itr++ ) {
        FacePtr face = *itr;
        for( int i=0; i<3; i++){
            float4 v = { face->vert[i][0], 
                         face->vert[i][1],
                         face->vert[i][2],
                         1.0};

            float4 n = { face->norm[i][0], 
                         face->norm[i][1],
                         face->norm[i][2],
                         1.0};

            // Scale model
            v *= scale;
            v.w = 1.0f;

             vList.push_back(v);
             nList.push_back(n);
        }
    }

    // Copy vertices into static buffer
    vertices = new float4[vList.size()];
    normals  = new float4[nList.size()];

    numVertices=0;
    std::vector<float4>::iterator it;
    for( it=vList.begin(); it!=vList.end(); it++ ) {
        //printf("[%i] %f %f %f\n", numVertices,(*it).x, (*it).y, (*it).z);
        vertices[numVertices++] = *it;
    }

    numNormals=0;
    for( it=nList.begin(); it!=nList.end(); it++ ) {
        //printf("[%i] %f %f %f\n", numVertices,(*it).x, (*it).y, (*it).z);
        normals[numNormals++] = *it;
    }


    printf("[PolyShape] numVertices loaded: %i - numNormals: %i\n", numVertices, numNormals);
}


