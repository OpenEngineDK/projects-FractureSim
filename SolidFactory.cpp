#include "SolidFactory.h"

#include <Core/Exceptions.h>
#include <Logging/Logger.h>

#include "MshObjLoader.h"
#include "TetGenLoader.h"
#include "TypeConverter.h"

using namespace OpenEngine;

Solid* SolidFactory::Create(std::string name) {
    logger.info << "Loading solid: " << name << logger.end;

    Solid* solid = new Solid();

    ISolidLoader* loader = NULL;
    std::string dataDir = "projects/TLED/data/RegistrationShapes/";

    float scale = 1.0;
    if (name == "prostate") {
        //PROSTATE: vpool: 3386, body tetrahedra: 16068, surface triangles: 2470
        loader = new MshObjLoader(dataDir + "PROSTATE.msh",
                                  dataDir + "PROSTATE.obj");
        scale = 1.1;
    } else if (name == "tand2") {
        //tand2: vpool: 865, body tetrahedra: 3545, surface triangles: 946
        loader = new MshObjLoader(dataDir + "tand2.msh",
                                  dataDir + "tand2.obj");
        scale = 5;
    } else if (name == "tetrahedra") {
        //tetrahedra: vpool: 4, body tetrahedra: 1, surface triangles: 4
        loader = new TetGenLoader
            (dataDir + "tetrahedron.ascii.1.node",
             dataDir + "tetrahedron.ascii.1.ele",
             dataDir + "tetrahedron.ascii.1.smesh");
    } else if (name == "box") {
        //box: vpool: 14, body tetrahedra: 17, surface triangles: 24
        loader = new TetGenLoader
            (dataDir + "box.ascii.1.node",
             dataDir + "box.ascii.1.ele",
             dataDir + "box.ascii.1.smesh");
        scale = 10;
    } else if (name == "sphere") {
        //sphere: vpool: 119, body tetrahedra: 328, surface triangles: 212
        loader = new TetGenLoader
            (dataDir + "sphere.ascii.1.node", 
             dataDir + "sphere.ascii.1.ele", 
             dataDir + "sphere.ascii.1.smesh");
        scale = 0.3;
    } else if (name == "bar") {
        //Bar: vpool: 119, body tetrahedra: 328, surface triangles: 212
        loader = new TetGenLoader
            (dataDir + "testbar2.ascii.1.node", 
             dataDir + "testbar2.ascii.1.ele", 
             dataDir + "testbar2.ascii.1.smesh");
        scale = 1;
    } else if (name == "bunny") {
        //bunny: vpool: , body tetrahedra: , surface triangles: 
        loader = new TetGenLoader
            (dataDir + "bunny.ascii.1.node",
             dataDir + "bunny.ascii.1.ele",
             dataDir + "bunny.ascii.1.smesh");
        scale = 30;
    } else
        throw Core::Exception("unknown solid");

    loader->Load();

    logger.info << "number of vertices: "
                << loader->GetVertexPool().size() << logger.end;
    logger.info << "number of body tetrahedra: " 
                << loader->GetBody().size() << logger.end;
    logger.info << "number of surface triangles: " 
                << loader->GetSurface().size() << logger.end;

    CHECK_FOR_CUDA_ERROR();
    solid = new Solid();
	solid->state = new TetrahedralTLEDState(); 
    solid->vertexpool = TypeConverter
        ::ConvertToVertexPool(loader->GetVertexPool());
    solid->body = TypeConverter
        ::ConvertToBody(loader->GetBody());
    solid->surface = TypeConverter
        ::ConvertToSurface(loader->GetSurface());

    solid->vertexpool->Scale(scale);
    return solid;
}
