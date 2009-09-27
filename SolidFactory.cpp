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
    if (name == "tetrahedra") {
        //tetrahedra: vpool: 4, body tetrahedra: 1, surface triangles: 4
        loader = new TetGenLoader
            (dataDir + "tetrahedron.1.node",
             dataDir + "tetrahedron.1.ele",
             dataDir + "tetrahedron.1.smesh");
        scale = 5.0;
    } else if (name == "box") {
        //box: vpool: 14, body tetrahedra: 17, surface triangles: 24
        loader = new TetGenLoader
            (dataDir + "box.1.node",
             dataDir + "box.1.ele",
             dataDir + "box.1.smesh");
        scale = 10;
    } else if (name == "sphere") {
        //sphere: vpool: 119, body tetrahedra: 328, surface triangles: 212
        loader = new TetGenLoader
            (dataDir + "sphere.1.node", 
             dataDir + "sphere.1.ele", 
             dataDir + "sphere.1.smesh");
        scale = 0.3;
    } else if (name == "bar_20x20x20" || name == "bar20") {
        loader = new TetGenLoader
            (dataDir + "bar_20x20x20.1.node", 
             dataDir + "bar_20x20x20.1.ele", 
             dataDir + "bar_20x20x20.1.smesh");
        scale = 1;
    } else if (name == "bar_10x10x10" || name == "bar10") {
        loader = new TetGenLoader
            (dataDir + "bar_10x10x10.1.node", 
             dataDir + "bar_10x10x10.1.ele", 
             dataDir + "bar_10x10x10.1.smesh");
        scale = 1;
    } else if (name == "bar_5x5x5" || name == "bar5") {
        loader = new TetGenLoader
            (dataDir + "bar_5x5x5.1.node", 
             dataDir + "bar_5x5x5.1.ele", 
             dataDir + "bar_5x5x5.1.smesh");
        scale = 1;
    } else if (name == "bar_2_5x2_5x2_5" || name == "bar2_5") {
        loader = new TetGenLoader
            (dataDir + "bar_2_5x2_5x2_5.1.node", 
             dataDir + "bar_2_5x2_5x2_5.1.ele", 
             dataDir + "bar_2_5x2_5x2_5.1.smesh");
        scale = 1;
    } else if (name == "tooth_slice") {
        loader = new TetGenLoader
            (dataDir + "tooth_slice.1.node",
             dataDir + "tooth_slice.1.ele",
             dataDir + "tooth_slice.1.smesh");
        scale = 10;
    } else if (name == "tooth_noslice") {
        loader = new TetGenLoader
            (dataDir + "tooth_noslice.1.node",
             dataDir + "tooth_noslice.1.ele",
             dataDir + "tooth_noslice.1.smesh");
        scale = 10;
    } else if (name == "tooth_slice_simple") {
        loader = new TetGenLoader
            (dataDir + "tooth_slice_simple.1.node",
             dataDir + "tooth_slice_simple.1.ele",
             dataDir + "tooth_slice_simple.1.smesh");
        scale = 10;
    } else if (name == "tooth_noslice_simple") {
        loader = new TetGenLoader
            (dataDir + "tooth_noslice_simple.1.node",
             dataDir + "tooth_noslice_simple.1.ele",
             dataDir + "tooth_noslice_simple.1.smesh");
        scale = 10;
    } else if (name == "test_tooth") {
       loader = new TetGenLoader
            (dataDir + "test_tooth.1.node",
             dataDir + "test-tooth.1.ele",
             dataDir + "test_tooth.1.smesh");
        scale = 1.0;
    } else if (name == "test_tooth_high_res") {
       loader = new TetGenLoader
            (dataDir + "test_tooth_high_res.1.node",
             dataDir + "test-tooth_high_res.1.ele",
             dataDir + "test_tooth_high_res.1.smesh");
        scale = 1.0;
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
