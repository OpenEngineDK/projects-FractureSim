 /**
 * Short description.
 *
 * @class Visualizer Visualizer.h ts/TLED/Visualizer.h
 */
#ifndef _VBO_MANAGER_H_
#define _VBO_MANAGER_H_

#include "CUDA.h"

#include <map>
#include "Shapes.h"
#include "Visualization_kernels.h"

enum {
    SURFACE_VERTICES = 0,
    SURFACE_NORMALS,
    CENTER_OF_MASS,
    BODY_MESH,
    STRESS_TENSORS,
    NUM_BUFFERS
};


class VboManager {
private:
    VisualBuffer* vb;
    
    void RegisterBufferObject(VisualBuffer& vBuf);
    unsigned int sizeOfElement(GLenum mode);
    unsigned int indicesForMode(GLenum mode);

public:
    VboManager();
    ~VboManager();

    VisualBuffer& AllocBuffer(int id, int numElm, GLenum mode);
    VisualBuffer& AllocBuffer(int id, int numElm, PolyShape ps);

    VisualBuffer& GetBuf(unsigned int id);

    void MapAllBufferObjects();
    void UnmapAllBufferObjects();

    void Enable(int id);
    void Disable(int id);

    void Render();
    
};

#endif //_VBO_MANAGER_H_

