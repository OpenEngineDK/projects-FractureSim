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
    BODY_COLORS,
    BODY_NORMALS,
    EIGEN_VALUES,
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

    bool IsEnabled(int id);
    void Enable(int id);
    void Disable(int id);
    void Toggle(int id);

    void Render(int id);
    
    void Render(VisualBuffer& vert, VisualBuffer& colr, VisualBuffer& norm);

    // debug
    void dumpBufferToFile(char* filename, VisualBuffer& vb);
    void dumpBufferToFile(char* filename, GLuint vboID, unsigned int size);

    void CopyBufferDeviceToHost(VisualBuffer& vb, float* data);

};

#endif //_VBO_MANAGER_H_

