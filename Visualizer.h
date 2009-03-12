 /**
 * Short description.
 *
 * @class Visualizer Visualizer.h ts/TLED/Visualizer.h
 */
#ifndef _VISUALIZER_H_
#define _VISUALIZER_H_

#include <Meta/OpenGL.h>
#include <Logging/Logger.h>
#include <map>
#include "VisualShapes.h"
#include "Visualization_kernels.h"
#include <cufft.h>
#include <cutil.h>
#include <cuda.h>
#include <driver_types.h> // includes cudaError_t
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h> // includes cudaMalloc and cudaMemset


class Visualizer {
private:
    VisualBuffer* vb;
    
    void RegisterBufferObject(VisualBuffer& vBuf) {
        // create buffer object
        glGenBuffers( 1, &vBuf.vboID);
        // TODO: error check genBuffer
        logger.info << "glGenBufferID: " << vBuf.vboID << logger.end;
        glBindBuffer( GL_ARRAY_BUFFER, vBuf.vboID);
        // initialize buffer object
        glBufferData( GL_ARRAY_BUFFER, vBuf.byteSize, NULL, GL_DYNAMIC_DRAW);
        glBindBuffer( GL_ARRAY_BUFFER, 0);
        // register buffer object with CUDA
        CUDA_SAFE_CALL(cudaGLRegisterBufferObject(vBuf.vboID));
        CUT_CHECK_ERROR_GL();
    }
    
    unsigned int sizeOfElement(GLenum mode){
        switch(mode){
        case POINTS:
            return sizeof(float4);
        case LINES:
            return 2 * sizeof(float4);
        case TRIANGLES: 
            return 3 * sizeof(float4);
        default: break;
        }
        return 0;
    }


public:
    Visualizer() {
        vb = new VisualBuffer[NUM_BUFFERS];
    }
    
    void AllocBuffer(int id, int numElm, GLenum mode) {    
        vb[id].numElm = numElm;
        vb[id].enabled = true;
        vb[id].mode = mode;
        
        vb[id].byteSize = sizeOfElement(mode) * numElm;
        logger.info << "bufferSize: " << vb[id].byteSize << logger.end;
            
        // Cuda malloc
        cudaMalloc((void**)&vb[id].buf, vb[id].byteSize);
        cudaMemset(vb[id].buf, 0, vb[id].byteSize);
            
        // Register with cuda
        RegisterBufferObject(vb[id]);
            
        // Map VBO id to buffer
        CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&vb[id].buf, vb[id].vboID));    
    }

    void AllocPolyBuffer(int id, int numElm, PolyShape ps) {
        printf("[AllocPolyBuffer] alloc poly buffer with numElm: %i numPolyVertices: %i\n", numElm, ps.numVertices);

        vb[id].numElm = numElm;
        vb[id].enabled = true;
        vb[id].mode = POLYGONS;
 
        // ------------------ MATRIX BUFFER ----------------- //
        int matByteSize = numElm * sizeof(float4) * 4;
        // Cuda malloc array for matrix transformations of the vertices.
        cudaMalloc((void**)&vb[id].buf, matByteSize); // float4 x 4 = 16 (4x4 matix) 
        cudaMemset(vb[id].buf, 0, matByteSize);

        // ----------- VERTEX BUFFER ------------------- //            
        // Each element has a float4 pr. vertex
        vb[id].byteSize = numElm * sizeof(float4) * ps.numVertices; 
        // Cuda malloc array for the actual vertices of the polygon model.
        cudaMalloc((void**)&vb[id].bufExt, vb[id].byteSize);
        cudaMemset(vb[id].bufExt, 0, vb[id].byteSize);

        // create buffer object
        glGenBuffers( 1, &vb[id].vboID);
        // TODO: error check genBuffer
        logger.info << "glGenBufferID: " << vb[id].vboID << logger.end;
        glBindBuffer( GL_ARRAY_BUFFER, vb[id].vboID);
        // initialize buffer object
        glBufferData( GL_ARRAY_BUFFER, vb[id].byteSize, NULL, GL_DYNAMIC_DRAW);
        glBindBuffer( GL_ARRAY_BUFFER, 0);
        // register buffer object with CUDA
        CUDA_SAFE_CALL(cudaGLRegisterBufferObject(vb[id].vboID));
        CUT_CHECK_ERROR_GL();
           
        // Map VBO id to buffer
        CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&vb[id].bufExt, vb[id].vboID));    
 
        // Copy the poly shape to cuda, one for each element
        cudaError_t stat;
        stat = cudaMemcpy(vb[id].bufExt, ps.vertices, ps.numVertices * sizeof(float4), cudaMemcpyHostToDevice);
        if( stat == cudaSuccess )
            printf("PolyShape copied successfully to gfx\n");
 
        CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vb[id].vboID ));

    }

    float4* GetBuffer(unsigned int id) {
        return vb[id].buf;
    }


    void MapAllBufferObjects() {   
        // Map VBO id to buffer
        for( int i=0; i<NUM_BUFFERS; i++ )
            CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&vb[i].buf, vb[i].vboID));
    }

    void UnmapAllBufferObjects() {
        // Unmap VBO
        for( int i=0; i<NUM_BUFFERS; i++ )
            CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vb[i].vboID ));
    }


    /**
     * Enables rendering for specified buffer id.
     *
     * @param id of visual buffer to enable rendering for.
     */
    void Enable(int id) {
        vb[id].enabled = true;
    }


    /**
     * Disables rendering for specified buffer id.
     *
     * @param id of visual buffer to enable rendering for.
     */
    void Disable(int id) {
        vb[id].enabled = false;
    }

    /**
     * Render all visual buffers that are enabled for rendering.
     */
    void Render() {
        for(int i=0; i<NUM_BUFFERS; i++) {
            if( vb[i].enabled && vb[i].vboID > 0 ) {
 
               // Draw VBO
                glShadeModel(GL_FLAT);
                glEnable(GL_DEPTH_TEST);

                glEnable(GL_AUTO_NORMAL); 
                glEnable(GL_NORMALIZE); 
         
                glEnableClientState(GL_VERTEX_ARRAY);
      
                // If the visual buffer is a polygon the vertex buffer
                // must be calculated by applying transformation matrix to model.
                if( vb[i].mode == POLYGONS ){

                    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&vb[i].bufExt, vb[i].vboID));
                    applyTransformation(vb[i]);
                    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vb[i].vboID ));

                    glBindBuffer(GL_ARRAY_BUFFER, vb[i].vboID);
                    glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);
                    glDrawArrays(GL_TRIANGLES, 0, 36);
                }
                else {
 
                    glBindBuffer(GL_ARRAY_BUFFER, vb[i].vboID);
                    glPointSize(20);
                    // TODO change number of GL_FLOATS according to vb.mode
                    glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glDrawArrays(vb[i].mode, 0, vb[i].numElm*vb[i].mode);
                }

                glDisableClientState(GL_VERTEX_ARRAY);
                glBindBuffer(GL_ARRAY_BUFFER, 0);

                glDisable(GL_DEPTH_TEST);
                CHECK_FOR_GL_ERROR();
            }
        }
    }
};

#endif

