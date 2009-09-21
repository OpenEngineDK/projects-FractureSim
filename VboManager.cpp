#include "VboManager.h"
#include <fstream>

#include "CudaMem.h"
#include <Logging/Logger.h>

VboManager::VboManager() {
    vb = new VisualBuffer[NUM_BUFFERS];
}

VboManager::~VboManager() {
    // Unregister all vbo's
    for (int i = 0; i<NUM_BUFFERS; i++ )
        if( vb[i].vboID > 0 )
            cudaGLUnregisterBufferObject(vb[i].vboID);

    // Free matrix and model buffers on polygon buffers
    for (int i = 0; i<NUM_BUFFERS; i++ )
        if( vb[i].mode == GL_POLYGON && vb[i].vboID > 0 ) {
            cudaFree(vb[i].matBuf);
            cudaFree(vb[i].modelVertBuf);
            cudaFree(vb[i].modelNormBuf);
        }
            
    CHECK_FOR_CUDA_ERROR();
    
    // Unbind buffer
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

    for (int i = 0; i<NUM_BUFFERS; i++ )
        if (vb[i].vboID != 0)
            FreeGLBuffer(vb[i].vboID);

    // Check for errors
    CHECK_FOR_GL_ERROR();
    CHECK_FOR_CUDA_ERROR();

    delete[] vb;
    //printf("[VboManger] Cleaned up\n");
}


void VboManager::RegisterBufferObject(VisualBuffer& vBuf) {
    vBuf.vboID = AllocGLBuffer(vBuf.byteSize);
}
    
unsigned int VboManager::sizeOfElement(GLenum mode){
    switch(mode){
    case GL_POINTS:
        return sizeof(float4);
    case GL_LINES:
        return 2 * sizeof(float4);
    case GL_TRIANGLES: 
        return 3 * sizeof(float4);
    default: 
        printf("[VboManager] ERROR: unknown buffer mode, use GL_POINTS, GL_LINES og GL_TRIANGLES\n");
        exit(-1);
        break;
    }
    return 0;
}

unsigned int VboManager::indicesForMode(GLenum mode) {
    switch(mode){
    case GL_POINTS:
        return 1;
    case GL_LINES:
        return 2;
    case GL_TRIANGLES: 
        return 3;
    default: break;
    }
    return 0;
}

VisualBuffer& VboManager::AllocBuffer(int id, int numElm, GLenum mode){
    vb[id].numElm = numElm;
    vb[id].enabled = true;
    vb[id].mode = mode;        
    vb[id].byteSize = sizeOfElement(mode) * numElm;
    vb[id].numIndices = numElm * indicesForMode(mode);
            
    // Register with cuda
    RegisterBufferObject(vb[id]);

    return vb[id];
}

VisualBuffer& VboManager::AllocBuffer(int id, int numElm, PolyShape ps) {
    vb[id].numElm = numElm;        
    vb[id].enabled = true;
    vb[id].mode = GL_TRIANGLES;
    vb[id].numIndices = numElm * ps.numVertices;

    // ------------------ MATRIX BUFFER ----------------- //
    cudaError_t stat;
    int matByteSize = numElm * sizeof(float4) * 4; // float4 x 4 = 16 (4x4 matix) 
    // Cuda malloc array for matrix transformations of the vertices.
    stat = CudaMemAlloc((void**)&(vb[id].matBuf), matByteSize);
    if( stat == cudaSuccess )
        CudaMemset(vb[id].matBuf, 0, matByteSize);
    else printf("[VboManager] Error: could not allocate matrix buffer\n");
    
    // ------------------ MODEL VERTEX BUFFER --------------- //
    int byteSize = ps.numVertices * sizeof(float4);
    stat = CudaMemAlloc((void**)&(vb[id].modelVertBuf), byteSize);
    if( stat != cudaSuccess )
        printf("[VboManager] Error: could not allocate model buffer\n");
    // Copy the poly shape once to cuda.
    stat = CudaMemcpy(vb[id].modelVertBuf, ps.vertices, byteSize, cudaMemcpyHostToDevice);
    if( stat == cudaSuccess )
        printf("PolyShape vertices uploaded successfully\n");

    // ------------------ MODEL NORMAL BUFFER --------------- //
    stat = CudaMemAlloc((void**)&(vb[id].modelNormBuf), byteSize);
    if( stat != cudaSuccess )
        printf("[VboManager] Error: could not allocate normal buffer\n");
    // Copy the poly shape once to cuda.
    stat = CudaMemcpy(vb[id].modelNormBuf, ps.normals, byteSize, cudaMemcpyHostToDevice);
    if( stat == cudaSuccess )
        printf("PolyShape normals uploaded successfully\n");

    // ----------- VERTEX BUFFER ------------------- //            
    // Each element has a float4 pr. vertex
    vb[id].byteSize = numElm * ps.numVertices * sizeof(float4); 
          
    // Register with cuda
    RegisterBufferObject(vb[id]);
 
    CHECK_FOR_GL_ERROR();   
    return vb[id];
}


VisualBuffer& VboManager::GetBuf(unsigned int id) {
    //printf("bufAddr: %i\n", vb[id].buf);
    return vb[id];
}


void VboManager::MapAllBufferObjects() {   
    // Map VBO id to buffer
    for( int i=0; i<NUM_BUFFERS; i++ ) 
        if( vb[i].vboID > 0 ) {
            //printf("mapping bufferAddress %i -  with ID: %i \n", vb[i].buf, vb[i].vboID);
            CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&vb[i].buf, vb[i].vboID));
        }
}

void VboManager::UnmapAllBufferObjects() {
    // Unmap VBO
    for( int i=0; i<NUM_BUFFERS; i++ )
        if( vb[i].vboID > 0 ) {
            CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vb[i].vboID ));
        }
}

/**
 * Returns true if specified buffer is enabled, false otherwise.
 *
 * @param id of visual buffer.
 */
bool VboManager::IsEnabled(int id) { 
    return vb[id].enabled && vb[id].vboID > 0;
}

/**
 * Enables rendering for specified buffer id.
 *
 * @param id of visual buffer to enable rendering for.
 */
void VboManager::Enable(int id) {
    vb[id].enabled = true;
}


/**
 * Disables rendering for specified buffer id.
 *
 * @param id of visual buffer to enable rendering for.
 */
void VboManager::Disable(int id) {
    vb[id].enabled = false;
}

/**
 * Toggle rendering of specified buffer. 
 *
 * @param id of visual buffer to enable rendering for.
 */
void VboManager::Toggle(int id) {
    vb[id].enabled = !vb[id].enabled;
}



void VboManager::RenderWithNormals(VisualBuffer& vertBuf, VisualBuffer& normBuf) {
    UseNormalArray(normBuf);
    Render(vertBuf);
    glDisableClientState( GL_NORMAL_ARRAY );
}

void VboManager::RenderWithColors(VisualBuffer& vertBuf, VisualBuffer& colrBuf, bool useAlpha) {
    if( useAlpha )
        glDisable(GL_DEPTH_TEST);

    UseColorArray(colrBuf, useAlpha);
    Render(vertBuf);
    glDisableClientState(GL_COLOR_ARRAY);
    glEnable(GL_DEPTH_TEST);
}

void VboManager::Render(VisualBuffer& vert, VisualBuffer& colr, VisualBuffer& norm) {
    Render(vert, colr, norm, false);
}

void VboManager::Render(VisualBuffer& vert, VisualBuffer& colr, VisualBuffer& norm, bool useAlpha) {
    if( useAlpha )
        glDisable(GL_DEPTH_TEST);

    UseColorArray(colr, useAlpha);  
    UseNormalArray(norm);
    Render(vert);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glEnable(GL_DEPTH_TEST);
}

void VboManager::Render(int id) {
    VisualBuffer& buf = vb[id];
    if( buf.enabled )
        Render(buf);
}

void VboManager::Render(VisualBuffer& vert) {
    if( vert.enabled && vert.vboID > 0 ) {
        //        glEnable(GL_AUTO_NORMAL);
        glEnable(GL_NORMALIZE); 
        glEnable(GL_COLOR_MATERIAL);
        //glShadeModel(GL_FLAT);
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
       
        //glDisable(GL_CULL_FACE);
        //glEnable(GL_CULL_FACE);
        //glCullFace(GL_BACK);
        //glFrontFace(GL_CCW);

        glBindBuffer(GL_ARRAY_BUFFER, vert.vboID);
        glPointSize(10);
        glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);
        glEnableClientState(GL_VERTEX_ARRAY);
        glDrawArrays(vert.mode, 0, vert.numIndices);
        CHECK_FOR_GL_ERROR();
         
        glDisableClientState(GL_VERTEX_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

void VboManager::UseColorArray(VisualBuffer& colr, bool useAlpha) {
    // Enable normal array
    if( colr.vboID > 0 ) {
        glBindBufferARB(GL_ARRAY_BUFFER, colr.vboID);
        if( useAlpha )
            glColorPointer(4, GL_FLOAT, sizeof(float4), 0);
        else
            glColorPointer(3, GL_FLOAT, sizeof(float4), 0);
        glEnableClientState(GL_COLOR_ARRAY);
    }
}

void VboManager::UseNormalArray(VisualBuffer& norm) {
    // Enable normal array
    if( norm.vboID > 0 ) {
        glBindBufferARB(GL_ARRAY_BUFFER, norm.vboID);
        glNormalPointer(GL_FLOAT, sizeof(float4), 0);
        glEnableClientState(GL_NORMAL_ARRAY);
    }
}



////////////////////////////////////////////////////////////////////////////////
//! Check if the result is correct or write data to file for external
//! regression testing
////////////////////////////////////////////////////////////////////////////////
void VboManager::dumpBufferToFile(char* filename, VisualBuffer& vb){
    return dumpBufferToFile(filename, vb.vboID, vb.byteSize);
}

void VboManager::dumpBufferToFile(char* filename, GLuint vboID, unsigned int size) {
    CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(vboID));

    // map buffer object
    glBindBuffer( GL_ARRAY_BUFFER_ARB, vboID );
    float* data = (float*) glMapBuffer( GL_ARRAY_BUFFER, GL_READ_ONLY);

    // write file for regression test
    CUT_SAFE_CALL( cutWriteFilef( filename, data, size, false));
 
    // unmap GL buffer object
    if( ! glUnmapBuffer( GL_ARRAY_BUFFER)) {
        fprintf( stderr, "Unmap buffer failed.\n");
        fflush( stderr);
    }

    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(vboID));

    CHECK_FOR_CUDA_ERROR();
    CHECK_FOR_GL_ERROR();
}

void VboManager::CopyBufferDeviceToHost(VisualBuffer& vb, std::string filename) {

    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&vb.matBuf, vb.vboID));

    // Alloc buffer
    float4* data = (float4*)malloc(vb.byteSize);
    // Copy
    CudaMemcpy(data, vb.matBuf, vb.byteSize, cudaMemcpyDeviceToHost);

    std::ofstream output(filename.c_str());

    for(unsigned int i=0; i<vb.numIndices; i++)
    {
        float4 v = data[i];

        /*        if( (i % 3) == 0 ){
            output << "Dot = " << dot(data[i], data[i+1]) << " ----- " <<  dot(data[i], data[i+2]) << " ---- " <<  dot(data[i+1], data[i+2]) << std::endl;
            output << "Length: " << length(data[i]) << ", " << length(data[i+1]) << ", " << length(data[i+2]) << std::endl;
        } 
        */

        output << v.x << ", " << v.y << ", " << v.z << ", " << v.w << std::endl;
    }

    free(data);

    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vb.vboID ));
    
    CHECK_FOR_CUDA_ERROR();
    CHECK_FOR_GL_ERROR();
}

