#include "VboManager.h"
#include "CUDA.h"

VboManager::VboManager() {
    vb = new VisualBuffer[NUM_BUFFERS];
}

VboManager::~VboManager() {
    for (int i = 0; i<NUM_BUFFERS; i++ )
        cudaGLUnregisterBufferObject(vb[i].vboID);
    CHECK_FOR_CUDA_ERROR();
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

    for (int i = 0; i<NUM_BUFFERS; i++ )
        glDeleteBuffersARB(1, &vb[i].vboID);

    delete[] vb;
    printf("[VboManger] all cleaned up\n");
}


void VboManager::RegisterBufferObject(VisualBuffer& vBuf) {
    // create buffer object
    glGenBuffers( 1, &vBuf.vboID);
    // TODO: error check genBuffer
    printf("glGenBufferID: %i\n", (int)vBuf.vboID);
    // Bind buffer
    glBindBuffer( GL_ARRAY_BUFFER, vBuf.vboID);
    // initialize buffer object
    glBufferData( GL_ARRAY_BUFFER, vBuf.byteSize, NULL, GL_DYNAMIC_DRAW);
    // Unbind buffer
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    // Register buffer object with CUDA
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(vBuf.vboID));
    // Update total bytes allocated
    totalByteAlloc += vBuf.byteSize;
    // Check for errors
    CHECK_FOR_GL_ERROR();
}
    
unsigned int VboManager::sizeOfElement(GLenum mode){
    switch(mode){
    case GL_POINTS:
        return sizeof(float4);
    case GL_LINES:
        return 2 * sizeof(float4);
    case GL_TRIANGLES: 
        return 3 * sizeof(float4); // NOTE: float3 here!
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
    printf("numElm: %i   -  ByteSize: %i\n", numElm, vb[id].byteSize);
            
    // Register with cuda
    RegisterBufferObject(vb[id]);

    return vb[id];
}

VisualBuffer& VboManager::AllocBuffer(int id, int numElm, PolyShape ps) {
    printf("[AllocPolyBuffer] alloc poly buffer with numElm: %i numPolyVertices: %i\n", numElm, ps.numVertices);
    vb[id].numElm = numElm;        
    vb[id].enabled = true;
    vb[id].mode = GL_POLYGON;
    vb[id].numIndices = numElm * ps.numVertices;

    // ------------------ MATRIX BUFFER ----------------- //
    int matByteSize = numElm * sizeof(float4) * 4; // float4 x 4 = 16 (4x4 matix) 
    // Cuda malloc array for matrix transformations of the vertices.
    cudaMalloc((void**)&vb[id].matBuf, matByteSize); 
    cudaMemset(vb[id].matBuf, 0, matByteSize);
    totalByteAlloc += matByteSize;

    // ------------------ MODEL BUFFER --------------- //
    int byteSize = ps.numVertices * sizeof(float4);
    cudaMalloc((void**)&(vb[id].modelBuf), byteSize); 
    totalByteAlloc += byteSize;

    // Copy the poly shape once to cuda.
    cudaError_t stat;
    stat = cudaMemcpy(vb[id].modelBuf, ps.vertices, byteSize, cudaMemcpyHostToDevice);
    if( stat == cudaSuccess )
        printf("PolyShape uploaded successfully\n");
 
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
        if( vb[i].mode != GL_POLYGON ) {
            //printf("mapping bufferAddress %i -  with ID: %i \n", vb[i].buf, vb[i].vboID);
            CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&vb[i].buf, vb[i].vboID));
        }
}

void VboManager::UnmapAllBufferObjects() {
    // Unmap VBO
    for( int i=0; i<NUM_BUFFERS; i++ )
        if( vb[i].mode != GL_POLYGON ) {
            CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vb[i].vboID ));
        }
}

/**
 * Returns true if specified buffer is enabled, false otherwise.
 *
 * @param id of visual buffer.
 */
bool VboManager::IsEnabled(int id) { 
    return vb[id].enabled;
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


/**
 * Render all visual buffers that are enabled for rendering.
 */
void VboManager::Render(int id) {
    VisualBuffer& buf = vb[id];
    if( buf.enabled && buf.vboID > 0 ) {
        // Set color
        float4 color = buf.color;
        float4 ambcolor;
        ambcolor.x = 1.0 * color.x;
        ambcolor.y = 1.0 * color.y;
        ambcolor.z = 1.0 * color.z;
        ambcolor.w = 0.1 * color.w;
        glMaterialfv(GL_FRONT, GL_DIFFUSE, (GLfloat*)&color);
        glMaterialfv(GL_FRONT, GL_AMBIENT, (GLfloat*)&ambcolor);
        glColor4f(color.x,color.y,color.z,color.w);
     
        // Draw VBO
        /*
          glShadeModel(GL_FLAT);
          glEnable(GL_DEPTH_TEST);
          glEnable(GL_AUTO_NORMAL); 
          glEnable(GL_NORMALIZE); 
        */
        // If the visual buffer is a polygon the vertex buffer
        // must be calculated by applying transformation matrix to model.
        if( buf.mode == GL_POLYGON ) {
            CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&buf.buf, buf.vboID));
            applyTransformation(buf);
            CUDA_SAFE_CALL(cudaGLUnmapBufferObject( buf.vboID ));
 
            glBindBuffer(GL_ARRAY_BUFFER, buf.vboID);
            glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);
            glEnableClientState(GL_VERTEX_ARRAY);
            glDrawArrays(GL_TRIANGLES, 0, buf.numIndices);
            //                printf("numIndices: %i\n", buf.numIndices);
            CHECK_FOR_GL_ERROR();
        }
        else {
            glBindBuffer(GL_ARRAY_BUFFER, buf.vboID);
            glPointSize(2);
            glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);
            glEnableClientState(GL_VERTEX_ARRAY);
            glDrawArrays(buf.mode, 0, buf.numIndices);
            CHECK_FOR_GL_ERROR();
        }

        glDisableClientState(GL_VERTEX_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        //glDisable(GL_DEPTH_TEST);
        CHECK_FOR_GL_ERROR();
    }
}


void VboManager::Render(VisualBuffer& vert, VisualBuffer& colr, VisualBuffer& norm) {
    if( vert.enabled && vert.vboID > 0 && colr.vboID > 0 && norm.vboID > 0 ) {
        // Draw VBO
        /*glShadeModel(GL_FLAT);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_AUTO_NORMAL); 
        */

        //       glEnable(GL_CULL_FACE);
        glEnable(GL_NORMALIZE); 
        glEnable(GL_COLOR_MATERIAL);
        glShadeModel(GL_FLAT);

        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);

        // Enable color array
        glBindBufferARB(GL_ARRAY_BUFFER, colr.vboID);
        glColorPointer( 3, GL_FLOAT, sizeof(float4), 0 );
        glEnableClientState( GL_COLOR_ARRAY );
            
        // Enable normal array
        glBindBufferARB(GL_ARRAY_BUFFER, norm.vboID);
        glNormalPointer(GL_FLOAT, sizeof(float4), 0);
        glEnableClientState(GL_NORMAL_ARRAY);
 
        // If the visual buffer is a polygon the vertex buffer
        // must be calculated by applying transformation matrix to model.
        if( vert.mode == GL_POLYGON ) {
            CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&vert.buf, vert.vboID));
            applyTransformation(vert);
            CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vert.vboID ));
        
            glBindBuffer(GL_ARRAY_BUFFER, vert.vboID);
            glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);
            glEnableClientState(GL_VERTEX_ARRAY);
            
            glDrawArrays(GL_TRIANGLES, 0, vert.numIndices);
        }
        else {
            glBindBuffer(GL_ARRAY_BUFFER, vert.vboID);
            glPointSize(2);
            glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);
            glEnableClientState(GL_VERTEX_ARRAY);
 
            glDrawArrays(vert.mode, 0, vert.numIndices);
        }

        glDisableClientState( GL_VERTEX_ARRAY );
        glDisableClientState( GL_COLOR_ARRAY );
        glDisableClientState( GL_NORMAL_ARRAY );

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBufferARB(GL_ARRAY_BUFFER, 0);

        //glDisable(GL_DEPTH_TEST);
        CHECK_FOR_GL_ERROR();
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
