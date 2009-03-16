#include "VboManager.h"

VboManager::VboManager() {
    vb = new VisualBuffer[NUM_BUFFERS];
}

VboManager::~VboManager() {

    for (int i = 0; i<NUM_BUFFERS; i++ )
        cudaGLUnregisterBufferObject(vb[i].vboID);
    CUT_CHECK_ERROR("cudaGLUnregisterBufferObject failed");
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
    // Check for errors
    CUT_CHECK_ERROR_GL();
}
    
unsigned int VboManager::sizeOfElement(GLenum mode){
    switch(mode){
    case GL_POINTS:
        return sizeof(float4);
    case GL_LINES:
        return 2 * sizeof(float4);
    case GL_TRIANGLES: 
        return 3 * sizeof(float4); // NOTE: float3 here!
    default: break;
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

    // ------------------ MODEL BUFFER --------------- //
    // Upload the original model once. 
    //float4* vertices = ps.vertices;
    

    // ----------- VERTEX BUFFER ------------------- //            
    // Each element has a float4 pr. vertex
    vb[id].byteSize = numElm * ps.numVertices * sizeof(float4); 
          
    // Register with cuda
    RegisterBufferObject(vb[id]);
 
    // Map VBO id to buffer
    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&vb[id].buf, vb[id].vboID));    
 
    // Copy the poly shape to cuda, one for each element
    cudaError_t stat;
    stat = cudaMemcpy(vb[id].buf, ps.vertices, ps.numVertices * sizeof(float4), cudaMemcpyHostToDevice);
    if( stat == cudaSuccess )
        printf("PolyShape copied successfully to gfx\n");
 
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vb[id].vboID ));
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
        CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vb[i].vboID ));
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
 * Render all visual buffers that are enabled for rendering.
 */
void VboManager::Render() {
    for(int i=0; i<NUM_BUFFERS; i++) {
        if( vb[i].enabled && vb[i].vboID > 0 ) {
            // Set color
            float4 color = vb[i].color;
            float4 ambcolor;
            ambcolor.x = 1.0 * color.x;
            ambcolor.y = 1.0 * color.y;
            ambcolor.z = 1.0 * color.z;
            ambcolor.w = 0.1 * color.w;
            glMaterialfv(GL_FRONT, GL_DIFFUSE, (GLfloat*)&color);
            glMaterialfv(GL_FRONT, GL_AMBIENT, (GLfloat*)&ambcolor);
            glColor4f(color.x,color.y,color.z,color.w);
                
            // Draw VBO
            glShadeModel(GL_FLAT);
            glEnable(GL_DEPTH_TEST);

            glEnable(GL_AUTO_NORMAL); 
            glEnable(GL_NORMALIZE); 
         
            // If the visual buffer is a polygon the vertex buffer
            // must be calculated by applying transformation matrix to model.
            if( vb[i].mode == GL_POLYGON ) {
                CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&vb[i].buf, vb[i].vboID));
                applyTransformation(vb[i]);
                CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vb[i].vboID ));

                //if( ((PolygonBuffer*)vb[i]).normBuf != NULL )
                //    printf("Has normals\n");
 
                glBindBuffer(GL_ARRAY_BUFFER, vb[i].vboID);
                glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);
                glEnableClientState(GL_VERTEX_ARRAY);
                glDrawArrays(GL_TRIANGLES, 0, vb[i].numIndices);
                //printf("numElm: %i", vb[i].numElm);
            }
            else {
                glBindBuffer(GL_ARRAY_BUFFER, vb[i].vboID);
                glPointSize(2);
                glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);
                glEnableClientState(GL_VERTEX_ARRAY);
                glDrawArrays(vb[i].mode, 0, vb[i].numIndices);
            }

            glDisableClientState(GL_VERTEX_ARRAY);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            glDisable(GL_DEPTH_TEST);
            CUT_CHECK_ERROR_GL();
        }
    }
}
