#include "Solid.h"

Solid::Solid() {
    vertexpool = NULL;
    body = NULL;
    surface = NULL;
    state = NULL;
}

Solid::~Solid() {
}


void Solid::DeAlloc() {
    if (body != NULL)
        body->DeAlloc();
    if (surface != NULL)
        surface->DeAlloc();
    if (vertexpool != NULL)
        vertexpool->DeAlloc();
    vertexpool = NULL;
    body = NULL;
    surface = NULL;
    state = NULL;
}

void Solid::Print() {
    printf("--------- vertexpool --------\n");
    vertexpool->Print();
    printf("--------- body indices --------\n");
    body->Print();
    printf("--------- surface indecies --------\n");
    surface->Print();
    printf("--------- end  --------\n");    
}

bool Solid::IsInitialized() {
    if (state == NULL ||
        body == NULL ||
        surface == NULL ||
        vertexpool == NULL)
        return false;
    else
        return true;
}

void Solid::SetMaterialProperties(MaterialProperties* mp) {
    this->mp = mp;
}
