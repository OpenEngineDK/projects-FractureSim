#ifndef _HELPER_FUNCTIONS_
#define _HELPER_FUNCTIONS_

#include <Math/Vector.h>
#include <Geometry/Line.h>
#include <Geometry/FaceSet.h>
#include <Scene/GeometryNode.h>
#include <Scene/TransformationNode.h>
#include <Scene/RenderStateNode.h>

using OpenEngine::Math::Vector;
using OpenEngine::Geometry::Line;
using namespace OpenEngine::Geometry;

//from: http://www.openengine.dk
//      /code/extensions/FFMPEGResource/Utils/Billboard.h
static Scene::TransformationNode* Create
(int textureHosisontalSize, int textureVerticalSize, float scale, 
 Vector<4,float> color) {
    FaceSet* faces = new FaceSet();
    
    float horisontalhalfsize = textureHosisontalSize * 0.5;
    Vector<3,float>* lowerleft = 
        new Vector<3,float>(horisontalhalfsize,0,0);//-textureVerticalSize*0.5,0);
    Vector<3,float>* lowerright =
        new Vector<3,float>(-horisontalhalfsize,0,0);//-textureVerticalSize*0.5,0);
    Vector<3,float>* upperleft =
        new Vector<3,float>(horisontalhalfsize,textureVerticalSize*0.5,0);
    Vector<3,float>* upperright =
        new Vector<3,float>(-horisontalhalfsize,textureVerticalSize*0.5,0);
    
    FacePtr leftside = FacePtr(new Face(*lowerleft,*lowerright,*upperleft));
    leftside->texc[1] = Vector<2,float>(1,0);
    leftside->texc[0] = Vector<2,float>(0,0);
    leftside->texc[2] = Vector<2,float>(0,1);
    leftside->norm[0] = leftside->norm[1] 
        = leftside->norm[2] = Vector<3,float>(0,0,-1);
    leftside->CalcHardNorm();
    leftside->Scale(scale);
    faces->Add(leftside);

    leftside->colr[0] = leftside->colr[1] = leftside->colr[2] = color;
        
    FacePtr rightside = FacePtr(new Face(*lowerright,*upperright,*upperleft));
    rightside->texc[2] = Vector<2,float>(0,1);
    rightside->texc[1] = Vector<2,float>(1,1);
    rightside->texc[0] = Vector<2,float>(1,0);
    rightside->norm[0] = rightside->norm[1]
        = rightside->norm[2] = Vector<3,float>(0,0,-1);
    rightside->CalcHardNorm();
    rightside->Scale(scale);
    faces->Add(rightside);
    
    rightside->colr[0] = rightside->colr[1] = rightside->colr[2] = color;

    MaterialPtr m = MaterialPtr(new Material());
	//m->texr = ResourceManager<ITexture2D>::Create(textureFile);
	leftside->mat = rightside->mat = m;
        
    Scene::GeometryNode* node = new Scene::GeometryNode();
    node->SetFaceSet(faces);

    Scene::RenderStateNode* rsn = new Scene::RenderStateNode();
    rsn->DisableOption(Scene::RenderStateNode::WIREFRAME);
    rsn->AddNode(node);

    Scene::TransformationNode* tnode = new Scene::TransformationNode();
    tnode->AddNode(rsn);

    tnode->Rotate(0.0, Math::PI/2, 0.0);    
    return tnode;
}

#endif // _HELPER_FUNCTIONS_
