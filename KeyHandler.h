#ifndef _KEY_HANDLER_H_
#define _KEY_HANDLER_H_

#include <Core/IListener.h>
#include <Core/IEngine.h>
#include <Devices/Symbols.h>
#include <Devices/IKeyboard.h>
#include <Display/Camera.h>
#include <Math/Vector.h>

#include "TLEDNode.h"
#include "ForceModifier.h"
#include "SolidFactory.h"
#include "MaterialPropertiesFactory.h"

using namespace OpenEngine;
using namespace OpenEngine::Display;
using namespace OpenEngine::Devices;

class KeyHandler : public Core::IListener<KeyboardEventArg> {
 private:
    Display::Camera& camera;
    TLEDNode* tled;
    Core::IEngine& engine;
    Math::Vector<3,float> eye;
    Math::Vector<3,float> point;
    std::string solidname, mpname;
 public:

 KeyHandler(Camera& camera, TLEDNode* tled, Core::IEngine& engine, 
            std::string solidname, std::string mpname)
     : camera(camera), tled(tled), engine(engine),
        solidname(solidname), mpname(mpname) {}

    void SetEye(Math::Vector<3,float> eye) {
        this->eye = eye;
    }
    void SetPoint(Math::Vector<3,float> point) {
        this->point = point;
    }
    void Handle(KeyboardEventArg arg) {
        float xStep = 1.0f;
        int numItr = 0;
        if (arg.type == EVENT_PRESS) {
            switch (arg.sym) {
            case KEY_r:
                camera.SetPosition(eye);
                camera.LookAt(point);
                break;
            case KEY_o:
                if (tled->numIterations < 50)
                    tled->numIterations++;
                logger.info << "physics number of iterations = " << 
                    tled->numIterations << logger.end;
                break;
            case KEY_l:
                if (tled->numIterations > 0)
                    tled->numIterations--;
                logger.info << "physics number of iterations = " << 
                    tled->numIterations << logger.end;
                break;
            case KEY_SPACE:
                tled->paused = ! tled->paused;
                logger.info << "physics paused  = " << 
                    tled->paused << logger.end;
                break;
            case KEY_i:
                tled->minX += xStep;
                logger.info << "x plane = " << 
                    tled->minX << logger.end;
                break;
            case KEY_k:
                tled->minX -= xStep;
                logger.info << "x plane = " << 
                    tled->minX << logger.end;
                break;
            case KEY_p:
                tled->renderPlane = !tled->renderPlane;
                break;
            case KEY_b:
                tled->useAlphaBlending = !tled->useAlphaBlending;
                break;
            case KEY_0: 
                tled->crackTrackingEnabled = !tled->crackTrackingEnabled;
                logger.info << "CrackTrackingEnabled = " << tled->crackTrackingEnabled << logger.end;
                break;
            case KEY_c:
                try {
                    tled->crackStrategy->ApplyCrackTracking(tled->solid);
                }catch(Core::Exception ex) { 
                    tled->paused = true; 
                    logger.info << "EXCEPTION: " << ex.what() << logger.end;
                }
                break;
            case KEY_z:
                tled->crackStrategy->InitializeCrack(tled->solid);
                break;
            case KEY_v:
                tled->crackTrackAllWay = !tled->crackTrackAllWay;
                logger.info << "Crack All way through = " << 
                    tled->crackTrackAllWay << logger.end;
                break;
            case KEY_n: //reset
                Scene::ISceneNode* parentnode;
                parentnode = tled->GetParent();
                // Save numIterations
                numItr = tled->numIterations;

                // remove old node
                parentnode->RemoveNode(tled);
                engine.InitializeEvent().Detach(*tled);
                engine.ProcessEvent().Detach(*tled);
                engine.DeinitializeEvent().Detach(*tled);
                tled->Handle(Core::DeinitializeEventArg());

                delete tled;
                logger.info << "deletion of tled done" << logger.end;

                {
                    // add new node
                    Solid* solid = SolidFactory::Create(solidname);
                    solid->SetMaterialProperties(MaterialPropertiesFactory::Create(mpname));
                    tled = new TLEDNode(solid);
                }
                parentnode->AddNode(tled);
                tled->Handle(Core::InitializeEventArg());
                engine.ProcessEvent().Attach(*tled);
                engine.DeinitializeEvent().Attach(*tled);
                
                // Restore numIterations 
                tled->numIterations = numItr;
                break;
            case KEY_q:
                logger.info << "camera = {position: " << camera.GetPosition()
                            << ", direction: " << camera.GetDirection()
                            << "}" << logger.end;
                break;

            case KEY_LEFT:
                tled->modifier.front()->Move(-0.05,0,0);
                break;
            case KEY_RIGHT:
                tled->modifier.front()->Move(0.05,0,0);
                break;
            case KEY_UP:
                tled->modifier.front()->Move(0,0.04,0);
                break;
            case KEY_DOWN:
                tled->modifier.front()->Move(0,-0.04,0);
                break;
            case KEY_j:
                tled->modifier.front()->Rotate(0, 0.004, 0);
                break;
            case KEY_h:
                tled->modifier.front()->Rotate(0, -0.004, 0);
                break;
            case KEY_m:
                ((ForceModifier*)tled->modifier.front())->SelectNodes(tled->solid);
                break;

            case KEY_x:
                tled->dump = true;
                break;
            case KEY_t:
                if (tled->paused)
                    tled->StepPhysics();
                break;
            case KEY_1:
                tled->vbom->Toggle(SURFACE_VERTICES);
                tled->vbom->Toggle(SURFACE_NORMALS);
                break;
            case KEY_2:
                tled->vbom->Toggle(BODY_MESH);
                //                tled->vbom->Toggle(BODY_COLORS);
                //tled->vbom->Toggle(BODY_NORMALS);
                break;
            case KEY_3:
                tled->vbom->Toggle(CENTER_OF_MASS);
                break;
            case KEY_4:
                tled->vbom->Toggle(STRESS_TENSOR_VERTICES);
                break;
            default: break;
            }
        }
    }
};

#endif
