#ifndef _KEY_HANDLER_H_
#define _KEY_HANDLER_H_

#include <Core/IListener.h>
#include <Core/IEngine.h>
#include <Devices/Symbols.h>
#include <Devices/IKeyboard.h>
#include <Display/Camera.h>
#include <Math/Vector.h>

#include "TLEDNode.h"

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
 public:

 KeyHandler(Camera& camera, TLEDNode* tled, Core::IEngine& engine)
     : camera(camera), tled(tled), engine(engine) {}

    void SetEye(Math::Vector<3,float> eye) {
        this->eye = eye;
    }
    void SetPoint(Math::Vector<3,float> point) {
        this->point = point;
    }
    void Handle(KeyboardEventArg arg) {
        float xStep = 1.0f;
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
            case KEY_c:
                tled->crackStrategy->ApplyCrackTracking(tled->solid);
                break;
            case KEY_n: //reset
                Scene::ISceneNode* parentnode;
                parentnode = tled->GetParent();
                // Save numIterations
                int numItr = tled->numIterations;

                // remove old node
                parentnode->RemoveNode(tled);
                engine.InitializeEvent().Detach(*tled);
                engine.ProcessEvent().Detach(*tled);
                engine.DeinitializeEvent().Detach(*tled);
                tled->Handle(Core::DeinitializeEventArg());
                delete tled;
                logger.info << "deletion of tled done" << logger.end;

                // add new node
                tled = new TLEDNode();
                parentnode->AddNode(tled);
                tled->Handle(Core::InitializeEventArg());
                engine.ProcessEvent().Attach(*tled);
                engine.DeinitializeEvent().Attach(*tled);

                
                break;
            case KEY_q:
                logger.info << "camera = {position: " << camera.GetPosition()
                            << ", direction: " << camera.GetDirection()
                            << "}" << logger.end;
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
