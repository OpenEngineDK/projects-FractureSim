#ifndef _KEY_HANDLER_H_
#define _KEY_HANDLER_H_

#include <Core/IListener.h>
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
    TLEDNode& tled;
    Math::Vector<3,float> eye;
    Math::Vector<3,float> point;
 public:
 KeyHandler(Camera& camera, TLEDNode& tled) : camera(camera), tled(tled) {}
    void SetEye(Math::Vector<3,float> eye) {
        this->eye = eye;
    }
    void SetPoint(Math::Vector<3,float> point) {
        this->point = point;
    }
    void Handle(KeyboardEventArg arg) {
        float xStep = 5.0f;
        if (arg.type == EVENT_PRESS) {
            switch (arg.sym) {
            case KEY_r:
                camera.SetPosition(eye);
                camera.LookAt(point);
                break;
            case KEY_o:
                if (tled.numIterations < 50)
                    tled.numIterations++;
                logger.info << "physics number of iterations = " << 
                    tled.numIterations << logger.end;
                break;
            case KEY_l:
                if (tled.numIterations > 0)
                    tled.numIterations--;
                logger.info << "physics number of iterations = " << 
                    tled.numIterations << logger.end;
                break;
            case KEY_SPACE:
                tled.paused = ! tled.paused;
                logger.info << "physics paused  = " << 
                    tled.paused << logger.end;
                break;
            case KEY_i:
                tled.minX += xStep;
                logger.info << "x plane = " << 
                    tled.minX << logger.end;
                break;
            case KEY_k:
                tled.minX -= xStep;
                logger.info << "x plane = " << 
                    tled.minX << logger.end;
                break;
            case KEY_p:
                tled.renderPlane = !tled.renderPlane;
                break;
            default: break;
            }
        }
    }
};

#endif
