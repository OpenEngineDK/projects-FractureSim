#ifndef _KEY_HANDLER_H_
#define _KEY_HANDLER_H_

#include <Core/IListener.h>
#include <Devices/Symbols.h>
#include <Devices/IKeyboard.h>
#include <Display/Camera.h>
#include <Math/Vector.h>

using namespace OpenEngine;
using namespace OpenEngine::Display;
using namespace OpenEngine::Devices;

class KeyHandler : public Core::IListener<KeyboardEventArg> {
 private:
    Display::Camera& camera;
    Math::Vector<3,float> eye;
    Math::Vector<3,float> point;
 public:
    KeyHandler(Camera& camera) : camera(camera) {}
    void SetEye(Math::Vector<3,float> eye) {
        this->eye = eye;
    }
    void SetPoint(Math::Vector<3,float> point) {
        this->point = point;
    }
    void Handle(KeyboardEventArg arg) {
        if (arg.type == EVENT_PRESS) {
            switch (arg.sym) {
            case KEY_r:
                camera.SetPosition(eye);
                camera.LookAt(point);
                break;
            default: break;
            }
        }
    }
};

#endif
