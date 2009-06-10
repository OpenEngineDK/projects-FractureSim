// 
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------


#ifndef _CRACK_STRATEGY_H_
#define _CRACK_STRATEGY_H_

class Solid;

class CrackStrategy {
public:
    CrackStrategy() {}
    virtual ~CrackStrategy() {}

    // Returns true if crack has been initialized.
    virtual bool CrackInitialized(Solid* solid) { return false; }

    // Handles the crack tracking strategy
    virtual void ApplyCrackTracking(Solid* solid) {}

    // Returns true if the solid has been fragmented into 
    // two or more separate parts.
    virtual bool FragmentationDone() { return false; }

    // Debug
    virtual void RenderDebugInfo(Solid* solid) {}

};

#endif _CRACK_STRATEGY_H_
