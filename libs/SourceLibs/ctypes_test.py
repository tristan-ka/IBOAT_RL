#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:39:46 2018

@author: albert
"""

from ctypes import *

libc = CDLL("./libBoatModel.so")
initialize = libc.Model_2_EXPORT_Discrete_initialize
step = libc.Model_2_EXPORT_Discrete_step
terminate = libc.Model_2_EXPORT_Discrete_terminate

class DW_Model_2_EXPORT_Discrete_T(Structure):
    _fields_ = [("incidencemeasure_DSTATE", c_double),
                ("speedmeasure_DSTATE", c_double),
                ("FixPtUnitDelay1_DSTATE", c_double),
                ("DiscreteTimeIntegrator1_DSTATE", c_double),
                ("DiscreteTimeIntegrator_DSTATE", c_double),
                ("DiscreteTimeIntegrator_DSTATE_p", c_double),
                ("DiscreteFilter_states", c_double),
                ("Discretemotormodel1_states", c_double),
                ("Discretemotormodel_states", c_double),
                ("PrevY", c_double),
                ("PrevY_d", c_double),
                ("FixPtUnitDelay2_DSTATE", c_ubyte),
                ("DiscreteTimeIntegrator_IC_LOADI", c_ubyte),
                ("DiscreteTimeIntegrator_IC_LOA_m", c_ubyte),
                ("G3_PreviousInput", c_ubyte)]
    
class RT_MODEL_Model_2_EXPORT_Discr_T(Structure):
    _fields_ = [("dwork",POINTER(DW_Model_2_EXPORT_Discrete_T))]


Model_2_EXPORT_Discrete_DW = DW_Model_2_EXPORT_Discrete_T()
Model_2_EXPORT_Discrete_M_ = RT_MODEL_Model_2_EXPORT_Discr_T()

Model_2_EXPORT_Discrete_M = pointer(Model_2_EXPORT_Discrete_M_)
Model_2_EXPORT_Discrete_M_.dwork = pointer(Model_2_EXPORT_Discrete_DW)


Model_2_EXPORT_Discrete_U_sailpos = c_double()
Model_2_EXPORT_Discrete_U_rudderpos = c_double()
Model_2_EXPORT_Discrete_U_truewindspeed = c_double()
Model_2_EXPORT_Discrete_U_truewindheading = c_double()
Model_2_EXPORT_Discrete_U_truewaterspeed = c_double()
Model_2_EXPORT_Discrete_U_truewaterheading = c_double()
Model_2_EXPORT_Discrete_U_hdg0 = c_double()
Model_2_EXPORT_Discrete_U_speed0 = c_double()
Model_2_EXPORT_Discrete_Y_Windincidence = c_double()
Model_2_EXPORT_Discrete_Y_SpeedOverGround = c_double()



initialize(Model_2_EXPORT_Discrete_M,
           byref(Model_2_EXPORT_Discrete_U_sailpos),
           byref(Model_2_EXPORT_Discrete_U_rudderpos), 
           byref(Model_2_EXPORT_Discrete_U_truewindspeed), 
           byref(Model_2_EXPORT_Discrete_U_truewindheading), 
           byref(Model_2_EXPORT_Discrete_U_truewaterspeed), 
           byref(Model_2_EXPORT_Discrete_U_truewaterheading), 
           byref(Model_2_EXPORT_Discrete_U_hdg0),
           byref(Model_2_EXPORT_Discrete_U_speed0),
           byref(Model_2_EXPORT_Discrete_Y_Windincidence),
           byref(Model_2_EXPORT_Discrete_Y_SpeedOverGround));


timer = 0
while timer<10:
    print("time:",timer)  
    
    #Input definition
    Model_2_EXPORT_Discrete_U_sailpos = c_double(40)
    Model_2_EXPORT_Discrete_U_rudderpos = c_double(0)
    Model_2_EXPORT_Discrete_U_truewindspeed = c_double(12)
    Model_2_EXPORT_Discrete_U_truewindheading = c_double(60)
    Model_2_EXPORT_Discrete_U_truewaterspeed = c_double(0)
    Model_2_EXPORT_Discrete_U_truewaterheading = c_double(0)
    Model_2_EXPORT_Discrete_U_hdg0 = c_double(0)
    Model_2_EXPORT_Discrete_U_speed0 = c_double(10)
    
    
    step(Model_2_EXPORT_Discrete_M,
         Model_2_EXPORT_Discrete_U_sailpos,
         Model_2_EXPORT_Discrete_U_rudderpos, 
         Model_2_EXPORT_Discrete_U_truewindspeed, 
         Model_2_EXPORT_Discrete_U_truewindheading, 
         Model_2_EXPORT_Discrete_U_truewaterspeed, 
         Model_2_EXPORT_Discrete_U_truewaterheading, 
         Model_2_EXPORT_Discrete_U_hdg0,
         Model_2_EXPORT_Discrete_U_speed0,
         byref(Model_2_EXPORT_Discrete_Y_Windincidence),
         byref(Model_2_EXPORT_Discrete_Y_SpeedOverGround));
         
    print("Output:",Model_2_EXPORT_Discrete_Y_Windincidence)
    print("Output:",Model_2_EXPORT_Discrete_Y_SpeedOverGround)
    timer = timer+0.01
    
terminate(Model_2_EXPORT_Discrete_M)


