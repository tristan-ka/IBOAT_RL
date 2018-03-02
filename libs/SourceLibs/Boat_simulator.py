#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iBoat SIMULATOR

@author: albert bonet
"""

#Load libraries
from ctypes import *
import matplotlib.pyplot as plt

libc = CDLL("./libBoatModel.so")

#Rename functions for readability
initialize = libc.Model_2_EXPORT_Discrete_initialize
step = libc.Model_2_EXPORT_Discrete_step
terminate = libc.Model_2_EXPORT_Discrete_terminate

#Declare C_types structures
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

#Initialize state variables
Model_2_EXPORT_Discrete_DW = DW_Model_2_EXPORT_Discrete_T()
Model_2_EXPORT_Discrete_M_ = RT_MODEL_Model_2_EXPORT_Discr_T()

Model_2_EXPORT_Discrete_M = pointer(Model_2_EXPORT_Discrete_M_)
Model_2_EXPORT_Discrete_M_.dwork = pointer(Model_2_EXPORT_Discrete_DW)

#Declare inputs and outputs
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


#Initialize model
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
           byref(Model_2_EXPORT_Discrete_Y_SpeedOverGround))

#Simulation parameters
timer = 0
simulation_time = 50
step_time = 0.01

#Post process variables
time = []
incidence = []
SOG = []

#Simulation loop
while timer < simulation_time :
    
    #Perform control calculations here
    pass 

    if timer%0.1 == 0:
        pass #call control functions
    
    #Update inputs
    Model_2_EXPORT_Discrete_U_sailpos = c_double(40)
    Model_2_EXPORT_Discrete_U_rudderpos = c_double(0)
    Model_2_EXPORT_Discrete_U_truewindspeed = c_double(12)
    Model_2_EXPORT_Discrete_U_truewindheading = c_double(60)
    Model_2_EXPORT_Discrete_U_truewaterspeed = c_double(0)
    Model_2_EXPORT_Discrete_U_truewaterheading = c_double(0)
    Model_2_EXPORT_Discrete_U_hdg0 = c_double(0)
    Model_2_EXPORT_Discrete_U_speed0 = c_double(10)
    
    #Step the model
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
         byref(Model_2_EXPORT_Discrete_Y_SpeedOverGround))
    
    #Perform measurements and other real time processing here
    timer = round(timer,2)
    if timer%1==0: #Measures every second
        time.append(timer)
        incidence.append(float(Model_2_EXPORT_Discrete_Y_Windincidence.value))
        SOG.append(float(Model_2_EXPORT_Discrete_Y_SpeedOverGround.value))
    
    #Step simulation time
    timer = timer + step_time
    

#Perform post-process here
plt.plot(time,incidence)
plt.ylabel('incidence')
plt.show()

plt.plot(time,SOG)
plt.ylabel('SpeedOverGround')
plt.show()

#Terminate model    
terminate(Model_2_EXPORT_Discrete_M)


