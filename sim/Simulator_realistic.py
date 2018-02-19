#Load libraries
import random
import numpy as np

from ctypes import *

BSTRESH = 2.4
libc = CDLL("../libs/libBoatModel.dylib")

#Rename functions for readability
model_initialize = libc.Model_2_EXPORT_Discrete_initialize
model_step = libc.Model_2_EXPORT_Discrete_step
model_terminate = libc.Model_2_EXPORT_Discrete_terminate

#ctypes structured data for model state variables
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
    

class Simulator_realistic:
	
	def __init__(self, simulation_duration, hdg0, speed0):
		#Simulation variables and parameters
		self.simulation_duration = simulation_duration
		self.time_step = 0.01
		self.simulation_time = 0
        
		#Model variables
		self.DW = DW_Model_2_EXPORT_Discrete_T()
		self.M_ = RT_MODEL_Model_2_EXPORT_Discr_T()
		self.M = pointer(self.M_)
		self.M_.dwork = pointer(self.DW)
		
		#Input output instancing
		
		self.U_hdg_consigne = c_double()#main variable
		
		self.U_sailpos = c_double()
		self.U_truewindspeed = c_double()
		self.U_truewindheading = c_double()
		self.U_truewaterspeed = c_double()
		self.U_truewaterheading = c_double()
		self.U_hdg0 = c_double()
		self.U_speed0 = c_double()
		self.Y_Windincidence = c_double()
		self.Y_SpeedOverGround = c_double()
		
		#Initialize model
		model_initialize(self.M,
				         byref(self.U_sailpos),
				         byref(self.U_hdg_consigne), 
				         byref(self.U_truewindspeed), 
				         byref(self.U_truewindheading), 
				         byref(self.U_truewaterspeed), 
				         byref(self.U_truewaterheading), 
				         byref(self.U_hdg0),
				         byref(self.U_speed0),
				         byref(self.Y_Windincidence),
				         byref(self.Y_SpeedOverGround))
		
		#Define constant values, everything not redeclared is 0 by default		   
		self.U_sailpos.value = -40
		self.U_hdg0.value = hdg0
		self.U_speed0.value = speed0
		self.U_truewindspeed.value = 15

	def step(self, delta_hdg, truewindheading, truewindheading_std, duration, precision):
				
		self.U_hdg_consigne.value = self.U_hdg_consigne.value + delta_hdg
		self.U_truewindheading.value = random.gauss(truewindheading, truewindheading_std)
		
		incidence = []
		sog = []
		
		incidence.append(float(self.Y_Windincidence.value))
		sog.append(float(self.Y_SpeedOverGround.value))
		
		time = 0
		print(duration)
		while time < duration:
			
			model_step(self.M,
					   self.U_hdg_consigne,
					   self.U_sailpos,
					   self.U_truewindspeed,
					   self.U_truewindheading,
					   self.U_truewaterspeed,
					   self.U_truewaterheading,
					   self.U_hdg0,
					   self.U_speed0,
					   byref(self.Y_Windincidence),
					   byref(self.Y_SpeedOverGround))
			
			time = round(time,2)
			if round(time/precision,3)%1 == 0:
				incidence.append(float(self.Y_Windincidence.value))
				sog.append(float(self.Y_SpeedOverGround.value))
			
			self.simulation_time = self.simulation_time + self.time_step
			time = time + self.time_step
			
		incidence = np.array(incidence)
		sog = np.array(sog)
		print(sog)
		
		return incidence,sog
	
	def terminate(self):
		model_terminate(self.M)
	
		
		
