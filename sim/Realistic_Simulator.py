# Load libraries
import numpy as np

from ctypes import *

BSTRESH = 2.4
libc = CDLL("../libs/libBoatModel.dylib")

# Rename functions for readability
model_initialize = libc.Model_2_EXPORT_Discrete_initialize
model_step = libc.Model_2_EXPORT_Discrete_step
model_terminate = libc.Model_2_EXPORT_Discrete_terminate


# ctypes structured data for model state variables
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
    _fields_ = [("dwork", POINTER(DW_Model_2_EXPORT_Discrete_T))]


class Realistic_Simulator:
    """
    This class holds a Simulink-based boat simulation, with methods to simulate and act upon it. Real simulation inputs
    are the heading reference for sail position is supposed to be fixed for a given configuration.

    :ivar float time_step: Smallest possible time step, inherited from Simulink model
    :ivar float simulation_time: Timer of the real simulation time
    :ivar c_double U_hdg_ref: Heading reference
    :ivar c_double U_sailpos: Sail position
    :ivar c_double U_truewindspeed: Absolute wind speed
    :ivar c_double U_truewindheading: Absolute wind heading
    :ivar c_double U_truewaterspeed: Absolute water speed
    :ivar c_double U_truewaterheading: Absolute water heading
    :ivar c_double U_hdg0: Initial absolute boat heading
    :ivar c_double U_speed0: Initial absolute boat speed
    :ivar c_double Y_Windincidence: Relative wind incidence wrt the wing
    :ivar c_double Y_SpeedOverGround: Absolute boat speed
    """
    def __init__(self):
        # Simulation variables and parameters
        self.time_step = 0.01
        self.simulation_time = 0

        # Model variables
        self.DW = DW_Model_2_EXPORT_Discrete_T()
        self.M_ = RT_MODEL_Model_2_EXPORT_Discr_T()
        self.M = pointer(self.M_)
        self.M_.dwork = pointer(self.DW)

        # Input output instancing
        self.U_hdg_ref = c_double()  # main variable
        self.U_sailpos = c_double()
        self.U_truewindspeed = c_double()
        self.U_truewindheading = c_double()
        self.U_truewaterspeed = c_double()
        self.U_truewaterheading = c_double()
        self.U_hdg0 = c_double()
        self.U_speed0 = c_double()
        self.Y_Windincidence = c_double()
        self.Y_SpeedOverGround = c_double()

        # Initialize Model
        model_initialize(self.M,
                         byref(self.U_sailpos),
                         byref(self.U_hdg_ref),
                         byref(self.U_truewindspeed),
                         byref(self.U_truewindheading),
                         byref(self.U_truewaterspeed),
                         byref(self.U_truewaterheading),
                         byref(self.U_hdg0),
                         byref(self.U_speed0),
                         byref(self.Y_Windincidence),
                         byref(self.Y_SpeedOverGround))

    def step(self, duration, precision):
        '''
        Advance the simulation a certain amount of time and perform incidence and speed measures
        with a certain sample time, stored in np.arrays

        :param float duration: How much to advance the simulation
        :param float precision: Sampling time for the measures
        :return: list of np.array Incidence, SpeedOverGround
        '''

        incidence = []
        sog = []

        incidence.append(float(self.Y_Windincidence.value))
        sog.append(float(self.Y_SpeedOverGround.value))

        time = 0
        while time < duration:
            model_step(self.M,
                       self.U_sailpos,
                       self.U_hdg_ref,
                       self.U_truewindspeed,
                       self.U_truewindheading,
                       self.U_truewaterspeed,
                       self.U_truewaterheading,
                       self.U_hdg0,
                       self.U_speed0,
                       byref(self.Y_Windincidence),
                       byref(self.Y_SpeedOverGround))
            
            #print("time",self.simulation_time,"incidence",self.Y_Windincidence.value,"speed",self.Y_SpeedOverGround.value)
            
            time = round(time, 2)
            if round(time / precision, 3) % 1 == 0:
                incidence.append(float(self.Y_Windincidence.value))
                sog.append(float(self.Y_SpeedOverGround.value))

            self.simulation_time = self.simulation_time + self.time_step
            time = time + self.time_step

        incidence = np.array(incidence)
        sog = np.array(sog)

        return incidence, sog

    def terminate(self):
        """
        Terminate the simulation and its variables
        """
        model_terminate(self.M)