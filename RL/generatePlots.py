import sys

from visualization import ValidateNetwork
from DDPGvisualization import DDPGVisualization

sys.path.append("../sim/")

from Simulator import TORAD

#v = ValidateNetwork(6, 1, 0.1, 2, 1, 45 * TORAD, 0 * TORAD, 0, "../Networks/lighter_archi", 640)
#f1 = v.generateQplots()

# Simulation class for DDPG in same format as DQN
v2 = DDPGVisualization(6, 1, 0.1, 1, -3.0, 3.0, 45 * TORAD, 0 * TORAD, 0, "../Networks/DDPG/Network_Noise_1000",80)

f2 = v2.simulateDDPGControl(18)
#f3 = v2.simulateDQNControl(0)
f3 = v2.simulateDDPGControl(0)

#f4 = v2.simulateGustsControl()