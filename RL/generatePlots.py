import sys

from visualization import Visualization

# from DDPGvisualization import DDPGVisualization

sys.path.append("../sim/")

from Simulator import TORAD

v = Visualization(3, 1, 0.1, 2, 1, 45 * TORAD, 0 * TORAD, 0, "../Networks/dqn_lighter_archi_loss", 100)
f1 = v.simulateDQNControl(0)
f2 = v.simulateDQNControl(20)
f4 = v.simulateGustsControl()


