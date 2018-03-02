import sys

from visualization import Visualization

# from DDPGvisualization import DDPGVisualization

sys.path.append("../sim/")

from Simulator import TORAD

v = Visualization(3, 1, 0.1, 2, 1, 45 * TORAD, 0 * TORAD, 0, "../Networks/dqn-test", 80)
anim = v.generateDeltaAnimation(20)


