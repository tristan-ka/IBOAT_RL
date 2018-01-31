import sys

from visualization import ValidateNetwork

sys.path.append("../sim/")

from Simulator import TORAD

v = ValidateNetwork(6, 1, 0.1, 2, 1, 45 * TORAD, 0 * TORAD, 0, "../Networks/epsilon_pi", 320)
f1 = v.generateQplots()

#v2 = ValidateNetwork(6, 1, 0.1, 2, 1, 45 * TORAD, 0 * TORAD, 0, "../Networks/dqn_epsilon_batch50", 80)
#f2 = v2.simulateDQNControl(18)
#f3 = v2.simulateDQNControl(0)
