import sys
from visualization import ValidateNetwork
sys.path.append("../sim/")
import matplotlib.pyplot as plt
from Simulator import TORAD

#v=ValidateNetwork(6,1,0.1,2,45*TORAD,0*TORAD,10,0,"../Networks/epsilon_pi",320)
#f1 = v.generateQplots()

v2 =ValidateNetwork(4,1,0.1,2,45*TORAD,0*TORAD,10,0,"../Networks/epsilon_pi_1convlayer",320)
f2 = v2.generateQplots()
