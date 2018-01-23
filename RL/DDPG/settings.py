
ENV = "IBOAT"

LOAD = False    # Réutilisation d'un NN existant ou non
DISPLAY = True  # Veut-on avoir la courbe d'apprentissage


DISCOUNT = 0.95

FRAME_SKIP = 0

# Rudder action bounds
HIGH_BOUND = 1.0
LOW_BOUND = -1.0


# CONVOLUTIONAL NEURAL NETWORK PARAMETERS
ACTOR_NETWORK_FILTERS = 40

ACTOR_LEARNING_RATE = 5e-4
CRITIC_LEARNING_RATE = 5e-4

# Memory size
BUFFER_SIZE = 5000
BATCH_SIZE = 32

# Number of episodes of game environment to train with
EPISODES = 2000
REWARD_SHAPING = True

# Maximal number of steps during one episode
TIME_PER_EPISODE = 60
TRAINING_FREQ = 10
PLOT_I_V = 10 # Number of last episodes to plot incidence and speed

# Rate to update target network toward primary network
UPDATE_TARGET_RATE = 0.9


# scale of the exploration noise process (1.0 is the range of each action
# dimension)
NOISE_SCALE_INIT = 1

# EXPLORATION
EPSILON = 1
# decay rate (per episode) of the scale of the exploration noise process
FREQ_DECAY = 800 # Factor NOISE_DECAY applied every FREQ_DECAY episodes
NOISE_DECAY = 0.5

# settings for the exploration noise process:
# dXt = theta*(mu-Xt)*dt + sigma*dWt
EXPLO_MU = 0.0
EXPLO_THETA = 0.0
EXPLO_SIGMA = 0.2

# Display Frequencies
DISP_EP_REWARD_FREQ = 1
PLOT_FREQ = 100000
RENDER_FREQ = 1
GIF_FREQ = 1
