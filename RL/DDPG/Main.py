
import tensorflow as tf
import sys

sys.path.append('./DDPG/')
sys.path.append("../sim/")

from Agent import Agent

from Displayer import DISPLAYER
from Saver import SAVER

'''
MDP Parameters
'''
history_duration = 6  # Duration of state history [s]
mdp_step = 1  # Step between each state transition [s]
time_step = 0.1  # time step [s] <-> 10Hz frequency of data acquisition
if __name__ == '__main__':
    
    tf.reset_default_graph()

    with tf.Session() as sess:

        agent = Agent(sess,history_duration, mdp_step, time_step)
        SAVER.set_sess(sess)

        SAVER.load()

        print("Beginning of the run")
        try:
            agent.run()
        except KeyboardInterrupt:
            pass
        print("End of the run")

        # Sauvegarde de l'agent désigné par son nombre de pas total et un commentaire
        commentaire = "_5000_epochs_state_60_buffer_5000"
        SAVER.save(agent.total_steps,commentaire)
        DISPLAYER.disp(commentaire)
        DISPLAYER.disp_q(commentaire)
        DISPLAYER.disp_critic_loss(commentaire)
        DISPLAYER.disp_actor_loss(commentaire)
        DISPLAYER.disp_i_v(commentaire)
