
import tensorflow as tf

from DDPG.Agent import Agent

from Displayer import DISPLAYER
from DDPG.Saver import SAVER

import DDPG.settings as settings

if __name__ == '__main__':
    
    tf.reset_default_graph()

    with tf.Session() as sess:

        agent = Agent(sess)
        SAVER.set_sess(sess)

        SAVER.load()
        agent.play(1, "results/gif/gif_save".format(settings.ENV))

    agent.close()
