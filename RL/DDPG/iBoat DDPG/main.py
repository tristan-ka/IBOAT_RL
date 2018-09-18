
import tensorflow as tf

from Agent import Agent

from Displayer import DISPLAYER

import parameters

if __name__ == '__main__':
    
    tf.reset_default_graph()

    with tf.Session() as sess:

        agent = Agent(sess)

        print("Beginning of the run")
        try:
            #print("I do not do the run")
            agent.run()
        except KeyboardInterrupt:
            agent.save("NetworkParam/FinalParam")
        print("End of the run")
        DISPLAYER.dispR()

        agent.playActor()
        agent.playCritic()

    #agent.close()
