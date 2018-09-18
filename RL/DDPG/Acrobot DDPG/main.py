'''


¡¡¡¡¡¡IMPORTANT!!!!!


This acrobot uses a customized gym version
in order for it to work, clone the gym github repository, go to
folder gym/envs/classic_control, and replace the acrobot.py file by the one that is 
in the same folder as this main.py file. Once done that, install gym in your 
computer as indicated in its github page:

git clone https://github.com/openai/gym.git (you should have already done this command)
cd gym
pip install -e .


'''




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
            agent.run()
        except KeyboardInterrupt:
            agent.save("NetworkParam/FinalParam")
        print("End of the run")
        DISPLAYER.disp()
        

        agent.play(5)

    agent.close()
