Package RL
==========

Policy Learner
-----------------

.. automodule:: policyLearning
	:members:
	:undoc-members:
	:show-inheritance:

DQN
-----------------

.. automodule:: dqn
	:members:
	:undoc-members:   
	:show-inheritance:

DDPG    
-----------------
.. automodule:: DDPG
	:members:
	:undoc-members:
	:show-inheritance:


Tutorial
---------------


.. code-block:: python
   :emphasize-lines: 37,38,39,42

    history_duration = 3  # Duration of state history [s]
    mdp_step = 1  # Step between each state transition [s]
    time_step = 0.1  # time step [s] <-> 10Hz frequency of data acquisition
    mdp = MDP(history_duration, mdp_step, time_step)

    mean = 45 * TORAD
    std = 0 * TORAD
    wind_samples = 10
    WH = np.random.uniform(mean - std, mean + std, size=10)

    hdg0=0*np.ones(10)
    mdp.initializeMDP(hdg0,WH)

    hdg0_rand_vec=(-4,0,2,4,6,8,18,20,21,22,24)

    action_size = 2
    policy_angle = 18
    agent = PolicyLearner(mdp.size, action_size, policy_angle)
    #agent.load("policy_learning_i18_test_long_history")
    batch_size = 120

    EPISODES = 500

    for e in range(EPISODES):
        WH = w.generateWind()
        hdg0_rand = random.choice(hdg0_rand_vec) * TORAD
        hdg0 = hdg0_rand * np.ones(10)

        mdp.simulator.hyst.reset()

        #  We reinitialize the memory of the flow
        state = mdp.initializeMDP(hdg0, WH)
        loss_sim_list = []
        for time in range(80):
            WH = w.generateWind()
            action = agent.act(state)
            next_state, reward = mdp.transition(action, WH)
            agent.remember(state, action, reward, next_state)  # store the transition + the state flow in the
            # final state !!
            state = next_state
            if len(agent.memory) >= batch_size:
                loss_sim_list.append(agent.replay(batch_size))
                print("time: {}, Loss = {}".format(time, loss_sim_list[-1]))
                print("i : {}".format(mdp.s[0, -1] / TORAD))
            # For data visualisation
        loss_over_simulation_time = np.sum(np.array([los    s_sim_list])[0]) / len(np.array([loss_sim_list])[0])
        loss_of_episode.append(loss_over_simulation_time)
