
import os

import numpy as np
import matplotlib.pyplot as plt

import settings as settings


def save(saver, fig_name):
    if settings.DISPLAY:
        for path, data in saver:
            plt.plot(data)
        fig = plt.gcf()
        os.makedirs(os.path.dirname(fig_name), exist_ok=True)
        fig.savefig(fig_name)
        plt.show(block=False)
        fig.clf()
    else:
        for path, data in saver:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = " ".join(map(str, data))
            with open(path, "w") as file:
                file.write(data)


class Displayer:

    def __init__(self):
        self.rewards = []
        self.q_buf = []
        self.critic_loss = []
        self.actor_loss = []
        self.incidence = []
        self.vitesse = []
        self.q_luff = []
        self.q_bear_off = []

    def add_reward(self, reward):
        self.rewards.append(reward)
        if len(self.rewards) % settings.PLOT_FREQ == 0:
            if settings.DISPLAY:
                self.disp("")
            else:
                print(self.rewards[-50:])

    def disp(self, commentaire):
        mean_reward = [np.mean(self.rewards[max(1, i - 50):i])
                       for i in range(2, len(self.rewards))]
        saver = [("results/Reward", self.rewards),
                 ("results/Mean_reward", mean_reward)]
        save(saver, "results/Reward" + commentaire + ".png")

    def add_q(self, q, q_luff,q_bear_off):
        self.q_buf.append(q)
        self.q_luff.append(q_luff)
        self.q_bear_off.append(q_bear_off)

    def disp_q(self, commentaire):
        mean_q = [np.mean(self.q_buf[max(1, i - 10):i])
                  for i in range(1, len(self.q_buf))]
        saver = [("results/Q", self.q_buf),
                 ("results/Q_mean", mean_q), ("results/Q_luff", self.q_luff), ("results/Q_bear_off", self.q_bear_off)]
        save(saver, "results/Q" + commentaire + ".png")

    def add_critic_loss(self,loss):
        self.critic_loss.append(loss)

    def disp_critic_loss(self, commentaire):
        mean_loss = [[np.mean(self.critic_loss[max(1, i - 40):i])
                  for i in range(1, len(self.critic_loss))]]
        saver = [("results/critic_loss", self.critic_loss),
                 ("results/critic_loss", mean_loss)]
        save(saver, "results/critic_loss" + commentaire + ".png")

    def add_actor_loss(self,loss):
        self.actor_loss.append(loss)

    def disp_actor_loss(self, commentaire):
        mean_loss = [[np.mean(self.actor_loss[max(1, i - 40):i])
                  for i in range(1, len(self.actor_loss))]]
        saver = [("results/actor_loss", self.actor_loss),
                 ("results/actor_loss", mean_loss)]
        save(saver, "results/actor_loss" + commentaire + ".png")

    def add_i_v(self, incidence, vitesse):
        self.incidence.append(incidence)
        self.vitesse.append(vitesse)

    def disp_i_v(self, commentaire):
        saver_i = [("results/i", self.incidence)]
        save(saver_i, "results/i" + commentaire + ".png")
        saver_v = [("results/v", self.vitesse)]
        save(saver_v, "results/v" + commentaire + ".png")




DISPLAYER = Displayer()
