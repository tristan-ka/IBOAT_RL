import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def plot_hist_best_policy(self, idscenario=-1, interactive=False):
    """
    Plot the best policy as in :py:meth:`plot_best_policy`, with the histogram of the best action at each node\
     (`Animation <https://matplotlib.org/api/animation_api.html>`_)
    :param int idscenario: id of the corresponding worker tree to be plot. If -1 (default), the global tree is plotted.
    :param bool interactive: if True the plot is not an animation but can be browsed step by step
    :return: the `figure <https://matplotlib.org/api/figure_api.html>`_ of the current plot
    """
    # check if the best_policy has been computed
    if len(self.best_policy) == 0:
        self.get_best_policy()

    # Get the right policy:
    nodes_policy = self.best_nodes_policy[idscenario]
    policy = self.best_policy[idscenario]

    # Plot
    fig, ax1 = self.plot_best_policy(idscenario=idscenario, number_subplots=2)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Histogram of returns for given action")
    barcollection = ax2.bar(Hist.MEANS, [0 for _ in Hist.MEANS],
                            Hist.THRESH[1] - Hist.THRESH[0])
    pt, = ax1.plot(0, 0, color="green", marker='o', markersize='7')
    x0, y0 = 0, 0
    x_list = [x0]
    y_list = [y0]
    for node in nodes_policy[1:]:
        x = x0 + 1 * sin(node.arm * pi / 180)
        y = y0 + 1 * cos(node.arm * pi / 180)
        x_list.append(x)
        y_list.append(y)
        x0, y0 = x, y

    def animate(i):
        n = nodes_policy[i]
        if i == len(nodes_policy) - 1:
            # last nodes: same reward for all actions
            a = 0
        else:
            a = A_DICT[policy[i]]
        if idscenario is -1:
            hist = sum(n.rewards[ii, a].h * self.probability[ii] for ii in range(len(n.rewards[:, a])))
        else:
            hist = n.rewards[idscenario, a].h
        for j, b in enumerate(barcollection):
            b.set_height(hist[j])
        ax2.set_ylim([0, np.max(hist) + 1])
        pt.set_data(x_list[i], y_list[i])


    return barcollection, pt


    anim = animation.FuncAnimation(fig, animate, frames=len(nodes_policy), interval=1000, blit=False)
    plt.show()

return anim