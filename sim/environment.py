import numpy as np


class wind:
    """
    Generetate the wind samples of the environment. The wind intesity is assumed constant and equal to 15 knots

    :ivar float mean: mean direction of the wind in [rad]
    :ivar float std: standard deviation of the wind direction in [rad]
    :ivar float samples: number of samples to generate
    """

    def __init__(self, mean, std, samples):
        self.mean = mean
        self.std = std
        self.samples = samples

    def generateWind(self):
        """
        Generates the wind samples
        :return: np.array of wind samples
        """
        return np.random.uniform(self.mean - self.std, self.mean + self.std, size=self.samples)
