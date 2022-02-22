import numpy as np
import scipy.stats as stats

class GaussianMixture():
    def __init__(self, weights, means, covariances):
        self.ndim = len(means[0])
        self.n_components = len(means)
        self.means = means
        self.covariances = covariances
        self.weights = weights
        assert len(self.weights) == len(self.covariances) == len(self.means)
        for i in range(self.n_components):
            assert self.means[i].shape[0] == self.covariances[i].shape[0]
            assert self.means[i].shape[0] == self.covariances[i].shape[1]

    def pdf(self, x):
        values = [stats.multivariate_normal(
                    mean=self.means[i], cov=self.covariances[i]).pdf(x)
                  for i in range(self.n_components)]

        return np.average(np.array(values), weights=self.weights)

    def sample(self, n):
        indexes = np.random.choice(
                self.n_components, size=n, p=self.weights)

        samples_list = [stats.multivariate_normal(
                            self.means[i], cov=self.covariances[i]).rvs() 
                        for i in indexes]
        return np.array(samples_list)
