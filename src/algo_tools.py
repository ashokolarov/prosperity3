class WelfordStatsWithPriors:
    def __init__(self, initial_mean=None, initial_variance=None, initial_count=None):
        self.n = initial_count if initial_mean is not None else 0
        self.mean = initial_mean if initial_mean is not None else 0.0
        self.M2 = (
            initial_variance * initial_count if initial_variance is not None else 0.0
        )

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean(self):
        return self.mean

    def get_std(self):
        return (self.M2 / self.n if self.n > 1 else 1.0) ** 0.5
