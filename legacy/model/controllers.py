class LearningRateScheduler:
    def __init__(self, initial_lr, lr_decay, decay_every, learning_algo):
        self.lr = initial_lr
        self._lr_decay = lr_decay
        self._decay_every = decay_every
        self._model = learning_algo

        self._model.setLearningRate(self.lr)
        self._n_seen_samples_overall = 0

    def new_samples_seen(self, n_samples):
        self._n_seen_samples_overall += n_samples

        if self._n_seen_samples_overall % self._decay_every == 0 and self._n_seen_samples_overall:
            self.lr *= self._lr_decay
            self._model.setLearningRate(self.lr)
