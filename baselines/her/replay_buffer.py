import threading

import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        # for storing transitions that are not a whole batch
        self.current_outer_idx = 0
        self.current_inner_idx = 0

        # self.lock = threading.Lock()

    @property
    def full(self):
        # with self.lock:
        return self.current_size == self.size

    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        # with self.lock:
        assert self.current_size > 0
        for key in self.buffers.keys():
            buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        transitions = self.sample_transitions(buffers, batch_size)

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def sample_for_ggs(self, t, batch_size):
        """
        a sample function for training a supervised global goal setter
        :param t: time steps in advance we want to predict
        :return: transitions of batch_size containing (s, s+t, g)
        """
        buffers = {}

        # with self.lock:
        assert self.current_size > 0
        for key in self.buffers.keys():
            buffers[key] = self.buffers[key][:self.current_size]
        # import pdb; pdb.set_trace()
        episode_idxs = np.random.randint(0, self.current_size, batch_size)
        # max_index = min(self.T, self.current_inner_idx)
        max_index = self.T
        s = np.random.randint(0, max_index - t, batch_size)
        s_plus_t = s + t
        ends = s_plus_t + (np.random.uniform(size=s.shape) * (min(max_index, 500) - s_plus_t)).astype(int)

        return dict (s=buffers['ag'][episode_idxs, s], s_plus_t=buffers['ag'][episode_idxs, s_plus_t], g=buffers['ag'][episode_idxs, ends])




    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        # with self.lock:
        idxs = self._get_storage_idx(batch_size)

        # load inputs into buffers
        for key in self.buffers.keys():
            self.buffers[key][idxs] = episode_batch[key]

        self.n_transitions_stored += batch_size * self.T

    def store_transition(self, transition):
        # with self.lock:
        for key in self.buffers.keys():
            self.buffers[key][self.current_outer_idx][self.current_inner_idx] = transition[key]
        if self.current_inner_idx:
            self.buffers['successes'][self.current_outer_idx][self.current_inner_idx - 1] = 0
        self.current_inner_idx +=1
        self.n_transitions_stored += 1

    def new_transition(self):
        self.current_inner_idx = 0
        self.current_outer_idx = self._get_storage_idx()

    def get_current_episode_size(self):
        # with self.lock:
        return self.current_size

    def get_current_size(self):
        # with self.lock:
        return self.current_size * self.T

    def get_transitions_stored(self):
        # with self.lock:
        return self.n_transitions_stored

    def clear_buffer(self):
        # with self.lock:
        self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx
