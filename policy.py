from __future__ import division
import numpy as np
import config
import time

class Policy(object):

    def __init__(self):
        self.mask = None
        self.qlogger = None
        self.eps_forB = 0
        self.eps_forC = 0

    def _set_agent(self, agent):
        self.agent = agent

    def set_mask(self, mask):
        self.mask = mask

    @property
    def metrics_names(self):
        return []

    @property
    def metrics(self):
        return []

    def select_action(self, **kwargs):
        raise NotImplementedError()

    def get_config(self):
        return {}

    def log_qvalue(self, q_values):

        if self.qlogger is not None:

            if self.mask is not None:
                q_values = q_values - self.mask * 1e20

            self.qlogger.pre_maxq = self.qlogger.cur_maxq

            self.qlogger.cur_maxq = np.max(q_values)

            if self.qlogger.maxq < self.qlogger.cur_maxq:
                self.qlogger.maxq = self.qlogger.cur_maxq

            self.qlogger.mean_maxq.append(self.qlogger.cur_maxq)

class RandomPolicy(Policy):

    def select_action(self, q_values):
        #self.log_qvalue(q_values)
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        action = np.random.random_integers(0, nb_actions - 1)
        return action

class GreedyPolicy(Policy):

    def select_action(self, q_values):
        #self.log_qvalue(q_values)
        assert q_values.ndim == 1
        action = np.argmax(q_values)
        return action


class MaskedRandomPolicy(Policy):

    def __init__(self):
        super(MaskedRandomPolicy, self).__init__()
        self.mask = None

    def select_action(self, q_values):
        #self.log_qvalue(q_values)
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        probs = np.ones(nb_actions)
        if self.mask is not None:
            probs -= self.mask
        sum_probs = np.sum(probs)
        assert sum_probs >= 1.0
        probs /= sum_probs
        action = np.random.choice(range(nb_actions), p=probs)
        return action

    def get_config(self):
        config = super(MaskedRandomPolicy, self).get_config()
        return config


class MaskedGreedyPolicy(Policy):

    def __init__(self):
        super(MaskedGreedyPolicy, self).__init__()
        self.mask = None

    def select_action(self, q_values):
        #self.log_qvalue(q_values)
        assert q_values.ndim == 1
        if self.mask is not None:
            q_values += self.mask
        action = np.argmax(q_values)
        return action


class EpsGreedyPolicy(Policy):
    def __init__(self, eps=.1,end_eps=0.1, steps=200000):
        super(EpsGreedyPolicy, self).__init__()
        self.eps = eps
        self.end_eps = end_eps
        self.steps = steps

    def select_action(self, q_values):
        if self.eps > self.end_eps:
            self.eps -= (self.eps-self.end_eps)/self.steps

        if q_values.ndim == 1:
            nb_actions = q_values.shape[0]

            if np.random.uniform() < self.eps:
                action = np.random.random_integers(0, nb_actions - 1)
            else:
                action = np.argmax(q_values)
            return action
        elif q_values.ndim == 2:
            nb_actions = q_values.shape[1]
            actions = []
            for q_value in q_values:
                if np.random.uniform() < self.eps:
                    action = np.random.random_integers(0, nb_actions - 1)
                else:
                    action = np.argmax(q_value)
                actions.append(action)
            return actions

    def get_config(self):
        config = super(EpsGreedyPolicy, self).get_config()
        config['eps'] = self.eps
        return config

class MultiDisPolicy(Policy):
    def select_action(self, q_values):
        # print "distribution", q_values
        while np.sum(q_values) > 1 - 1e-8:
            q_values /= (1 + 1e-5)
        # choice = np.random.multinomial(1, masked_q, size=1).tolist()[0].index(1)
        choice = np.random.choice(range(len(q_values)), p=q_values)
        # print choice, q_values[choice], np.argmax(q_values), np.max(q_values)
        if max(q_values) > 0.2:
            print("max p:", max(q_values), np.argmax(q_values))
        return choice


class MaskedMultiDisPolicy(Policy):
    def select_action(self, q_values):
        # print "distribution", q_values
        masked_q = q_values * self.mask
        masked_q = masked_q/np.sum(masked_q)
        if not np.isfinite(q_values).all() or not np.isfinite(masked_q).all():
            for i in range(len(q_values)):
                if self.mask[i] == 1:
                    masked_q[i] = 1.0
                    # print '0?',i, q_values[i]
                else:
                    masked_q[i] = 0
            # print masked_q
            # assert 0
            masked_q = masked_q / np.sum(masked_q)

        choice = np.random.choice(range(config.doc_max_len), p=masked_q)
        # print choice, q_values[choice], np.argmax(q_values), np.max(q_values)
        if max(q_values) > 0.5:
            print("max p:", max(q_values), np.argmax(q_values))
        return choice