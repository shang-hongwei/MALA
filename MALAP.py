"""
This is a Metropolis Adjusted Langevin Algorithm (MALA) proposal.
Author:
    Ilias Bilionis
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

__all__ = ['MALAProposal']


import numpy as np
# from scipy.stats import norm
# from nlu import GradProposal
# from nlu import SingleParameterTunableProposalConcept
from numpy import linalg as LA
import math
from autograd import grad


class MALAProposal(object):

    """
    A MALA proposal.
    :param h:      The time step. The larger you pick it, the bigger the steps
                    you make and the acceptance rate will go down.
    :type h:       float

    The rest of the keyword arguments is what you would find in:
        + :class:`pymcmc.GradProposal`
        + :class:`pymcmc.SingleParameterTunableProposal`
    """
    def __init__(self, energy_fn, h=0.01):
        self.energy_fn = energy_fn
        self.h = h


    def proposal_sample(self, old_params, old_grad_params):
        noise = np.random.randn(old_params.shape[0])
        return (old_params - self.h * old_grad_params +
                (2.0*self.h)**0.5 * noise)

    def energy_grad_fn(self, params):
        # if energy_grad is None:
        e_grad = grad(self.energy_fn)
        # force energy gradient to ignore cached results if using Autograd
        return e_grad(params)






class MetropolisHastings(MALAProposal):
    def __init__(self, energy_fn, h=0.01, mhflag=True):
        """
        :param energy_fn:
        :param h:
        :param mhflag: bool, if use M-H adjustment
        """
        super(MetropolisHastings,self).__init__(energy_fn,h)
        self.ifMH = mhflag


    # @property
    def acceptance_rate(self):
        """
        Get the acceptance rate.
        """
        # print 'accept', self.accepted
        # print 'count', self.count
        accept_rate = self.accepted / self.count
        print('accept_rate', accept_rate)
        # return accept_rate

    def state_fn(self, x, z, grads_z):
        """
        :param x: state x
        :param z: state z
        :param grads_z: gradient z
        :return: log(pi(z)*P(z,x))
        """

        return -1.0*self.energy_fn(z) - (LA.norm(x - z + self.h * grads_z))**2 / (4*self.h)

    def sample(self, lazy_version, params_init, num_samples, num_thin=1, num_burn=0,
               init_model_state=None, init_proposal_state=None,
               start_tuning_after=0, stop_tuning_after=None,
               tuning_frequency=1000,
               verbose=False):
        num_dim = params_init.shape[0]
        # print(num_samples.dtype)
        all_samples = np.empty([(num_samples - num_burn)//num_thin,  num_dim])
        # Initialize counters
        self.accepted = 0.
        self.count = 0.
        sample_add = 0
        # Initialize the database
        # Start sampling
        old_params = params_init
        old_grad_params = self.energy_grad_fn(old_params)
        # print 'old_grad_params', type(old_grad_params)
        for i in range(num_samples):
            if (not lazy_version) or (lazy_version and (np.random.uniform() < 0.5)):
                # MCMC Step
                new_params = self.proposal_sample(old_params, old_grad_params)
                new_grad_params = self.energy_grad_fn(new_params)
                ## Metropolis-Hasting
                log_p_numerator = self.state_fn(old_params, new_params, new_grad_params)
                log_p_denom = self.state_fn(new_params, old_params, old_grad_params)
                # log_p_numerator = self.state_fn(new_params, old_params, new_grad_params)
                # log_p_denom = self.state_fn(old_params, new_params, old_grad_params)
                log_p = min(0.0, log_p_numerator - log_p_denom)
                log_u = math.log(np.random.uniform())
                if log_u <= log_p or not self.ifMH:
                    old_params = new_params
                    old_grad_params = new_grad_params
                    self.accepted += 1
                self.count += 1
            # Output
            if (i >= num_burn) and ((i - num_burn) % num_thin == 0):
                all_samples[sample_add, :] = old_params
                sample_add = sample_add+1
        return all_samples