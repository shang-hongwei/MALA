"""
This is a Metropolis Adjusted Langevin Algorithm (MALA) proposal.
Author:
    Ilias Bilionis
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

__all__ = ['MALA']


import numpy as np
# from scipy.stats import norm
# from nlu import GradProposal
# from nlu import SingleParameterTunableProposalConcept
from numpy import linalg as LA
from autograd import grad
import abc
from scipy.stats import multivariate_normal

class AbstractSampler(object):
    """ Abstract Langevin Monte Carlo sampler base class. """
    def __init__(self, energy_fn, h=0.01, mhflag=True):
        """
        Abstract Langevin sampler constructor

        Parameters
        ----------
        energy_func : function(vector, dictionary]) -> scalar
            Function which returns energy (marginal negative log density) of a
            position state.

        """
        self.energy_fn = energy_fn
        self.h = h
        self.ifMH = mhflag

    @abc.abstractmethod
    def proposal_sample(self, old_params, old_grad_params):
        """
        proposal a new sample
        :param old_params: vectors
        :param old_grad_params: gradient of energy funciton at old params
        :return: new params
        """
        pass
    @abc.abstractclassmethod
    def energy_grad_fn(self,params):
        """
        compute the gradient of energy function at params
        :param params: vector
        :return: gradient of energy function, vector
        """
        pass

    @property
    def acceptance_rate(self):
        """
        Get the acceptance rate.
        """
        accept_rate = self.accepted / self.count
        print('accept_rate', accept_rate)

    @abc.abstractclassmethod
    def state_fn(self, x, z, grads_z):
        """
        Used in MH accept-reject
        :param x: state x
        :param z: state z
        :param grads_z: gradient z
        :return: log(pi(z)*P(z,x))
        """
        pass
    @abc.abstractclassmethod
    def sample(self, lazy_version, params_init, num_samples, num_thin=1, num_burn=0,
               init_model_state=None, init_proposal_state=None,
               start_tuning_after=0, stop_tuning_after=None,
               tuning_frequency=1000,
               verbose=False):
        """
        given the inital params, return a list of samples

        :param lazy_version: bool, if apply lazy version of the chain
        :param params_init: np.array, initial point of the chain
        :param num_samples: int, number of samples
        :param num_thin: the gap between two samples
        :param num_burn: the number of samples need to burn
        :param init_model_state: inital state
        :param init_proposal_state:
        :param start_tuning_after:
        :param stop_tuning_after:
        :param tuning_frequency:
        :param verbose:
        :return:
        np.array, a list of samples
        """
        pass

class MALA(AbstractSampler):

    """
    A MALA proposal.
    :param h:      The time step. The larger you pick it, the bigger the steps
                    you make and the acceptance rate will go down.
    :type h:       float

    The rest of the keyword arguments is what you would find in:
        + :class:`pymcmc.GradProposal`
        + :class:`pymcmc.SingleParameterTunableProposal`
    """
    def __init__(self, energy_fn, h=0.01, mhflag=True):
        self.energy_fn = energy_fn
        self.h = h
        self.ifMH = mhflag


    def proposal_sample(self, old_params, old_grad_params):
        noise = np.random.randn(old_params.shape[0])
        return (old_params - self.h * old_grad_params +
                (2.0*self.h)**0.5 * noise)

    def energy_grad_fn(self, params):
        # if energy_grad is None:
        e_grad = grad(self.energy_fn)
        # force energy gradient to ignore cached results if using Autograd
        return e_grad(params)

    # @property
    # def acceptance_rate(self):
    #     """
    #     Get the acceptance rate.
    #     """
    #     print 'accept', self.accepted
        # print 'count', self.count
        # accept_rate = self.accepted / self.count
        # print('accept_rate', accept_rate)
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
                log_u = np.log(np.random.uniform())
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



class UnderdampedLangenvin(AbstractSampler):
    """
    implemnetation of UnderdampedLangenvin sampler
    NOTICE: params, a tuple, consists two parts, the postion and velocity
    """
    def __init__(self, energy_fn, L,  h=0.01, mhflag=True):
        """
        :param L: the largest eigenvalue of Hessian, used in the scheme
        """
        super(UnderdampedLangenvin, self).__init__(energy_fn=energy_fn, h=h, mhflag=mhflag)
        self.L = L



    def energy_grad_fn(self,params):
        """
        compute the gradient of energy function at params
        :param params: position and velocity
        :return: gradient of energy function, vector
        """
        pos,vel = params
        e_grad = grad(self.energy_fn)
        return e_grad(pos)






    def proposal_sample(self, old_params, old_pos_grad_params):
        """
        old_params: tuple
        old_pos_grad_params: the gradient of energy function at pos
        """

        pos, vel = old_params
        mean_v = np.exp(-2*self.h)*vel - 1.0/(2*self.L)*(1-np.exp(-2*self.h))*old_pos_grad_params
        mean_x = pos + 0.5*(1-np.exp(-2*self.h))*vel-1.0/(2*self.L)*(self.h-0.5*(1-np.exp(-2*self.h)))*old_pos_grad_params
        uleft = np.eye(len(pos))*1.0/self.L*(self.h - 0.25 * np.exp(-4*self.h)-0.75 + np.exp(-2*self.h))
        uright = np.eye(len(vel))*1.0/(2*self.L)*(1+np.exp(-4*self.h)-2*np.exp(-2*self.h))
        bright = np.eye(len(vel))*1.0/self.L * (1-np.exp(-4*self.h))
        cov = np.block([
            [uleft, uright],
            [uright, bright]
        ])
        new_point = np.random.multivariate_normal(np.concatenate([mean_x,mean_v]), cov)
        new_pos = new_point[:len(pos)]
        new_vel = new_point[len(pos):]
        return (new_pos,new_vel)


    def state_fn(self, x, z, grads_z):
        """
        Now the implementation is only in position variable.
        :param x:
        :param z:
        :param grads_z:
        :return:
        """
        pos_z, vel_z = z
        pos_x, vel_x = x
        mean_v = np.exp(-2 * self.h) * vel_z - 1.0/(2 * self.L) * (1-np.exp(-2 * self.h)) * grads_z
        mean_x = pos_z + 0.5 * (1-np.exp(-2 * self.h)) * vel_z-1.0/(2 * self.L) \
                 * (self.h-0.5 * (1-np.exp(-2 * self.h))) * grads_z
        up_left = np.eye(len(pos_z)) * 1.0/self.L * (self.h - 0.25 * np.exp(-4 * self.h)-0.75 + np.exp(-2 * self.h))
        up_right = np.eye(len(vel_z)) * 1.0/(2 * self.L) * (1+np.exp(-4 * self.h)-2 * np.exp(-2 * self.h))
        bm_right = np.eye(len(vel_z)) * 1.0/self.L * (1-np.exp(-4 * self.h))

        cov = np.block([
            [up_left, up_right],
            [up_right, bm_right]
        ])
        log_p_zx = multivariate_normal.logpdf(np.concatenate([pos_x,vel_x]), mean=np.concatenate([mean_x,mean_v]),cov=cov)
        log_pi_z = -1.0*self.energy_fn(pos_z)-self.L/2.0*LA.norm(vel_z)**2
        # if only consider pos variable
        # log_pi_z = -1.0*self.energy_fn(pos_z)
        # log_p_zx = multivariate_normal.logpdf(pos_x, mean=mean_x, cov=up_left)


        return log_pi_z + log_p_zx



    def sample(self, lazy_version, params_init, num_samples, num_thin=1, num_burn=0,
               init_model_state=None, init_proposal_state=None,
               start_tuning_after=0, stop_tuning_after=None,
               tuning_frequency=1000,
               verbose=False):
        num_dim = params_init[0].shape[0]
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
                log_u = np.log(np.random.uniform())
                if log_u <= log_p or not self.ifMH:
                    old_params = new_params
                    old_grad_params = new_grad_params
                    self.accepted += 1
                self.count += 1
            # Output
            if (i >= num_burn) and ((i - num_burn) % num_thin == 0):
                all_samples[sample_add, :] = old_params[0]
                sample_add = sample_add+1
        return all_samples
