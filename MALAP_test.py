from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from MALAP import MALA, UnderdampedLangenvin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm
import math

#fix the random seed
NUM_CHAIN=10
np.random.seed(42)
#Choose which sampler to use: 'MALA' or 'UDLG'
sampler ='MALA'#'UDLG'


def find_mix_time(samples, percentile, true_quantile, error_bound=0.1):
    '''
    find mix time.
    percentile: [0, 100]
    '''
    for i in range(10, len(samples)):
        #I think it is better to let the chain run for 10 steps then to
        #decide if the chain is mixing. I don't think it makes sense
        #that the first 5 steps already attain the staitonary distribution
        samples_sub = samples[:i]
        # print 'samples_sub', samples_sub
        emp_quantile = np.percentile(samples_sub, percentile)
        # print 'i', i, emp_quantile
        if abs(emp_quantile - true_quantile) < (error_bound*abs(true_quantile)):
            return i

lambda_max = 4.0
percentile = 75
true_quantile = norm.ppf(0.01*percentile, loc=0, scale=math.sqrt(lambda_max))
print('true_quantile', true_quantile)


def logGaussian(x):
    return 0.5*((1.0/lambda_max)*x[0]**2 + 1.0*x[1]**2)  # + math.log(math.sqrt(2*3.1416)*math.sqrt(lambda_max)) + math.log(math.sqrt(2*3.1416)*math.sqrt(1.0))


mixs = np.zeros([NUM_CHAIN])



for myrun in range(NUM_CHAIN):
    if sampler == 'MALA':
        myMH = MALA(energy_fn=logGaussian, h=0.35355339, mhflag=True)
        init = np.random.multivariate_normal([0,0], [[1,0], [0,1]])
    elif sampler == 'UDLG':
        myMH = UnderdampedLangenvin(energy_fn=logGaussian, L=1.0, h=0.1, mhflag=True)
        # init_pos = np.random.multivariate_normal([0,0], [[1,0], [0,1]])
        init_pos = np.array([.0,.0])
        init_vel = np.array([0.0,0.0])
        init = (init_pos, init_vel)
    mySamples = myMH.sample(lazy_version=True, params_init=init, num_samples=10000, num_thin=1, num_burn=0)
    myMH.acceptance_rate
    # print 'accept_rate', accept_rate
    k_mix = find_mix_time(mySamples[:, 0], percentile, true_quantile, error_bound=0.1)
    # emp_quant = np.percentile(mySamples[:, 0], percentile)
    # print 'emp_quant', emp_quant
    '''
    with PdfPages('samples_hist_run_' + str(myrun) + '.pdf') as pdf:
        n_bins = 100
        plt.hist(mySamples[:, 0], n_bins, histtype='bar')
        plt.ylim(0.0, 400)
        pdf.savefig()
        plt.close()
    '''
    mixs[myrun] = k_mix
    # normal_samples = np.random.normal(0, 2, 10000)
    # k_mix = find_mix_time(normal_samples, percentile, true_quantile, error_bound=0.1)
    print(myrun, k_mix)
print("average mixing time", np.mean(mixs))

'''
with PdfPages('MALAP_samples.pdf') as pdf:
    plt.scatter(mySamples[:, 0], mySamples[:, 1])
    pdf.savefig()
    plt.close()
'''







