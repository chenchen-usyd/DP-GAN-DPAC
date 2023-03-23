import os
import sys
import numpy as np
from autodp import rdp_acct, rdp_bank

sys.path.insert(0, '../source')
from config import *


def main(config):
    delta = 1e-5
    prob = 1 / args.num_discriminators
    n_steps = args.iterations
    sigma = args.noise_multiplier
    batch_size = args.batchsize
    func = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)

    acct = rdp_acct.anaRDPacct()
    acct.compose_subsampled_mechanism(func, prob, coeff=n_steps * batch_size)
    epsilon = acct.get_eps(delta)
    print("Privacy cost is: epsilon={}, delta={}".format(epsilon, delta))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)