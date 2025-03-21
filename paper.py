#!/usr/bin/env python3 3.7.4 project3 env
# -*- coding: utf-8 -*-
"""
Run all scripts necessary for full simulation.
    1) setup parameters
    2) initial training
    3) adaptation experiment
"""
import os
from pathlib import Path
import setup_parameters 
import setup_OFC_network
import adaptation_learning

dir = os.path.expandvars('$FEULNER_FEEDBACK_PLASTICITY_DATA')
dir = Path(dir)
resdir = dir / 'results'
if not os.path.exists(resdir):
    print('Creating results directory',resdir)
    os.mkdir(resdir)

j = 0 # define the seed you want to use
name = 'test_%d'%j
setup_parameters.main(name=name,rand_seed=j)
setup_OFC_network.main(name=name)
adaptation_learning.main(name=name)
