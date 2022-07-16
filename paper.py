#!/usr/bin/env python3 3.7.4 project3 env
# -*- coding: utf-8 -*-
"""
Run all scripts necessary to reproduce the paper.
"""
import setup_parameters 
import setup_OFC_network
import adaptation_learning


j = 0 # define the seed you want to use
name = 'test_%d'%j
setup_parameters.main(name=name,rand_seed=j)
setup_OFC_network.main(name=name)
adaptation_learning.main(name=name)
