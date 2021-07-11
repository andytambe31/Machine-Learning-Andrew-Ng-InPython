# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 23:30:35 2021

@author: user
"""

import pandas as pd

population_dict = { 'California': 38332521,
                    'Texas': 26448193,
                    'New York': 19651127,
                    'Florida': 19552860,
                    'Illinois': 12882135
                    }

population = pd.Series(population_dict)

