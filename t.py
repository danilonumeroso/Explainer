from datasets_ import Tox21_AHR

import os

path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'data',
    'Tox21_AHR'
)

dataset = Tox21_AHR(path)
input(dataset[0])
