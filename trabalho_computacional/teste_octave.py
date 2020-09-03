from oct2py import octave
import numpy as np
import pprint

octave.eval('pkg load statistics')
hvi = octave.EvalParetoApp('ExemploResultado.csv')
