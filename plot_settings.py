# Import necessary / commonly used modules #################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.stats import binned_statistic_2d

#### some constants ########################################################################################################################
M_sun = 1.989e33                                   # solar mass in g
sec_in_yr = 3.154e7                                # seconds in a year
c = 3e10                                           # speed of light, cm/s

####  Making nice plots ####################################################################################################################

# custom colours 
# (see https://betterfigures.org/2015/06/23/picking-a-colour-scale-for-scientific-graphics/)
col = {'black':          '#000000', 
       'orange':         '#E69F00', 
       'sky blue':       '#56B4E9', 
       'bluish green':   '#009E73', 
       'yellow':         '#F0E442', 
       'blue':           '#0072B2', 
       'vermillion':     '#D55E00', 
       'reddish purple': '#CC79A7'
      }
ls = {'-':   (), 
      '--':  (5,3), 
      ':':   (2,2), 
      '-.':  (5,3,2,3), 
      '-..': (5,3,2,3,2,3)
     }

# define my default cycler
my_colors = [col['blue'], col['vermillion'], col['sky blue'], col['orange'], col['bluish green']]
my_cycler = (cycler(color=my_colors))# + 
             #cycler(linestyle=[ls['-'], ls['--'], ls[':'], ls['-.'], ls['-..']]))

    
mpl_preamble = []
def set_plot_defaults(journal='aa', cycler=my_cycler, dpi=150, use_text_latex=False, return_midwidth=False):
    global mpl_preamble

    # page and column width of journal where plots are supposed to appear
    inches_per_pt = 1.0/72.27
    inches_per_mm = 0.03937008
    if (journal == 'aa'):
        pagewidth = 523.5307*inches_per_pt # A&A
        midwidth = 120.0*inches_per_mm
        columnwidth = 255.76535*inches_per_pt # A&A
    elif (journal == 'mnras'):
        pagewidth = 504.0*inches_per_pt # MNRAS
        midwidth = 120.0*inches_per_mm
        columnwidth = 240.0*inches_per_pt # MNRAS
        
    # enable some standard changes to plots
    #fontsize = 10 , standard
    fontsize = 14
    plt.rc('font', size=fontsize, family='serif')
    plt.rc('axes', prop_cycle=my_cycler, titlesize=fontsize)
    plt.rc('lines', linewidth=1, markersize=4) # a markersize of 4 roughly corresponds to using 's=13' in plt.scatter(...)
    plt.rc('xtick', top=True, direction='out')
    plt.rc('ytick', right=True, direction='out')
    plt.rc('legend', frameon=False, handletextpad=0.2)
    plt.rc('figure', figsize=(columnwidth, columnwidth * 0.6), dpi=dpi) # to enlarge inline displayed figures (saved images are vector, so this has no change)
    plt.rc('figure.constrained_layout', use=True)

#    if (use_text_latex):
#        # at the moment, the 'text.latex' is experimental and it can cause problems on some systems
#        plt.rc('text', usetex=True)

#        if (not mpl_preamble):
#            mpl_preamble = plt.rcParams.get('text.latex.preamble', [])
#            if (isinstance(mpl_preamble, str)):
#                mpl_preamble = [mpl_preamble]
#            mpl_preamble = mpl_preamble +   [
 #                                               r'\usepackage{amsmath} \usepackage{txfonts}'
 #                                           ]

#        plt.rc('text.latex', preamble=mpl_preamble)
                                    
    if (return_midwidth):
        return pagewidth, columnwidth, midwidth
    else:
        return pagewidth, columnwidth

