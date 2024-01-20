import random
import numpy as np
import matplotlib 
from matplotlib import pyplot as plt, colors
import pickle
import os
import pylab
#matplotlib.rcParams['backend'] = "GTK4Agg"

def visualize_dataset_for_bucketing_stats(text_pairs):

    x = np.array([a[0].count(' ') for a in text_pairs])
    y = np.array([a[1].count(' ') for a in text_pairs])

    plt.figure()  
    #plt.axis([0, 200, 0, 120000])  

    plt.subplots_adjust(hspace=.4)
    plt.hist(x, bins=10, alpha=0.5, label='English sentences')
    plt.hist(y, bins=10, alpha=0.5, label='Tamil sentences')
    plt.title('Overlapping')  
    plt.xlabel('No. of words')  
    plt.ylabel('No of sentences')  
    plt.legend() 
    pylab.savefig('utils/Histogram1.png', bbox_inches='tight')

    common_params = dict(bins=10, 
                         range=(0, 175))
    
    plt.clf()
    plt.title('Skinny shift')
    plt.hist((x, y), label=["English sentences", "Tamil sentences"])
    plt.legend(loc='upper right')
    common_params['histtype'] = 'step'
    plt.xlabel('No. of words')  
    plt.ylabel('No of sentences') 
    plt.legend() 
    pylab.savefig('utils/Histogram2.png', bbox_inches='tight')
    
    plt.clf()
    fig, ax = plt.subplots()
    hh = ax.hist2d(x, y, bins=10, range=[[0,70],[0,50]])
    plt.xlabel("# English words per sentence")
    plt.ylabel("# Tamil words per sentence")
    fig.colorbar(hh[3], ax=ax)
    pylab.savefig('utils/Correlation.png', bbox_inches='tight')

