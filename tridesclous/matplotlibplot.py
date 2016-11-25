import numpy as np
import matplotlib.pyplot as plt





def plot_waveforms(waveforms, channels, geometry):
    """
    
    
    """
    fig, ax = plt.subplots()
    for chan in channels:
        x, y = geometry[chan]
        ax.plot([x], [y], marker='o', color='w')
        ax.text(x, y, str(chan))

    wf = waveforms.copy()
    if wf.ndim ==2:
        wf = wf[None, : ,:]
    
    width = wf.shape[1]
    
    delta = 5.
    vect =np.zeros(wf.shape[1]*wf.shape[2])
    
    for i, chan in enumerate(channels):
        x, y = geometry[chan]
        vect[i*width:(i+1)*width] = np.linspace(x-delta, x+delta, num=width)
        wf[:, :, i] += y
    
    wf[:, 0,:] = np.nan
    wf = wf.swapaxes(1,2).reshape(wf.shape[0], -1).T
    
    ax.plot(vect, wf)
    
    return ax
    
    
    
    
    

    
    
    