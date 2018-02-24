import numpy as np
import visdom
vis = visdom.Visdom(port=[8097])

def plot_current_errors(epoch, counter_ratio, opt, errors):
        name= 'experiment_name'
        display_id = 1
        if not hasattr('plot_data'):
            plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        plot_data['X'].append(epoch + counter_ratio)
        plot_data['Y'].append([errors[k] for k in plot_data['legend']])
        vis.line(
            X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
            Y=np.array(plot_data['Y']),
            opts={
                'title': name + ' loss over time',
                'legend': plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=display_id)
