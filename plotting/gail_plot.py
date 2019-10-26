from plot import *
import os
import matplotlib.pyplot as plt
import argparse

matplotlib.rcParams.update({'font.size': 12})


def gail_plot(gailher, gail, her, x_scale=1., xlim=(0, 1000), pad_val=None):

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    plot_with_std(*gailher, x_scale=x_scale, ax=ax, label='GAIL+HER', color='C0', pad_value=pad_val)
    plot_with_std(*gail, x_scale=x_scale, ax=ax, label='GAIL', color='C7', pad_value=pad_val)
    plot_with_std(*her, x_scale=x_scale, ax=ax, label='HER', color='C4')

    plt.xlim(*xlim)
    plt.xlabel("Number of Environment Steps (x $ 10^3$)")
    plt.ylabel("Percentage of Goals Reached")
    ax.legend(fontsize='large')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, default=None, nargs='?')
    parser.add_argument('env', type=str, default='fourroom', help='fourroom, pointmass-block-pusher, pnp, stacktwo')
    args = parser.parse_args()

    y_key = 'Outer_Success'
    if args.env == 'pointmass-block-pusher':
        y_key += '_0.30'
    elif args.env == 'stacktwo':
        y_key +='_0.80'


    gailher = get_seeds([args.folder], dict(mode='gail_her'), y_key=y_key)
    gail = get_seeds([args.folder], dict(mode='gail'), y_key=y_key)
    her = get_seeds([args.folder], dict(mode='her'), y_key=y_key)

    if args.env == 'fourroom':
        x_scale = 75
        xlim=(0, 3000)
    elif args.env == 'pnp':
        x_scale = 25
        xlim = (0, 10000)
    elif args.env == 'pointmass-block-pusher':
        x_scale = 25
        xlim = (0, 10000)
    elif args.env == 'stacktwo':
        x_scale = 37.5
        xlim = (0, 10000)

    gail_plot(gailher, gail, her, x_scale=x_scale, xlim=xlim)
    plt.savefig(os.path.join('figures', args.env))