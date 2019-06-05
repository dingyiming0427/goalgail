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
    parser.add_argument('env', type=str, default='fourroom')
    args = parser.parse_args()

    gailher = get_seeds([args.folder], dict(mode='gail_her'))
    gail = get_seeds([args.folder], dict(mode='gail'))
    her = get_seeds([args.folder], dict(mode='her'))

    gail_plot(gailher, gail, her, x_scale=75 if args.env == 'fourroom' else 25, xlim=(0, 1500) if args.env=='fourroom' else (0, 10000))
    plt.savefig(os.path.join('figures', args.env))