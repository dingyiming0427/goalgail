import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
from rllab.misc import ext
import os
import base64
import pickle
import json
import itertools

from rllab.viskit.core import Selector

def plot_with_std(x, ys, x_shift = 0, x_scale = 1, color='b', ax=plt, label='', pad_value = None):
    """
    ys: data for different seeds
    x: x axis
    x * x_scale + x_shift will be plotted
    """
    length = min([len(y) for y in ys] + [len(x)])
    print("data of length %d" % length)
    x = x[:length]
    ys = [y[:length] for y in ys]
    x = x * x_scale + x_shift
    if pad_value is not None:
        x = np.concatenate([[pad_value] , x])
        ys = [np.concatenate([[pad_value], y]) for y in ys]
    avg = np.mean(np.array(ys), axis=0)
    stdd = np.std(np.array(ys), axis=0)
    ax.plot(x, avg, color=color, label=label)
    plt.fill_between(x, avg - stdd, avg + stdd, facecolor=color, alpha=0.3)


def load_exps_data(exp_folder_paths, disable_variant=False, ignore_missing_keys=True):
    exps = []
    for exp_folder_path in exp_folder_paths:
        exps += [x[0] for x in os.walk(exp_folder_path, followlinks=True)]
    print("finished walking exp folders")
    exps_data = []
    for exp in exps:
        try:
            exp_path = exp
            params_json_path = os.path.join(exp_path, "params.json")
            variant_json_path = os.path.join(exp_path, "variant.json")
            progress_csv_path = os.path.join(exp_path, "progress.csv")
            progress = load_progress(progress_csv_path)
            if disable_variant:
                params = load_params(params_json_path)
            else:
                try:
                    params = load_params(variant_json_path)
                except IOError:
                    params = load_params(params_json_path)
            exps_data.append(ext.AttrDict(
                progress=progress, params=params, flat_params=flatten_dict(params)))
        except IOError as e:
            # print(e)
            pass

    # a dictionary of all keys and types of values
    all_keys = dict()
    for data in exps_data:
        for key in data.flat_params.keys():
            if key not in all_keys:
                all_keys[key] = type(data.flat_params[key])

    # if any data does not have some key, specify the value of it
    if not ignore_missing_keys:
        default_values = dict()
        for data in exps_data:
            for key in sorted(all_keys.keys()):
                if key not in data.flat_params:
                    if key not in default_values:
                        default = input("Please specify the default value of \033[93m %s \033[0m: " % (key))
                        try:
                            if all_keys[key].__name__ == 'NoneType':
                                default = None
                            elif all_keys[key].__name__ == 'bool':
                                try:
                                    default = eval(default)
                                except:
                                    default = False
                            else:
                                default = all_keys[key](default)
                        except ValueError:
                            print("Warning: cannot cast %s to %s" % (default, all_keys[key]))
                        default_values[key] = default
                    data.flat_params[key] = default_values[key]

    return exps_data

def load_progress(progress_csv_path):
    # print("Reading %s" % progress_csv_path)
    entries = dict()
    with open(progress_csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for k, v in row.items():
                if k not in entries:
                    entries[k] = []
                try:
                    entries[k].append(float(v))
                except:
                    entries[k].append(0.)
    entries = dict([(k, np.array(v)) for k, v in entries.items()])
    return entries

def load_params(params_json_path):
    with open(params_json_path, 'r') as f:
        data = json.loads(f.read())
        if "args_data" in data:
            del data["args_data"]
        if "exp_name" not in data:
            data["exp_name"] = params_json_path.split("/")[-2]
    return data
    
def flatten_dict(d):
    flat_params = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            v = flatten_dict(v)
            for subk, subv in flatten_dict(v).items():
                flat_params[k + "." + subk] = subv
        else:
            flat_params[k] = v
    return flat_params


def get_selector(selector, filters):
    ret_selector = selector
    for k, v in filters.items():
        ret_selector = ret_selector.where(k, v)
    return ret_selector


def get_seeds(folder, filters, x_key='Outer_iter', y_key='Outer_Success'):
    exp_data = load_exps_data(folder)
    selector = Selector(exp_data)
    selector = get_selector(selector, filters)
    raw_data = selector.extract()
    progress = [data['progress'] for data in raw_data]
    x = [p[x_key] for p in progress][0]
    ys = [p[y_key] for p in progress]
    print("%d seeds" % len(ys))
    return x, ys


