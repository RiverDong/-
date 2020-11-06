# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Utility functions."""

import os
import logging
import inspect
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def logging_config(folder=None, name=None,
                   level=logging.DEBUG,
                   console_level=logging.INFO,
                   no_console=False):
    """ Config the logging.

    Parameters
    ----------
    folder : str or None
    name : str or None
    level : int
    console_level
    no_console: bool
        Whether to disable the console log
    Returns
    -------
    folder : str
        Folder that the logging file will be saved into.
    """
    if name is None:
        name = inspect.stack()[1][1].split('.')[0]
    if folder is None:
        folder = os.path.join(os.getcwd(), name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Remove all the current handlers
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + '.log')
    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)
    if not no_console:
        # Initialize the console logging
        log_console = logging.StreamHandler()
        log_console.setLevel(console_level)
        log_console.setFormatter(formatter)
        logging.root.addHandler(log_console)
    return folder


def save_list_dict(x, name, path):
    # Save a list or dictionary x to a file. The file name and file path are specified by "name" and "path"
    with open(os.path.join(path, name), 'wb') as f:
        pickle.dump(x, f)


def save_df_as_dict(df, name, path):
    # Put a pandas dataframe df to a dictionary and save the distionary to a file. The file name and file path are
    # specified by "name" and "path"
    save_list_dict({name: df}, name, path)


def load_list_dict(name, path):
    # Load a list or dictionary from a file. The file name and file path are specified by "name" and "path"
    with open(os.path.join(path, name), 'rb') as f:
        x = pickle.load(f)
        return x


def load_df_from_dict(name, path):
    # Load a pandas dataframe from a file. The file name and file path are specified by "name" and "path"
    return load_list_dict(name, path)[name]


def get_start_end_index(l, n):
    # Get the starting and ending indices to split a list of length l into n pieces
    assert isinstance(l, int) and isinstance(n, int) and (l >= n >= 1)
    k, m = divmod(l, n)
    return [(i * k + min(i, m), (i + 1) * k + min(i + 1, m)) for i in range(n)]


def get_key_with_value(d, value):
    # Get the key that has a given value from the dictionary d
    key = None
    for k, v in d.items():
        if v == value:
            key = k
            break
    return key


def pickle_dump(data, file_path):
    f_write = open(file_path, 'wb')
    pickle.dump(data, f_write, True)


def pickle_load(file_path):
    f_read = open(file_path, 'rb')
    data = pickle.load(f_read)
    return data


def plot_distr(distr, plot_path, plot_title, thres_interested=tuple()):
    """
    Plot a histogram based on a pandas series (like the one shown below), and save the plot as a pdf

    case1  |  v1
    case2  |  v2
    ...    |  ...
    casen  |  vn

    :param distr: a pandas series containing the values to plot
    :param plot_path: path to save the plot
    :param plot_title: title of the plot
    :param thres_interested: thresholds we are interested in
    """
    with PdfPages(plot_path) as pdf:
        fig, ax = plt.subplots(figsize=(10, 5))
        counts, bins, patches = ax.hist(distr, bins=80, edgecolor='gray', linewidth=0.2)

        ax.set_xticks(bins)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(3)
            tick.label.set_rotation('vertical')

        title = '{0:}: total number of cases: {1:d}, sum: {2:.3f}, mean: {3:.3f}, median: {4:.3f}, min: {5:.3f}, ' \
                'max: {6:.3f}, number of zeros: {7:d}'.format(plot_title, distr.shape[0], distr.sum(), distr.mean(),
                 distr.median(), distr.min(), distr.max(), (distr == 0).sum())
        for thres in thres_interested:
            title += ', number of values no less than {0:.3f}: {1:d}'.format(thres, (distr >= thres).sum())
        ax.set_title(title, fontsize=6)
        plt.ylabel('Number of cases', fontsize=5)

        pdf.savefig(bbox_inches='tight')
        plt.close()