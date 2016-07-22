# -*- coding: utf-8 -*-
"""
    CS - ColorSeason
    ~~~~~~
    ColorSeason application for image recognition (classification) by trained neural network model.
    :copyright: (c) 2016 by Anatoly Milkov <anatoly.milko@gmail.com>.
"""
import os
import csv


def get_filelist(filepath):
    if not os.path.isfile(filepath):
        print('ERROR: "%s" not found.' % filepath)
        return []

    filelist = []
    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            filelist.append(row)

    return filelist
