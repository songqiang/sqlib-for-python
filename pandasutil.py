# -*- coding: utf-8 -*-
""" Utilties for working with Pandas
Song Qiang <keeyang@ustc.edu> 2017
"""

from pandas import crosstab
from collections import  deque

def pdfreq(df, tables = None, weight = None, order = 'internal', converters = None, maxlevels = 20):
    """main interface to compute 1D or 2D frequency tables for pandas DataFrame
    using simiar conventions to SAS proc freq.

    Arguments:
    df: dataframe name

    tables: string with list of frequency table requeste; for 1D table, simply
    provide column name; for 2D table, using 'col1 * col2'; if not specifed, the
    function generate 1D frequency table for each column in the given dataframe

    weight: weight variable

    order: 'freq' or 'internal' (default)

    converters: a dictionary with column names as keys and functions as values

    maxlevels: the maximum number of levels for display with multiple level variables

    Example:

        pdfreq(d, tables = 'b * c d', weight = 'a', order = 'freq', converters = {'c': lambda x: x % 2})
    """
    if tables is None:
        tables = list(df.columns)
    else:
        tables = tables.split()
    tables = deque(tables)
    while tables:
        v1 = tables.popleft()
        if tables and tables[0] == '*':
            tables.popleft()
            v2 = tables.popleft()
        else:
            v2 = None

        if v2 is None:
            print('\n1D frequency table {}'.format(v1))
            print('------------------------------------')
        else:
            print('\n2D frequency table {} * {}'.format(v1, v2))
        pdfreq_(v1 = df[v1].apply(converters[v1]) if converters and v1 in converters else df[v1],
                v2 = None if v2 is None else (df[v2].apply(converters[v2]) if converters and v2 in converters else df[v2]),
                weight = None if weight is None else df[weight],
                order = order,
                maxlevels = maxlevels)

def pdfreq_(v1, v2 = None, weight = None, order = 'internal', maxlevels = 20):
    """function to produce actual frequency table; v1 and v2 are expected to be Series"""

    # 2D frequency tables
    if v2 is not None:
        if weight is not None:
            cnt = crosstab(v1, v2, weight, aggfunc = sum, margins = True, dropna = False)
        else:
            cnt = crosstab(v1, v2, margins = True, dropna = False)

        if order == 'freq':
            cnt.iloc[:-1, :] =  cnt.iloc[:-1, :].sort_values('All', ascending = False).values
            cnt = cnt.transpose()
            cnt.iloc[:-1, :] =  cnt.iloc[:-1, :].sort_values('All', ascending = False).values
            cnt = cnt.transpose()

            cnt.fillna(0, inplace=True)

        pct = cnt / cnt.iat[-1, -1] * 100
        col_pct = cnt.divide(cnt.iloc[-1, :]) * 100
        row_pct =cnt.divide(cnt.iloc[:, -1], axis = 0) * 100

        nrow, ncol = cnt.shape

        #  the following displace should also be done with multiindex
        col_width = max(8, *[len(str(x)) for x in cnt.columns])
        col1_width = max(8, *[len(str(x)) for x in cnt.index])

        header_tmplt = '| {:>{w1}} |' + '{:>{w}} |' * ncol
        cnt_tmplt = '| {:>{w1}} |' + '{:{w}.0f} |' * ncol
        pct_tmplt = ('| {:>{w1}} |' + '{:{w}.2f} |' * ncol)
        rowcol_pct_tmplt = ('| {:>{w1}} |' + '{:{w}.2f} |' * (ncol - 1) + '{:{w}} |')

        print('-' * ((col_width + 2) * ncol + col1_width + 4))
        print(header_tmplt.format(cnt.columns.name, *list(cnt.columns ), w = col_width, w1 = col1_width))
        print(header_tmplt.format(cnt.index.name, *([' '] * ncol), w = col_width, w1 = col1_width))
        print('-' * ((col_width + 2) * ncol + col1_width + 4))
        for i in range(nrow):
            print(cnt_tmplt.format(cnt.index[i], *list(cnt.iloc[i, :]), w = col_width, w1 = col1_width))
            print(pct_tmplt.format(' ', *list(pct.iloc[i, :]), w = col_width, w1 = col1_width))
            if i == 0:
                print(rowcol_pct_tmplt.format('Row Pct', *list(row_pct.iloc[i, :-1]), ' ', w = col_width, w1 = col1_width))
                print(rowcol_pct_tmplt.format('Col Pct', *list(col_pct.iloc[i, :-1]), ' ', w = col_width, w1 = col1_width))
            elif i < nrow - 1:
                print(rowcol_pct_tmplt.format(' ', *list(row_pct.iloc[i, :-1]), ' ', w = col_width, w1 = col1_width))
                print(rowcol_pct_tmplt.format(' ', *list(col_pct.iloc[i, :-1]), ' ',  w=col_width, w1 = col1_width))
            print('-' * ((col_width + 2) * ncol + col1_width + 4))

    # 1d frequency table
    else:
        if weight is not None:
            cnt = crosstab(v1, 'cnt', weight, aggfunc = sum, margins = True, dropna = False)
        else:
            cnt = crosstab(v1, 'cnt', margins = True, dropna = False)

        if order == 'freq':
            cnt.iloc[:-1, :] =  cnt.iloc[:-1, :].sort_values('All', ascending = False).values

        del cnt['All']
        cnt.columns.name = ''
        cnt['pct'] = cnt.cnt / cnt.iat[-1, 0] * 100
        with pd.option_context('display.max_rows', maxlevels):
            print(cnt)
