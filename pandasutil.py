# -*- coding: utf-8 -*-
""" Utilties for working with Pandas
Song Qiang <keeyang@ustc.edu> 2017
"""
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


from collections import  deque
from pprint import pprint
import functools

from pandas import crosstab
import pandas as pd
import numpy as np

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

def pdcompare(df1, df2, keys = None):
    """compare two pandas dataframes: list columns common to both dataframe or
    specific to any individual dataset.

    For common numberic columns, calculate summary statistics for differences

    For common string columns, calculate proportion of matching rows for each
    columns

    Usage:
    pdcompare(df1, df2)  # join by index, typically for two equal length
    dataframes

    pdcompare(df1, df2, keys = ['id']) # join by key columns
    """

    cols1 = df1.columns.to_series().to_frame('col')
    cols2 = df2.columns.to_series().to_frame('col')

    cols = cols1.merge(cols2, on = 'col', how = 'outer', indicator = True)
    print('\nColumn summary')
    print('-' * 100)
    pprint(cols._merge.value_counts())
    if cols._merge.eq('both').sum() > 0:
        print('\nShared columns')
        pprint(cols[cols._merge.eq('both')])
    if cols._merge.eq('left_only').sum() > 0:
        print('\nLeft specific columns')
        pprint(cols[cols._merge.eq('left_only')])
    if cols._merge.eq('right_only').sum() > 0:
        print('\nRight specific columns')
        pprint(cols[cols._merge.eq('right_only')])

    if keys is None:
        merged = df1.merge(df2, left_index = True, right_index = True,
                           how = 'outer', suffixes=('_left', '_right'),
                           indicator = True)
    else:
        merged = df1.merge(df2, on = keys,
                           how = 'outer', suffixes=('_left', '_right'),
                           indicator = True)

    print('\nRow summary')
    print('-' * 100)
    pprint(merged._merge.value_counts())

    merged = merged[merged._merge.eq('both')]

    num_cols = set(df1.select_dtypes(include = [np.number]).columns) \
                .intersection(set(cols.loc[cols._merge.eq('both'), 'col']))
    if keys is not None:
        num_cols = num_cols.difference(set(keys))

    diff_list = []
    for c in list(num_cols):
        diff = (merged[c + '_left'] - merged[c + '_right']).abs()
        rel_diff = diff / (merged[c + '_left'].abs() + merged[c + '_right'].abs()) * 2

        diff = diff.describe(percentiles = [0.5, 0.75, 0.9, 0.95, 0.99])
        rel_diff = rel_diff.describe(percentiles = [0.5, 0.75, 0.9, 0.95, 0.99])

        diff.name = c + '_absolute'
        rel_diff.name = c + '_relative'

        diff_list.append(diff)
        diff_list.append(rel_diff)

    diff_df_num = pd.concat(diff_list, axis = 1)
    mindex = pd.MultiIndex.from_tuples([(t[0], t[2]) for t in
                                        (c.rpartition('_') for c in diff_df_num.columns)],
                                       names = ('var', 'diff_type'))
    diff_df_num.columns = mindex
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                           "display.width", 200):
        print('\nNumeric variables')
        print('-' * 100)
        pprint(diff_df_num.transpose())


    cat_cols = set(df1.select_dtypes(exclude = [np.number]).columns) \
                .intersection(set(cols.loc[cols._merge.eq('both'), 'col']))
    if keys is not None:
        cat_cols = cat_cols.difference(set(keys))

    diff_list = []
    for c in list(cat_cols):
       diff_prop = merged[c + '_left'].eq(merged[c + '_right']).value_counts(normalize=True)
       diff_prop.name = c
       diff_list.append(diff_prop)

    diff_df_cat = pd.concat(diff_list, axis = 1)
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                           "display.width", 200):
        print('\nCategorical variables')
        print('-' * 100)
        pprint(diff_df_cat.transpose())

    return diff_df_num, diff_df_cat

def _decr_read_csv(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        logger.info('reading file {}'.format(args[0]))
        df = f(*args, **kwargs)
        logger.info('dataframe has {} rows and {} columns'.format(*df.shape))
        return df
    return decorated
read_csv = _decr_read_csv(pd.read_csv)

def freq(v):
    '''generate SAS style 1D frequency table
    '''
    
    cnt = v.value_counts()
    r = pd.DataFrame({'cnt': cnt, 'pct': cnt / cnt.sum()})
    r['pct_cum'] = r.pct.cumsum()
    return r

def summary(df, outfile = 'summary.csv'):
    from csv import DictWriter
    from pprint import pformat
    with open(outfile, 'w') as f:
        fieldnames = ['field', 'type', 'count', 'missing', 'levels', 'freq', 'stat']
        writer = DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()

        results = []
        for c in df.columns:
            r = {'field': c}
            r['type'] = df[c].dtype
            r['missing'] = df[c].isna().mean()

            cnt = df[c].value_counts()
            freq = pd.DataFrame({
                'cnt': cnt,
                'pct': cnt / cnt.sum(),
            })
            freq['pct_sum'] = freq.pct.cumsum()
            r['count'] = freq.cnt.sum()
            r['levels'] = freq.shape[0]
            if r['levels'] <= 20:
                freq_str = pformat(freq)
            else:
                freq_str = f'{freq.head(15)}\n...\n{freq.tail(5)}'
            r['freq'] = freq_str
            if np.issubdtype(df[c].dtype, np.number):
                r['stat'] = pformat(df[c].describe())
            else:
                samples = df[c].sample(min(5, df.shape[0]))
                r['stat'] = '\n'.join(map(str, samples))

            writer.writerow(r)
            results.append(r)

        return results

