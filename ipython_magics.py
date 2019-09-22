# -*- coding: utf-8 -*-
""" Jupyter magics for ipython
Song, Qiang <keeyang@ustc.edu> 2019
"""

from IPython.core import magic_arguments
from IPython.core.magic import (line_magic,
                                cell_magic,
                                line_cell_magic,
                                Magics,
                                magics_class)

@magics_class
class TimeStamp(Magics):
    '''the %%timestamp cell magic print out the starting time, finishing time and
    the job duration when running the cell

    It is similar to the jupyter notebook extension ExecuteTime, but add the
    timestamp directly to cell output instead of cell metadata so the
    information is visible with vanilla jupyter installation

    %%exectime and %%ts are two alias for the %%timestamp alias

    To enable these cell magics, simple add 'import ipython_magics' in your jupyter notebook
    '''
    @staticmethod
    def strftime_timedelta(td, fmt = '%H:%M:%S.%f'):
        days = td.days
        hours, rem = divmod(td.seconds, 3600)
        minutes, rem = divmod(rem, 60)
        seconds = rem

        fmt = (fmt.replace('%D', '{days:d}')
                  .replace('%H', '{hours:02d}')
                  .replace('%M', '{minutes:02d}')
                  .replace('%S', '{seconds:02d}')
                  .replace('%f', '{milli:03.0f}')
                )
        return fmt.format(days = days,
                          hours = hours,
                          minutes = minutes,
                          seconds = seconds,
                          milli = td.microseconds / 1000)

    @cell_magic
    def timestamp(self, line, cell):
        from datetime import datetime, timedelta
        start = datetime.now()
        self.shell.run_cell(cell)
        end = datetime.now()
        duration = end - start
        print(f'start {start:%Y-%m-%d %H:%M:%S}, finish {end:%Y-%m-%d %H:%M:%S}, duration {self.strftime_timedelta(duration)}')

    @cell_magic
    def exectime(self, *args):
        self.timestamp(*args)

    @cell_magic
    def ts(self, *args):
        self.timestamp(*args)

def register_magics(*magics):
    ip = get_ipython()
    for m in magics:
        ip.register_magics(m)

register_magics(TimeStamp)
