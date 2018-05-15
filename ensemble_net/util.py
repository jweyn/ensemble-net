#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Ensemble-net utilities.
"""

from datetime import datetime


# ==================================================================================================================== #
# General utility functions
# ==================================================================================================================== #

def get_object(module_class):
    """
    Given a string with a module class name, it imports and returns the class.
    This function (c) Tom Keffer, weeWX; modified by Jonathan Weyn.
    """
    # Split the path into its parts
    parts = module_class.split('.')
    # Get the top level module
    module = parts[0]  # '.'.join(parts[:-1])
    # Import the top level module
    mod = __import__(module)
    # Recursively work down from the top level module to the class name.
    # Be prepared to catch an exception if something cannot be found.
    try:
        for part in parts[1:]:
            module = '.'.join([module, part])
            # Import each successive module
            __import__(module)
            mod = getattr(mod, part)
    except ImportError as e:
        # Can't find a recursive module. Give a more informative error message:
        raise ImportError("'%s' raised when searching for %s" % (str(e), module))
    except AttributeError:
        # Can't find the last attribute. Give a more informative error message:
        raise AttributeError("Module '%s' has no attribute '%s' when searching for '%s'" %
                             (mod.__name__, part, module_class))

    return mod


# ==================================================================================================================== #
# Type conversion functions
# ==================================================================================================================== #

def date_to_datetime(date_str):
    """
    Converts a date from string format to datetime object.
    """
    if date_str is None:
        return
    if isinstance(date_str, str):
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M')


def date_to_string(date):
    """
    Converts a date from datetime object to string format.
    """
    if date is None:
        return
    if not isinstance(date, str):
        return datetime.strftime(date, '%Y-%m-%d %H:%M')


def file_date_to_datetime(date_str):
    """
    Converts a string date from config formatting %Y%m%d to a datetime object.
    """
    if date_str is None:
        return
    if isinstance(date_str, str):
        return datetime.strptime(date_str, '%Y%m%d%H')


def date_to_file_date(date):
    """
    Converts a string date from config formatting %Y%m%d to a datetime object.
    """
    if date is None:
        return
    if not isinstance(date, str):
        return datetime.strftime(date, '%Y%m%d%H')
