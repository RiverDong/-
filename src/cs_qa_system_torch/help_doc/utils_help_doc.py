import boto3
import difflib
import dill
import functools
import json
import logging
import numpy as np
import os
import pandas as pd
import pathlib
import pprint
import psutil
import pytz
import re
import shutil
import time
import warnings
from datetime import datetime
from logging import Handler, Logger, LogRecord
from multiprocessing_on_dill import Pool
from tqdm.auto import tqdm
from typing import Callable, List, Tuple, Union
from tzlocal import get_localzone

pprint.sorted = lambda x, key=None: x


def is_iterable(x) -> bool:
    try:
        _ = iter(x)
    except TypeError:
        return False
    else:
        return True


def non_element_wise_isnull(x) -> bool:
    # When pd.isnull(x) is an iterable, it will return False, instead of returning the result of an element-wise pd.isnull
    # (when pd.isnull(x) is an iterable, then x is definitely not an NA value, such as None or np.nan)
    result = pd.isnull(x)
    return False if is_iterable(result) else result


def non_element_wise_eq(x, y) -> bool:
    # When (x == y) is an iterable, it will return all(x == y), instead of returning the result of an element-wise comparison
    result = (x == y)
    return all(result) if is_iterable(result) else result


def hash_str(x: str, digit: int = 8) -> int:
    return abs(hash(x)) % (10 ** digit)


def is_number(x) -> bool:
    return isinstance(x, (int, float, np.number))


def unique_retain_order(x: list) -> list:
    # Return the unique elements of a list while retaining its original order
    seen = set()
    result = list()
    for elem in x:
        if elem in seen:
            continue
        result.append(elem)
        seen.add(elem)
    return result


def current_time(time_zone: str = 'US/Pacific', show_microsecond: bool = False) -> str:
    return datetime.now(tz=pytz.timezone(time_zone)).strftime('%Y-%m-%d-%H-%M-%S' + '-%f' * show_microsecond + '-%Z')


def str_to_datetime(x: str, time_zone: str = 'US/Pacific') -> datetime:
    # Convert the time string generated from "current_time" to a datetime object
    # The time_zone and time_format of these two functions must be consistent
    if re.fullmatch(r'(?as)([0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{6}-[A-Z]{3})', x):
        time_format = '%Y-%m-%d-%H-%M-%S-%f'
    elif re.fullmatch(r'(?as)([0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[A-Z]{3})', x):
        time_format = '%Y-%m-%d-%H-%M-%S'
    else:
        raise NotImplementedError
    return datetime.strptime('-'.join(x.split('-')[:-1]), time_format).replace(tzinfo=pytz.timezone(time_zone))


def has_word_char_str(x: str) -> bool:
    return isinstance(x, str) and (re.fullmatch(r'(?us)(\W*)', x) is None)

########################################################################################################################
def relpath(path1: str, path2: str) -> str:
    result = os.path.relpath(path1, path2)
    return '' if result == '.' else result


def file_ext_from_path(path: str) -> str:
    return pathlib.Path(path).suffix


def save_file(x, path: str) -> None:
    ext = file_ext_from_path(path)
    if ext in {'.csv'}:
        assert isinstance(x, pd.DataFrame)
        x.to_csv(path, sep=',', index=False)
    elif ext in {'.tsv'}:
        assert isinstance(x, pd.DataFrame)
        x.to_csv(path, sep='\t', index=False)
    elif ext in {'.xlsx', '.xls'}:
        assert isinstance(x, pd.DataFrame)
        warning_msg = ''
        for col in x.columns:
            n_truncated = sum(len(str(cell)) > 32767 for cell in x[col].tolist())
            warning_msg += '{0:d} cells in column "{1:}" truncated; '.format(n_truncated, col) * (n_truncated > 0)
        if warning_msg != '':
            warnings.warn(warning_msg)
        x.to_excel(path, index=False)
    elif ext in {'.json'}:
        if isinstance(x, pd.DataFrame):
            x.reset_index(drop=True).to_json(path, orient='records')
        else:
            assert isinstance(x, dict)
            with open(path, 'w') as f:
                json.dump(x, f)
    elif ext in {''}:
        assert isinstance(x, (dict, list, set))
        with open(path, 'wb') as f:
            dill.dump(x, f)
    else:
        raise NotImplementedError


def load_file(path: str, json_type: str = 'dict', df_col_eval: tuple = tuple()):
    ext = file_ext_from_path(path)
    if ext in {'.csv'}:
        x = pd.read_csv(path, sep=',', index_col=False)
    elif ext in {'.tsv'}:
        x = pd.read_csv(path, sep='\t', index_col=False)
    elif ext in {'.xlsx', '.xls'}:
        x = pd.read_excel(path, index_col=None)
    elif ext in {'.json'}:
        if json_type == 'dataframe':
            x = pd.read_json(path, orient='records', typ='frame')
        else:
            with open(path, 'r') as f:
                x = json.load(f)
    elif ext in {''}:
        with open(path, 'rb') as f:
            x = dill.load(f)
    else:
        raise NotImplementedError

    if isinstance(x, pd.DataFrame):
        for col in x.columns:
            if x[col].dtype == object or x[col].isnull().all():
                x[col] = x[col].fillna('')
                if col in df_col_eval:
                    x[col] = x[col].map(lambda z: eval(z) if z != '' else '')

    return x


def make_dir(path: str, method: str = 'raise'):
    if os.path.exists(path):
        if method == 'rename':
            time.sleep(1)
            os.rename(path.rstrip('/'), path.rstrip('/') + '_' + current_time())
            os.makedirs(path, exist_ok=False)
        elif method == 'overwrite':
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=False)
        elif method == 'raise':
            raise FileExistsError
        else:
            assert method == 'keep'
    else:
        os.makedirs(path, exist_ok=False)


def get_file_in_dir(root_dir: str, start_with: Union[str, Tuple[str]] = None, end_with: Union[str, Tuple[str]] = None,
                    not_start_with: Union[str, Tuple[str]] = None, not_end_with: Union[str, Tuple[str]] = None,
                    return_full_path: bool = False, keep_ext: bool = False, recursively: bool = True) -> Union[List[str], List[Tuple[str, str]]]:
    """
    Get file names of all the files in a given directory, where you can also
    (1) add some requirements to the file names
    (2) decide whether you want to get full paths of all the files
    (3) decide whether you want file names to have file extensions
    (4) decide whether you want to do this recursively for all the sub-directories

    :param root_dir: path of the root directory
    :param start_with: a string or a tuple of strings that the file name must start with
    :param end_with: a string or a tuple of strings that the file name must end with
    :param not_start_with: a string or a tuple of strings that the file name must not start with
    :param not_end_with: a string or a tuple of strings that the file name must not end with
    :param return_full_path: whether you want to get full paths of all the files
    :param keep_ext: whether you want file names to have file extensions
    :param recursively: whether you want to do this recursively for all the sub-directories
    :return: a list of file names, or a list of (file name, file directory) tuples
    """
    all_file_dir = []
    for d, _, files in os.walk(root_dir):
        all_file_dir.extend([(f, d) for f in files])
        if not recursively:
            break

    all_file_eligible = {f: True for f, _ in all_file_dir}
    for f in all_file_eligible:
        is_start_with = f.startswith(start_with) if start_with else True
        is_end_with = f.endswith(end_with) if end_with else True
        is_not_start_with = (not f.startswith(not_start_with)) if not_start_with else True
        is_not_end_with = (not f.endswith(not_end_with)) if not_end_with else True
        if not (is_start_with and is_end_with and is_not_start_with and is_not_end_with):
            all_file_eligible[f] = False

    if not return_full_path:
        if not keep_ext:
            return [os.path.splitext(f)[0] for f, _ in all_file_dir if all_file_eligible[f]]
        else:
            return [f for f, _ in all_file_dir if all_file_eligible[f]]
    else:
        if not keep_ext:
            return [(os.path.splitext(f)[0], d) for f, d in all_file_dir if all_file_eligible[f]]
        else:
            return [(f, d) for f, d in all_file_dir if all_file_eligible[f]]

########################################################################################################################
def copy_s3_dir(old_s3_dir: str, new_s3_dir: str, s3_bucket_name: str,
                s3_region: str = None, aws_access_key_id: str = None, aws_secret_access_key: str = None):

    assert not old_s3_dir.startswith('/') and not new_s3_dir.startswith('/')

    s3 = boto3.resource('s3', region_name=s3_region,
                        aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    bucket = s3.Bucket(s3_bucket_name)

    for s3_object in bucket.objects.all():
        s3_file_dir, s3_file_name = os.path.split(s3_object.key)
        if s3_object.key.endswith('/') or (not (s3_file_dir.rstrip('/') + '/').startswith(old_s3_dir.rstrip('/') + '/')):
            # We only need to copy all the objects, so we skip all the S3 "folders" (the key of an S3 "folder" always ends with '/')
            continue
        old = s3_object.key
        new = os.path.join(new_s3_dir, relpath(old, old_s3_dir))
        s3.Object(s3_bucket_name, new).copy_from(CopySource=os.path.join(s3_bucket_name, old))
        # s3.Object(s3_bucket_name, old).delete()


def s3_to_local(local_root_dir: str, s3_bucket_name: str, s3_dir: str = '', s3_file_name_prefix: str = '',
                s3_region: str = None, aws_access_key_id: str = None, aws_secret_access_key: str = None) -> str:
    """
    Download all the files with a particular prefix (s3_file_name_prefix) recursively from an S3 directory (s3_dir)
    in an S3 bucket (s3_bucket_name) to a local directory (local_root_dir)

    :param local_root_dir: local root directory
    :param s3_bucket_name: bucket name of the S3 bucket
    :param s3_dir: directory in the S3 bucket we want to donwload files from
                   (by default, we download the files that are in the bucket but not in any sub-directories of the bucket)
    :param s3_file_name_prefix: we only download the files whose file names start with s3_file_name_prefix
    :param s3_region: region of the S3 bucket
    :param aws_access_key_id: aws access key id of the S3 bucket. When it is None, boto3 will search through a list of
                              possible locations (environment variables, shared credential file, ...) in a given order,
                              and stop as soon as it finds this credential
    :param aws_secret_access_key: aws secret access key of the S3 bucket. When it is None, boto3 will search through a list of
                              possible locations (environment variables, shared credential file, ...) in a given order,
                              and stop as soon as it finds this credential
    :return: path to the local directory where the downloaded files are saved
    """
    # The second argument of "os.path.join" should not start with '/', so we make the following assertion to avoid
    # errors when doing os.path.join(local_root_dir, s3_dir), etc.
    assert not s3_bucket_name.startswith('/') and not s3_dir.startswith('/')

    s3 = boto3.resource('s3', region_name=s3_region,
                        aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    bucket = s3.Bucket(s3_bucket_name)

    for s3_object in bucket.objects.all():
        # If the file is in a directory "a/b/c" of the S3 bucket, s3_file_dir = 'a/b/c'
        # If the file is not in any directory of the S3 bucket, s3_file_dir = ''
        # Note that "s3_file_dir" and "s3_file_name" will never start with '/' and the S3 bucket name is not a part of the key
        s3_file_dir, s3_file_name = os.path.split(s3_object.key)
        if (not (s3_file_dir.rstrip('/') + '/').startswith(s3_dir.rstrip('/') + '/')) or \
           (len(s3_file_name) == 0) or (not s3_file_name.startswith(s3_file_name_prefix)):
            continue
        make_dir(os.path.join(local_root_dir, s3_bucket_name, s3_file_dir), method='keep')
        bucket.download_file(s3_object.key, os.path.join(local_root_dir, s3_bucket_name, s3_file_dir, s3_file_name))

    return os.path.join(local_root_dir, s3_bucket_name, s3_dir)


def local_to_s3(local_root_dir: str, local_dir: str, local_file_name_prefix: Union[str, Tuple[str]], s3_bucket_name: str,
                s3_root_dir: str = '', s3_region: str = None, aws_access_key_id: str = None, aws_secret_access_key: str = None) -> str:
    """
    Upload all the files with a particular prefix (local_file_name_prefix) recursively from a local directory (local_dir)
    to an S3 directory (s3_root_dir) in an S3 bucket (s3_bucket_name)
    """
    assert local_root_dir.startswith('/') and not local_dir.startswith('/') and not s3_root_dir.startswith('/')

    s3 = boto3.resource('s3', region_name=s3_region,
                        aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    all_name_dir = get_file_in_dir(os.path.join(local_root_dir, local_dir), start_with=local_file_name_prefix,
                                   return_full_path=True, keep_ext=True, recursively=True)
    for local_file_name, local_file_dir in all_name_dir:
        s3.meta.client.upload_file(os.path.join(local_file_dir, local_file_name), s3_bucket_name,
                                   os.path.join(s3_root_dir, relpath(local_file_dir, local_root_dir), local_file_name))

    s3_region_used = s3.meta.client.get_bucket_location(Bucket=s3_bucket_name)['LocationConstraint']
    if s3_region_used is None:
        return 'https://{0:}.s3.amazonaws.com/{1:}'.format(s3_bucket_name, os.path.join(s3_root_dir, local_dir))
    else:
        return 'https://{0:}.s3.{1:}.amazonaws.com/{2:}'.format(s3_bucket_name, s3_region_used, os.path.join(s3_root_dir, local_dir))

########################################################################################################################
def compare_str(old: str, new: str) -> Tuple[int, int]:
    old = old.split()
    new = new.split()

    n_deleted = 0
    n_added = 0
    for i, s in enumerate(difflib.ndiff(old, new)):
        if s[0] == ' ':
            continue
        elif s[0] == '-':
            n_deleted += len(s[1:].split())
        elif s[0] == '+':
            n_added += len(s[1:].split())

    return n_deleted, n_added


def compare_df(old: pd.DataFrame, new: pd.DataFrame, col_merge_on: Union[str, Tuple[str, ...]] = 'index',
               col_compare: Tuple[str, ...] = None, display_diff_only: bool = True, atol: float = 1e-6) -> pd.DataFrame:
    """
    It conducts side-by-side comparison of two dataframes, where the order of columns and the indices are ignored (
    two dataframes with different orders of columns or different indices are regarded as equal)
    :param old: the first dataframe to compare
    :param new: the second dataframe to compare
    :param col_merge_on: column(s) the two dataframes are aligned on, which is either column(s) existing in
                         both the two dataframes or "index" (meaning we simply put the two dataframes side-by-side)
    :param col_compare: column(s) we want to display the differences, which must be columns in the union of the columns of the two dataframes
    :param display_diff_only: whether to only display the different rows and columns
    :param atol: when the absolute difference between two numbers <= atol, we regard these two numbers as equal
    :return: a new dataframe displaying the differences between the two dataframes
    """
    old = old.sort_index(axis=1).reset_index(drop=True).reset_index(drop=(col_merge_on not in {'index', ('index', )}))
    new = new.sort_index(axis=1).reset_index(drop=True).reset_index(drop=(col_merge_on not in {'index', ('index', )}))

    if col_compare is None:
        col_compare = tuple(unique_retain_order(list(old.columns) + list(new.columns)))
    else:
        assert set(col_compare).issubset(set(list(old.columns) + list(new.columns)))

    old.columns = [col + '_old' for col in old.columns]
    new.columns = [col + '_new' for col in new.columns]

    old_on = (col_merge_on + '_old', ) if isinstance(col_merge_on, str) else (col + '_old' for col in col_merge_on)
    new_on = (col_merge_on + '_new', ) if isinstance(col_merge_on, str) else (col + '_new' for col in col_merge_on)
    assert old.loc[:, old_on].notnull().all().all() and new.loc[:, new_on].notnull().all().all()
    df = pd.merge(old, new, how='outer', left_on=old_on, right_on=new_on, suffixes=('', ''), validate='one_to_one').reset_index(drop=True)

    nrow = df.shape[0]
    col_both, row_both = list(), list()
    for col in col_compare:
        temp = list()
        if col + '_old' not in df.columns:
            # col is in the new dataframe only
            for elem_old_on, elem_new_on in tqdm(zip(df[old_on[0]].tolist(), df[new_on[0]].tolist()), desc=col, total=nrow):
                if non_element_wise_isnull(elem_old_on) and (not non_element_wise_isnull(elem_new_on)):
                    temp.append('COLUMN NEW ONLY; ROW NEW ONLY')
                elif (not non_element_wise_isnull(elem_old_on)) and non_element_wise_isnull(elem_new_on):
                    temp.append('COLUMN NEW ONLY; ROW OLD ONLY')
                else:
                    temp.append('COLUMN NEW ONLY')
        elif col + '_new' not in df.columns:
            # col is in the old dataframe only
            for elem_old_on, elem_new_on in tqdm(zip(df[old_on[0]].tolist(), df[new_on[0]].tolist()), desc=col, total=nrow):
                if non_element_wise_isnull(elem_old_on) and (not non_element_wise_isnull(elem_new_on)):
                    temp.append('COLUMN OLD ONLY; ROW NEW ONLY')
                elif (not non_element_wise_isnull(elem_old_on)) and non_element_wise_isnull(elem_new_on):
                    temp.append('COLUMN OLD ONLY; ROW OLD ONLY')
                else:
                    temp.append('COLUMN OLD ONLY')
        else:
            # col is in both the old and new dataframes
            col_both.append(col + '_compare')
            row_both = list()
            for i, (elem_old_on, elem_new_on, elem_old, elem_new) in tqdm(enumerate(zip(
                    df[old_on[0]].tolist(), df[new_on[0]].tolist(), df[col + '_old'].tolist(), df[col + '_new'].tolist())), desc=col, total=nrow):
                if non_element_wise_isnull(elem_old_on) and (not non_element_wise_isnull(elem_new_on)):
                    temp.append('ROW NEW ONLY')
                elif (not non_element_wise_isnull(elem_old_on)) and non_element_wise_isnull(elem_new_on):
                    temp.append('ROW OLD ONLY')
                else:
                    row_both.append(i)
                    assert (not non_element_wise_isnull(elem_old_on)) and (not non_element_wise_isnull(elem_new_on))
                    if non_element_wise_isnull(elem_old) and non_element_wise_isnull(elem_new):
                        temp.append('')  # An empty string means the two entries are same
                    elif non_element_wise_isnull(elem_old) and (not non_element_wise_isnull(elem_new)):
                        temp.append('OLD IS NONE BUT NEW IS NOT')
                    elif (not non_element_wise_isnull(elem_old)) and non_element_wise_isnull(elem_new):
                        temp.append('NEW IS NONE BUT OLD IS NOT')
                    else:
                        if is_number(elem_old) and is_number(elem_new):
                            temp.append('' if abs(elem_old - elem_new) <= atol else 'SAME TYPE BUT DIFFERENT VALUES')
                        elif isinstance(elem_old, str) and isinstance(elem_new, str):
                            if elem_old == elem_new:
                                temp.append('')
                            else:
                                n_deleted, n_added = compare_str(elem_old, elem_new)
                                if n_deleted == n_added == 0:
                                    temp.append('SAME TOKENS BUT DIFFERENT WHITE SPACES')  # For example, ' ' is different from '\xa0', though they look the same
                                else:
                                    temp.append('{0:d} TOKENS DELETED; {1:d} TOKENS ADDED'.format(n_deleted, n_added))
                        else:
                            if type(elem_old) != type(elem_new):
                                temp.append('DIFFERENT TYPES')
                            elif not non_element_wise_eq(elem_old, elem_new):
                                temp.append('SAME TYPE BUT DIFFERENT VALUES')
                            else:
                                temp.append('')
        assert len(temp) == df.shape[0]
        df[col + '_compare'] = temp

    if (not display_diff_only) or (len(col_both) == 0 or len(row_both) == 0):
        return df.reset_index(drop=True)

    if len(col_both) < len(col_compare) and len(row_both) == df.shape[0]:
        # In this case, we don't drop rows; otherwise we might drop all the rows so that the columns in old or new only won't be displayed
        assert row_both == list(range(df.shape[0]))
        return df.drop(columns=[col for col in col_both if (df.loc[row_both, col] == '').all()]).reset_index(drop=True)
    elif len(col_both) == len(col_compare) and len(row_both) < df.shape[0]:
        # In this case, we don't drop columns; otherwise we might drop all the columns so that the rows in old or new only won't be displayed
        assert col_both == [col + '_compare' for col in col_compare]
        return df.drop(index=[i for i in row_both if (df.loc[i, col_both] == '').all()]).reset_index(drop=True)
    else:
        return df.drop(columns=[col for col in col_both if (df.loc[row_both, col] == '').all()],
                       index=[i for i in row_both if (df.loc[i, col_both] == '').all()]).reset_index(drop=True)

########################################################################################################################
def get_start_end_index(l: int, n: int) -> List[tuple]:
    # Get the starting and ending indices to split a list of length l into n pieces
    assert isinstance(l, int) and isinstance(n, int) and (l >= n >= 1)
    k, m = divmod(l, n)
    return [(i * k + min(i, m), (i + 1) * k + min(i + 1, m)) for i in range(n)]


def enable_multiprocessing(n_cpu: int = psutil.cpu_count(logical=True)) -> Callable:
    assert (1 <= n_cpu <= psutil.cpu_count(logical=True))

    def enable_multiprocessing_decorator(old_func: Callable) -> Callable:
        """
        old_func must be a function that "can" (not "must") accept a list or tuple A as its first positional argument,
        and returns and only returns a list B when it accepts a list or tuple A as its first positional argument,
        where for i = 0, 1, ..., B[i] is computed using A[i], the other positional arguments and the keyword argumnets

        When the first positional argument of new_func is a long enough (length >= n_cpu) list or tuple A and n_cpu > 1,
        new_func will compute all B[i] using A[i], the other positional arguments and the keyword argumnets in parallel with n_cpu processes
        Otherwise, new_func will do exactly what old_func will do
        """
        # "@functools.wraps(old_func)" makes the decorated / returned function look like the function we will decorate: "old_func",
        # which enables us to pickle the decorated / returned function
        # Without it, the decorated / returned function will be the function defined below: "new_func", and we can't
        # pickle a function that is not defined at the top-level of a module (like this "new_func")
        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            if (n_cpu == 1) or (not args) or (not isinstance(args[0], (list, tuple))) or (len(args[0]) < n_cpu):
                # For now, we require the input argument we will split to be the first positional argument, but in the future,
                # we may allow it to be any positional or keyword argument specified by the user
                return old_func(*args, **kwargs)

            old_func_partial = functools.partial(old_func, **kwargs)
            args_list = [(args[0][start:end], ) + args[1:] for start, end in get_start_end_index(len(args[0]), n_cpu)]
            with Pool(n_cpu) as pool:
                result_list = pool.starmap(old_func_partial, args_list)

            return [elem for sub_list in result_list for elem in sub_list]

        return new_func

    return enable_multiprocessing_decorator


def single_to_list_input_decorator(old_func: Callable) -> Callable:
    """
    When the first positional argument of new_func is a list or tuple A,
    new_func will compute the result using A[i], the other positional arguments and the keyword argumnets and return all results in a list
    Otherwise, new_func will do exactly what old_func will do
    """
    @functools.wraps(old_func)
    def new_func(*args, **kwargs):
        if (not args) or (not isinstance(args[0], (list, tuple))):
            return old_func(*args, **kwargs)

        print(' ', end='', flush=True)  # This is a hack to display tqdm progress bars of multiple processes in jupyter notebooks
        return [old_func(arg0, *args[1:], **kwargs) for arg0 in tqdm(args[0], desc='proc{:d}'.format(os.getpid()))]

    return new_func

# Logger related utility functions #####################################################################################
def get_logger(name: str = None, logger_propagate: bool = True, logger_level: str = 'INFO',
               logger_filter: Callable[[LogRecord], bool] = None, all_handler_config: Tuple[dict, ...] = tuple()) -> Logger:
    if name is None:
        assert len(all_handler_config) > 0, 'The root logger must have at least 1 handlers'
        # Must not use logging.getLogger('root')
        logger = logging.getLogger()
    else:
        # Multiple calls to getLogger() with the same name will return a reference to the same logger object
        logger = logging.getLogger(name)

    for the_filter in logger.filters:
        logger.removeFilter(the_filter)
    assert len(logger.filters) == 0
    for the_handler in logger.handlers:
        logger.removeHandler(the_handler)
    assert len(logger.handlers) == 0

    logger.propagate = logger_propagate
    logger.setLevel(logger_level)
    if logger_filter is not None:
        logger.addFilter(logger_filter)
    for handler_config in all_handler_config:
        handler = get_handler(**handler_config)
        logger.addHandler(handler)

    return logger


class CustomizedFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)
        self.proc_id = None

    def _open(self):
        make_dir(os.path.dirname(self.baseFilename), method='keep')
        path, ext = os.path.splitext(self.baseFilename)
        assert (path != '') and (ext != '')
        base_filename_with_proc_id = '{0:}_proc{1:d}{2:}'.format(path, os.getpid(), ext)
        return open(base_filename_with_proc_id, self.mode, encoding=self.encoding)

    def emit(self, record):
        if self.proc_id != os.getpid() or self.stream is None:
            self.proc_id = os.getpid()
            self.stream = self._open()
        logging.StreamHandler.emit(self, record)


class CustomizedFormatter(logging.Formatter):
    converter = lambda *args: datetime.now(tz=get_localzone())

    def formatTime(self, record: LogRecord, datefmt: str = None) -> str:
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime(self.default_time_format)
            s = self.default_msec_format % (t, record.msecs)
        return s


def get_handler(handler_type: str, handler_level: str = 'INFO', handler_filter: Callable[[LogRecord], bool] = None,
                handler_format: str = None, handler_timezone: str = None, **kwargs) -> Handler:
    if handler_type == 'FileHandler':
        # If you create a FileHandler for directory D0 and then you rename it as D1,
        # the message will still be correctly sent to the log file in D1 when you log some messages
        # If you create a FileHandler for directory D0 and then you delete it,
        # the message won't be sent to any log file (but it won't raise any error)
        handler = CustomizedFileHandler(kwargs['handler_path'], mode='a', delay=True)
    elif handler_type == 'StreamHandler':
        handler = logging.StreamHandler()
    else:
        raise NotImplementedError

    handler.setLevel(handler_level)
    if handler_filter is not None:
        handler.addFilter(handler_filter)
    if handler_format is None:
        handler_format = '{asctime} {filename} {funcName} Line{lineno:d}: {message}'
    formatter = CustomizedFormatter(handler_format, datefmt='%Y/%m/%d-%H:%M:%S(%Z)', style='{')
    if handler_timezone is not None:
        formatter.converter = lambda *args: datetime.now(tz=pytz.timezone(handler_timezone))
    handler.setFormatter(formatter)

    return handler


def pretty_log(x, logger_func: callable, **kwargs):
    # logger_func can either be "logger.info", etc. or "print"
    for line in pprint.pformat(x, **kwargs).split('\n'):
        logger_func(line)
