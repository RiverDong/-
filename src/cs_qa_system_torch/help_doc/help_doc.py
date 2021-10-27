import copy
import functools
import json
import os
import pandas as pd
import psutil
import random
import re
import requests
import time
import timeout_decorator
from bs4 import BeautifulSoup, Comment
from collections import defaultdict
from logging import Logger
from random_user_agent.params import HardwareType, OperatingSystem, SoftwareEngine, SoftwareName, SoftwareType
from random_user_agent.user_agent import UserAgent
from selenium import webdriver
from typing import Callable, Dict, List, Optional, Tuple, Union

from .utils_help_doc import compare_df, copy_s3_dir, current_time, enable_multiprocessing, file_ext_from_path, \
                            get_file_in_dir, get_logger, has_word_char_str, hash_str, load_file, make_dir, pretty_log, \
                            s3_to_local, save_file, single_to_list_input_decorator, str_to_datetime, unique_retain_order


def help_doc_nodeid_from_s3(local_root_dir: str, s3_file_name_prefix: str, row_filter: Callable[[pd.Series], bool] = None,
                            s3_bucket_name: str = 'cs-help-index-data', s3_dir: str = '', s3_region: str = 'us-east-1',
                            aws_access_key_id: str = None, aws_secret_access_key: str = None) -> Tuple[List[str], pd.DataFrame]:

    file_dir = s3_to_local(local_root_dir, s3_bucket_name, s3_dir, s3_file_name_prefix, s3_region, aws_access_key_id, aws_secret_access_key)
    all_file_name = get_file_in_dir(file_dir, return_full_path=False, keep_ext=False, recursively=False)

    df_help_doc_list = list()
    file_name_col = list()
    for file_name in all_file_name:
        with open(os.path.join(file_dir, file_name), 'r') as f:
            file_content = json.load(f)
        df_help_doc_list.extend(file_content)
        file_name_col.extend([file_name] * len(file_content))
    df_help_doc = pd.DataFrame(df_help_doc_list).astype({'summary': str, 'country': str, 'site_variant': object,
        'keywords': object, 'hierarchy': str, 'title': str, 'content': str, 'autokeywords': object, 'parent_node_id': str,
        'tags': str, 'breadcrumb': object, 'support_area': str, 'node_id': str}).fillna('')
    df_help_doc['file_name'] = file_name_col

    row_mask = (df_help_doc['node_id'] != '') if row_filter is None else ((df_help_doc['node_id'] != '') & df_help_doc.apply(row_filter, axis=1))
    df_help_doc = df_help_doc.loc[row_mask, :]
    df_help_doc = df_help_doc.drop_duplicates('content').drop_duplicates('node_id').reset_index(drop=True)

    return df_help_doc['node_id'].tolist(), df_help_doc


class HelpDoc:

    def __init__(self, name: str, root_dir: str, logger_level: str = 'INFO', load: bool = False,
                 homepage_url: str = 'www.amazon.com',
                 language: str = None):
        if load:
            self.load(os.path.join(root_dir, name))
            return

        self.name = name
        self.root_dir = root_dir.rstrip('/')
        self.logger_level = logger_level

        self.logger = self.get_logger()

        self.homepage_url = homepage_url.strip()
        self.language = language

        self.raw_data = dict()
        self.data = dict()
        self.url_in_doc = dict()

    def save(self, name_saved: str = None, dir_saved: str = None, exclude: tuple = tuple()):
        if name_saved is None:
            name_saved = self.name
        if dir_saved is None:
            dir_saved = self.root_dir
        make_dir(dir_saved, method='keep')
        path_saved = os.path.join(dir_saved, name_saved)
        assert file_ext_from_path(path_saved) == ''  # To use dill.dump, file extension of the saved file must be ''

        assert set(exclude).issubset(set(self.__dict__.keys()))
        dict_saved = {k: v for k, v in self.__dict__.items() if k not in exclude}

        save_file(dict_saved, path_saved)

        if len(exclude) == 0:
            self.logger.info('{0:} saved to {1:}'.format(self.name, path_saved))
        else:
            self.logger.info('{0:} saved to {1:} with {2:} excluded'.format(self.name, path_saved, ', '.join(exclude)))

    def load(self, path_loaded: str):
        assert (file_ext_from_path(path_loaded) == '') and (not path_loaded.endswith('/'))

        self.__dict__ = load_file(path_loaded)

        old_name = self.name
        new_name = os.path.basename(path_loaded)
        if old_name != new_name:
            self.name = new_name

        old_root_dir = self.root_dir.rstrip('/')
        new_root_dir = os.path.dirname(path_loaded).rstrip('/')
        if old_root_dir != new_root_dir:
            self.root_dir = new_root_dir

        self.logger = self.get_logger()

        if old_name != new_name:
            self.logger.info('name changed from {0:} to {1:}'.format(old_name, new_name))

        if old_root_dir != new_root_dir:
            self.logger.info('root_dir changed from {0:} to {1:}'.format(old_root_dir, new_root_dir))

        self.logger.debug('{0:} loaded from {1:}'.format(self.name, path_loaded))

    def get_logger(self) -> Logger:
        time.sleep(2e-6)  # Sleep for 2e-6 second to avoid duplication of logger_name
        logger_name = os.path.join(self.root_dir, '{0:}_{1:}'.format(self.name, current_time(show_microsecond=True)))
        all_handler_config = (
            {'handler_type': 'FileHandler', 'handler_level': 'DEBUG', 'handler_filter': None,
             'handler_format': '{asctime}: {message}', 'handler_timezone': 'US/Pacific', 'handler_path': logger_name + '.log'},
        )
        return get_logger(logger_name,
            logger_propagate=True, logger_level=self.logger_level, logger_filter=None, all_handler_config=all_handler_config)

    def set_logger_level(self, logger_level: Union[str, int]):
        self.logger_level = logger_level
        self.logger.setLevel(logger_level)

    def init_dir(self, method: str = 'raise'):
        make_dir(self.root_dir, method=method)

    def time_list(self) -> List[str]:
        # Get a sorted list of all time_pulled in self.raw_data
        return sorted(self.raw_data.keys(), key=lambda x: str_to_datetime(x))

    def idx_to_time_pulled(self, x: Union[int, str]) -> str:
        if isinstance(x, int):
            return self.time_list()[x]
        else:
            return x

    # Functions to pull HTML ###########################################################################################
    def _get_url(self, x: str) -> Optional[str]:
        # Parse the input string to get the standard / canonical URL of an Amazon help document
        # If it is not possible to parse it or the final URL doesn't have desired format, return None
        input_regex = r'(?as)(((https?://)?{0:}/gp/help/customer/display[.]html[?]nodeId=)?(?P<nodeid>G?[0-9A-Z]+))'.format(self.homepage_url.replace('.', '[.]')) if not self.language else \
                      r'(?as)(((https?://)?{0:}/gp/help/customer/display[.]html[?]language={1:}&nodeId=)?(?P<nodeid>G?[0-9A-Z]+))'.format(self.homepage_url.replace('.', '[.]'), self.language)
        matchobj = re.fullmatch(input_regex, x)
        if not matchobj:
            return None

        nodeid = matchobj.group('nodeid')
        if re.fullmatch(r'(?as)([0-9]+)', nodeid) or (not re.fullmatch(r'(?as)(G[0-9]+)', nodeid) and len(nodeid) <= 15):
            nodeid = 'G' + nodeid

        url = r'https://{0:}/gp/help/customer/display.html?nodeId={1:}'.format(self.homepage_url, nodeid) if not self.language else \
              r'https://{0:}/gp/help/customer/display.html?language={1:}&nodeId={2:}'.format(self.homepage_url, self.language, nodeid)

        output_regex = r'(?as)(https://{0:}/gp/help/customer/display[.]html[?]nodeId=(G[0-9A-Z]+))'.format(self.homepage_url.replace('.', '[.]')) if not self.language else \
                       r'(?as)(https://{0:}/gp/help/customer/display[.]html[?]language={1:}&nodeId=(G[0-9A-Z]+))'.format(self.homepage_url.replace('.', '[.]'), self.language)
        if not re.fullmatch(output_regex, url):
            return None

        return url

    def _is_empty_help_doc(self, html_data: str) -> bool:
        try:
            soup = BeautifulSoup(html_data, 'html.parser')
            soup = soup.find_all('div', attrs={'class': ['help-content', 'cs-help-content']})[0]
            # If a tag has only one child, and that child is a NavigableString, the child is made available as .string
            # If a tag’s only child is another tag, and that tag has a .string, then the parent tag is considered to have the same .string as its child
            # If a tag contains more than one thing, then it’s not clear what .string should refer to, so .string is defined to be None
            return soup.string is not None and soup.string.strip() == ''
        except BaseException:
            return False

    def _help_doc_not_found(self, html_data: str) -> bool:
        # For non pop-up=1 version, it is not found iff all of the three words 'louis', 'french', 'bulldog' are in html_data.lower()
        # For pop-up=1 version, it is not found iff it is an empty webpage
        # This rule should hold for help documents in any marketplaces, but it might be changed in the future, so always double check before using it
        return all(x in html_data.lower() for x in ['louis', 'french', 'bulldog']) or self._is_empty_help_doc(html_data)

    def _treated_as_robot(self, html_data: str) -> bool:
        # This rule should hold for help documents in any marketplaces, but it might be changed in the future, so always double check before using it
        return 'To discuss automated access to Amazon data please contact' in html_data

    def _amazon_page_not_found(self, html_data: str) -> bool:
        """
        If you use "requests" to pull an Amazon webpage, it will raise HTTPError if the Amaozn webpage doesn't exist
        If you use "selenium" to pull an Amazon webpage, it won't raise any error if the Amaozn webpage doesn't exist
        In general, when using "selenium", it's impossible to get any HTTP response code, as it mimics a browsing behavior of a human

        Therefore, when using "selenium", we have to use this function to decide whether the Amazon webpage exists,
        directly according to the HTML we pulled

        Unlike the rules in "_help_doc_not_found" and "_treated_as_robot", the following rule may not hold for every marketplace,
        so you may want to double check the rule when you are working on a new marketplace
        """
        return re.search(r'(/error-image/|/images/G/[0-9]{2}/error/)', html_data) is not None

    def _get_user_agent(self, method: str = 'fixed') -> str:
        if method == 'random':
            user_agent_rotator = UserAgent(
                operating_systems=[OperatingSystem.MAC.value, OperatingSystem.WINDOWS.value],
                hardware_types=[HardwareType.COMPUTER.value],
                software_types=[SoftwareType.WEB_BROWSER.value],
                software_names=[SoftwareName.CHROME.value],
                software_engines=[SoftwareEngine.WEBKIT.value],
                limit=None
            )
            return user_agent_rotator.get_random_user_agent()
        else:
            return 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'

    def _pull_single_requests(self, url: str, time_limit: Union[int, float] = None) -> Tuple[Optional[str], Optional[str]]:
        try:
            header = requests.head(url, allow_redirects=True)
            content_type = header.headers.get('content-type')
            if not ('text' in content_type or 'html' in content_type):
                return None, 'NotWebpage'

            if time_limit is not None:
                requests_get_timeout = timeout_decorator.timeout(time_limit, use_signals=False)(requests.get)
                response = requests_get_timeout(url, headers={'User-Agent': self._get_user_agent(method='fixed')})
            else:
                response = requests.get(url, headers={'User-Agent': self._get_user_agent(method='fixed')})
            response.raise_for_status()
            result = response.text

            if self._help_doc_not_found(result):
                return None, 'HelpDocNotFound'
            elif self._treated_as_robot(result):
                return None, 'TreatedAsRobot'
            else:
                return result, None
        except BaseException as e:
            return None, type(e).__name__

    def _pull_single_selenium(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Using selenium as the backend to pull HTML is deprecated due to the following reasons:
        (1) The HTML it pulls often contains some unexpected duplications, or contains "javascript:void(0);", or misses some text
        (2) It doesn't handle the case of "AmazonPageNotFound" very well (see the comments in self._amazon_page_not_found)
        (3) It is 3 times slower than "requests"
        (4) It requires us to install the Chrome browswer and Chrome driver first
        (5) It doesn't solve the issue of being treated as robot, i.e., sometimes we will still be treated as robot
        """
        try:
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument('--headless')
            # Add the following settings to avoid being treated as robot
            chrome_options.add_argument('--disable-blink-features')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
            chrome_options.add_experimental_option('useAutomationExtension', False)

            # Disable JavaScript to speed up (0: default; 1: allow; 2: block)
            # prefs = {'profile.default_content_settings.javascript': 2,
            #          'profile.managed_default_content_settings.javascript': 2}
            # chrome_options.add_experimental_option('prefs', prefs)

            browser = webdriver.Chrome(chrome_options=chrome_options)
            # Add the following settings to avoid being treated as robot
            browser.execute_cdp_cmd('Network.setUserAgentOverride',
                {'userAgent': self._get_user_agent(method='fixed')}
            )
            browser.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument',
                {'source': "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"}
            )

            browser.get(url)
            result = browser.page_source

            if self._help_doc_not_found(result):
                return None, 'HelpDocNotFound'
            elif self._treated_as_robot(result):
                return None, 'TreatedAsRobot'
            elif self._amazon_page_not_found(result):
                return None, 'AmazonPageNotFound'
            else:
                return result, None
        except BaseException as e:
            return None, type(e).__name__

    def _pull_single(self, url: str, backend: str = 'requests', max_n_failure: int = 5, time_limit: Union[int, float] = None) -> Tuple[Optional[str], Optional[str]]:
        # This is a "general" function to pull HTMLs of any webpages, where the input must be the URL of a webpage (starting with http:// or https://)
        random.seed(hash_str(url))
        n_failure = 0
        while True:
            if backend == 'requests':
                result, error = self._pull_single_requests(url, time_limit=time_limit)
            else:
                assert backend == 'selenium'
                result, error = self._pull_single_selenium(url)

            if error is None:
                if n_failure >= 1:
                    self.logger.info('\n')
                return result, error
            elif error in {'NotWebpage', 'HelpDocNotFound', 'AmazonPageNotFound'}:
                self.logger.info('Fail to pull document {0:} with {1:} backend due to {2:}\n'.format(url, backend, error))
                return None, error
            else:
                n_failure += 1
                if n_failure > max_n_failure:
                    self.logger.info('Fail to pull document {0:} with {1:} backend due to {2:}, limit of number of failures ({3:d}) exceeded\n'.format(url, backend, error, max_n_failure))
                    return None, error
                else:
                    self.logger.info('Re-pull document {0:} with {1:} backend due to {2:}, number of failures: {3:d}'.format(url, backend, error, n_failure))
                    time.sleep(random.uniform(1 + n_failure, 6 + n_failure))

    def _pull(self, x: str, backend: str = 'requests', max_n_failure: int = 5) -> Dict[str, Optional[str]]:
        # This is a "specific" function to pull HTMLs of Amazon help documents, where the input can be the URL or node ID of an Amazon help document
        url = self._get_url(x)
        if url is None:
            self.logger.info('Fail to convert {0:} to a valid help document URL\n'.format(x))
            return {'url': x, 'html': None, 'html_simple': None}

        html_simple_data, _ = self._pull_single(url + '&pop-up=1', backend=backend, max_n_failure=max_n_failure, time_limit=None)
        html_data, _ = self._pull_single(url, backend=backend, max_n_failure=max_n_failure, time_limit=None)

        return {'url': url, 'html': html_data, 'html_simple': html_simple_data}

    def pull(self, x_list: Union[List[str], str], backend: str = 'requests', max_n_failure: int = 5,
             n_cpu: int = psutil.cpu_count(logical=True)) -> Union[List[dict], dict]:

        pull_func = enable_multiprocessing(n_cpu=n_cpu)(
            single_to_list_input_decorator(
                functools.partial(self._pull, backend=backend, max_n_failure=max_n_failure)
            )
        )

        return pull_func(x_list)

    # Function to insert raw_data ######################################################################################
    def pull_and_save(self, x_list: List[str], max_n_raw_data: int = None, backend: str = 'requests', max_n_failure: int = 5,
                      n_cpu: int = psutil.cpu_count(logical=True)) -> Tuple[str, pd.DataFrame]:
        assert isinstance(x_list, (list, tuple)) and len(x_list) > 0

        time.sleep(1)
        time_pulled = current_time()

        tic = time.time()
        data_pulled = self.pull(x_list, backend=backend, max_n_failure=max_n_failure, n_cpu=n_cpu)
        toc = time.time()

        success_url_list = [x['url'] for x in data_pulled if x['html'] is not None and x['html_simple'] is not None]
        failure_url_list = [x['url'] for x in data_pulled if x['html'] is None or x['html_simple'] is None]
        self.logger.info('{0:d} documents ({1:d} unique documents) successfully pulled with {2:} backend at {3:} in {4:.2f} seconds using '
            '{5:d} CPUs; {6:d} documents ({7:d} unique documents) failed to pull'.format(len(success_url_list),
            len(set(success_url_list)), backend, time_pulled, toc - tic, n_cpu, len(failure_url_list), len(set(failure_url_list))))

        for i in range(len(data_pulled)):
            assert 'time' not in data_pulled[i]
            data_pulled[i] = {**{'time': time_pulled}, **data_pulled[i]}

        if max_n_raw_data is not None:
            assert max_n_raw_data >= 1
            count = 1
            while len(self.raw_data) >= max_n_raw_data:
                self.logger.info('Raw data pulled at {0:} (the No. {1:d} oldest raw dataset) and its parsed data (if it exists) are removed, '
                'because the limit of the number of raw datasets is {2:d}'.format(self.idx_to_time_pulled(0), count, max_n_raw_data))
                self.del_raw_data(0)
                count += 1
        self.logger.info('\n')
        self.raw_data[time_pulled] = data_pulled

        return time_pulled, self.df_raw_data(time_pulled)

    # Functions to access and delete raw_data ##########################################################################
    def df_raw_data(self, idx_or_time_pulled: Union[int, str]) -> pd.DataFrame:
        time_pulled = self.idx_to_time_pulled(idx_or_time_pulled)

        df = copy.deepcopy(pd.DataFrame(self.raw_data[time_pulled]))

        for col in df.columns:
            if df[col].dtype == object or df[col].isnull().all():
                df[col] = df[col].fillna('')

        return df

    def del_raw_data(self, idx_or_time_pulled: Union[int, str]):
        """
        before calling: time_pulled is in raw_data and data;        after calling: time_pulled is not in raw_data or data
        before calling: time_pulled is in raw_data but not in data; after calling: time_pulled is not in raw_data or data
        before calling: time_pulled is not in raw_data but in data; after calling: NA (this scenario will never happen)
        before calling: time_pulled is not in raw_data or data;     after calling: NA (not allowed to call this function, as time_pulled is not in raw_data)
        """
        time_pulled = self.idx_to_time_pulled(idx_or_time_pulled)

        del self.raw_data[time_pulled]

        if time_pulled in self.data:
            del self.data[time_pulled]

    # Functions to parse HTML ##########################################################################################
    def _parse_node(self, node, title_list: list, url_list: list, level: int = None, para_dict: dict = None,
                    ignore_tags: tuple = ('script', 'style')) -> str:
        if node.name is None:
            # In this case, node is a string
            return '' if isinstance(node, Comment) else node

        last_text = ''
        text = ''
        for children in node:
            if children.name in ignore_tags:
                continue

            if children.name in {'h1', 'h2', 'h3'}:
                children_text = children.text.rstrip()
                if len(children_text) > 0:
                    if children_text[-1] in {':', '?', '.'}:
                        if children.name == 'h1' and len(title_list) == 0:
                            title_list.append(children_text[:-1].strip())
                        text = text.rstrip() + ' ' + children_text[:-1].rstrip() + ': '
                    else:
                        if children.name == 'h1' and len(title_list) == 0:
                            title_list.append(children_text.strip())
                        text = text.rstrip() + ' ' + children_text + ': '
            elif children.name == 'a':
                children_text = children.text.strip()
                children_href = children['href'].strip().replace(' ', '') if 'href' in children.attrs else ''
                if len(children_text) > 0 and len(children_href) > 0 and children_href[0] != '#' and children_href.lower() != 'javascript:void(0);':
                    # We exclude the ones starting with "#", e.g., [Back to the top](#GUID-D26A35E8-EE23-4B88-8B62-B357ABA6DDEE__SECTION_7D8EBD42E0184CC3BE4335003DD2222F)
                    # When using selenium as backend, the pulled html may randomly contain unexpected href="javascript:void(0);", so here we need to exclude such cases
                    # When using requests as backend, the pulled html will never contain unexpected href="javascript:void(0);"
                    if children_href[0] == '\\':
                        corrected_children_href = 'https://' + self.homepage_url + children_href.replace('\\', '/')
                        self.logger.debug('{0:} --> {1:} (https:// and homepage prepended; backslash replaced)'.format(children_href, corrected_children_href))
                    elif children_href[0] == '/':
                        corrected_children_href = 'https://' + self.homepage_url + children_href
                        self.logger.debug('{0:} --> {1:} (https:// and homepage prepended)'.format(children_href, corrected_children_href))
                    elif children_href.lower().startswith('gp/'):
                        corrected_children_href = 'https://' + self.homepage_url + '/' + children_href
                        self.logger.debug('{0:} --> {1:} (https:// and homepage prepended)'.format(children_href, corrected_children_href))
                    elif children_href.lower().startswith(('http://', 'https://', 'mailto:')):
                        corrected_children_href = children_href
                    elif self._get_url(children_href) is not None:
                        corrected_children_href = self._get_url(children_href)
                        self.logger.debug('{0:} --> {1:} (converted to standard URL of help document)'.format(children_href, corrected_children_href))
                    else:
                        corrected_children_href = 'https://' + children_href
                        self.logger.debug('{0:} --> {1:} (https:// prepended)'.format(children_href, corrected_children_href))
                    url_list.append(corrected_children_href)
                    text = text.rstrip() + ' [' + children_text + '](' + corrected_children_href + ')'
            elif children.name == 'table':
                children_text = self._parse_table(children, title_list, url_list)
                text = text.rstrip() + ' ' + children_text
            else:
                children_text = self._parse_node(children, title_list, url_list, level + 1, para_dict) if level is not None else self._parse_node(children, title_list, url_list)
                text = text + children_text

            if level is not None and text.rstrip() != last_text:
                para_dict[level].append(text[len(last_text):])
                last_text = text.rstrip()

        return text

    def _parse_table(self, node, title_list: list, url_list: list) -> str:
        text = ''
        colnames = []
        for children in node:
            if children.name == 'caption':
                caption = children.text.strip().strip(':').strip()
                text = text + ' ' + (caption + ': ') * int(len(caption) > 0)
            if children.name == 'thead':
                thead = children.find_all('th')
                colnames = [c.text.strip().strip(':').strip() for c in thead]
            if children.name == 'tbody':
                rows = children.find_all('tr')
                for i, r in enumerate(rows):
                    if len(colnames) == 0 and i == 0:
                        thead = r.find_all(['strong', 'b'])
                        if len(thead) > 0 and len(rows) > 1 and len(thead) == len(rows[1].find_all('td')) == len(r.find_all('td')):
                            colnames = [self._parse_node(c, title_list, url_list).strip().strip(':').strip() for c in thead]
                            continue
                    cols = r.find_all('td')
                    sentinel = ''
                    if len(colnames) == len(cols):
                        for j, c in enumerate(cols):
                            text = text + sentinel + (colnames[j] + '=') * int(len(colnames[j].strip()) > 0) + self._parse_node(c, title_list, url_list).strip()
                            sentinel = ', ' if len(text.strip()) > 0 and re.fullmatch(r'[])}>%/+*\w]', text.strip()[-1]) else ' '
                        text = text + ('. ' if len(text.strip()) > 0 and re.fullmatch(r'[])}>%/+*\w]', text.strip()[-1]) else ' ')
                    else:
                        for c in cols:
                            text = text + sentinel + self._parse_node(c, title_list, url_list).strip()
                            sentinel = ', ' if len(text.strip()) > 0 and re.fullmatch(r'[])}>%/+*\w]', text.strip()[-1]) else ' '
                        text = text + ('. ' if len(text.strip()) > 0 and re.fullmatch(r'[])}>%/+*\w]', text.strip()[-1]) else ' ')
        return text

    def _parse_breadcrumb(self, node, result: list):
        if node.name is None:
            # In this case, node is a string
            if has_word_char_str(node):
                result.append(node.strip())
            return

        for children in node:
            self._parse_breadcrumb(children, result)

    def _parse_sidebar(self, node, parent_name: Optional[str], parent_class: list, root: list, selected: list, result: list):
        if node.name is None:
            # In this case, node is a string
            if has_word_char_str(node):
                result.append(node.strip())
                if 'h3' == parent_name:
                    root.append(node.strip())
                if 'selected' in parent_class:
                    selected.append(node.strip())
            return

        for children in node:
            node_class = node.attrs['class'] if 'class' in node.attrs else list()
            self._parse_sidebar(children, node.name, node_class, root, selected, result)

    def _postpreprocess(self, x: str) -> str:
        x = re.sub(r'\s*[.]+\s*[\n]+\s*', '. ', x.strip())
        x = re.sub(r'[:]+\s*[\n]+\s*', ': ', x)
        x = re.sub(r'[?]+\s*[\n]+\s*', '? ', x)
        x = re.sub(r'(?<=[])}>%/+*\w])(\s*[\n]+\s*)([A-Z])', r'. \2', x)
        x = re.sub(r'\s*[\n\xa0 ]+\s*', ' ', x)
        return x

    def _concatenate_str(self, all_s: List[str]) -> List[str]:
        # For each string in the list "all_s", concatenate the next string to it
        # if it is very short, or it ends with ":" and is not very long
        if len(all_s) <= 1:
            return all_s

        next_must_be_concatenated_to_last = False
        result = list()
        for s in all_s:
            if next_must_be_concatenated_to_last:
                result[-1] += ' ' + s
            else:
                result.append(s)
            next_must_be_concatenated_to_last = (len(s.split()) <= 5) or (s.rstrip()[-1] == ':' and len(s.split()) <= 20)

        assert ' '.join(result) == ' '.join(all_s)
        return result

    def _parse_html_simple_single(self, html_simple_data: str) -> dict:
        try:
            soup = BeautifulSoup(html_simple_data, 'html5lib')
            title_backup = ':'.join(soup.title.text.split(':')[1:]).strip() if ':' in soup.title.text else soup.title.text.strip()
            soup = soup.find_all('div', attrs={'class': ['help-content', 'cs-help-content']})[0]

            para_dict = defaultdict(list)
            url_list = list()
            title_list = list()
            text = self._parse_node(soup, title_list, url_list, 0, para_dict)
            title = title_list[0] if len(title_list) > 0 else title_backup

            text = self._postpreprocess(text)
            title = self._postpreprocess(title)
            para_dict = {level: [self._postpreprocess(p) for p in para] for level, para in para_dict.items()}

            assert sorted(para_dict.keys()) == list(range(len(para_dict)))
            para_list = para_dict[0] if len(para_dict) > 0 else list()
            for level in range(len(para_dict)):
                para = para_dict[level]
                if len(para) >= 2:
                    para_list = [para[0]] + para_dict[level + 1] if (len(para) == 2) and (para[0].strip(': ') == title) and (level < len(para_dict) - 1) else para
                    break
            paragraphs = self._concatenate_str(para_list)

            assert re.sub(r'[. ]', '', ' '.join(paragraphs)) == re.sub(r'[. ]', '', text)
            return {'title': title, 'text': text, 'paragraphs': paragraphs, 'url_in_doc': url_list}
        except BaseException:
            return dict()

    def _parse_html_single(self, html_data: str) -> dict:
        try:
            soup = BeautifulSoup(html_data, 'html5lib')
            soup = soup.find_all('div', attrs={'class': ['cs-help-breadcrumb', 'cs-help-sidebar-module topic-sidebar']})
            assert len(soup) == 2
            assert soup[0].attrs == {'class': ['cs-help-breadcrumb']}
            assert soup[1].attrs == {'class': ['cs-help-sidebar-module', 'topic-sidebar']}

            breadcrumb = list()
            self._parse_breadcrumb(soup[0], breadcrumb)

            sidebar_root, sidebar_selected, sidebar_list = list(), list(), list()
            self._parse_sidebar(soup[1], None, list(), sidebar_root, sidebar_selected, sidebar_list)
            assert len(sidebar_root) <= 1 and len(sidebar_selected) <= 1

            sidebar_selected = sidebar_selected[0] if len(sidebar_selected) > 0 else ''
            assert sidebar_selected == '' or sidebar_selected in sidebar_list

            # If len(sidebar_root) > 0, sidebar_list[0] is a title like 'All Help Topics' (US EN documents)
            # and sidebar_list[1] is sidebar_root[0], so we skip these two entries ans use sidebar_list[2:]
            sidebar = {sidebar_root[0]: sidebar_list[2:]} if len(sidebar_root) > 0 else {'': sidebar_list[1:]}

            return {'breadcrumb': breadcrumb, 'sidebar': sidebar, 'sidebar_selected': sidebar_selected}
        except BaseException:
            return dict()

    def log_data_parsed(self, data_parsed: dict, logger_func: callable, width=120):
        for k, v in data_parsed.items():
            left_right_len = width - 2 - len(k)
            assert left_right_len >= 4
            left_len, right_len = left_right_len // 2 + (left_right_len % 2), left_right_len // 2
            logger_func('=' * left_len + ' ' + k + ' ' + '=' * right_len)

            if k == 'paragraphs':
                for i, p in enumerate(v):
                    if i != 0:
                        logger_func('-' * int(0.9 * width))
                    pretty_log(p, logger_func, width=int(0.9 * width))
            else:
                pretty_log(v, logger_func, width=width)
        logger_func('=' * width + '\n')

    def _parse(self, x: dict) -> dict:
        html_simple_parsed_single = self._parse_html_simple_single(x['html_simple'])
        html_parsed_single = self._parse_html_single(x['html'])

        assert {'time', 'url'}.isdisjoint(set(html_simple_parsed_single.keys()))
        assert {'time', 'url'}.isdisjoint(set(html_parsed_single.keys()))
        assert set(html_simple_parsed_single.keys()).isdisjoint(set(html_parsed_single.keys()))

        if 'time' in x:
            data_parsed_single = {**{'time': x['time'], 'url': x['url']}, **html_simple_parsed_single, **html_parsed_single}
        else:
            data_parsed_single = {**{'url': x['url']}, **html_simple_parsed_single, **html_parsed_single}
        self.log_data_parsed(data_parsed_single, self.logger.debug)

        return data_parsed_single

    def parse(self, x_list: Union[List[dict], dict], n_cpu: int = psutil.cpu_count(logical=True)) -> Union[List[dict], dict]:

        parse_func = enable_multiprocessing(n_cpu=n_cpu)(
            single_to_list_input_decorator(self._parse)
        )

        return parse_func(x_list)

    # Functions to insert and update data ##############################################################################
    def _update_data(self, time_pulled: str, data_new: List[dict]):
        if time_pulled not in self.data:
            self.data[time_pulled] = data_new
        else:
            # Check whether all rows are matched
            assert len(self.data[time_pulled]) == len(data_new)
            for old, new in zip(self.data[time_pulled], data_new):
                assert old['time'] == new['time'] and old['url'] == new['url']

            # Updates will be done only after making sure all rows are matched
            for old, new in zip(self.data[time_pulled], data_new):
                old.update(new)

    def parse_and_save(self, idx_or_time_pulled: Union[int, str], n_cpu: int = psutil.cpu_count(logical=True)) -> Tuple[str, pd.DataFrame]:
        """
        before calling: time_pulled is in raw_data and data;        after calling: time_pulled is in raw_data and data (data[time_pulled] is updated)
        before calling: time_pulled is in raw_data but not in data; after calling: time_pulled is in raw_data and data (data[time_pulled] is inserted)
        before calling: time_pulled is not in raw_data but in data; after calling: NA (not allowed to call this function, as time_pulled must be in raw_data, should call pull_and_save or change time_pulled)
        before calling: time_pulled is not in raw_data or data;     after calling: NA (not allowed to call this function, as time_pulled must be in raw_data, should call pull_and_save or change time_pulled)
        """
        time_pulled = self.idx_to_time_pulled(idx_or_time_pulled)

        assert self.raw_data[time_pulled] is not None

        tic = time.time()
        data_parsed = self.parse(self.raw_data[time_pulled], n_cpu)
        toc = time.time()

        if len(data_parsed) > 0:
            max_n_features = max(len(x) for x in data_parsed)
            assert max_n_features >= 3
            success_url_list = [x['url'] for x in data_parsed if len(x) == max_n_features]
            failure_url_list = [x['url'] for x in data_parsed if len(x) < max_n_features]
        else:
            success_url_list = list()
            failure_url_list = list()
        self.logger.info('{0:d} documents ({1:d} unique documents) pulled at {2:} successfully parsed in {3:.2f} seconds using '
            '{4:d} CPUs; {5:d} documents ({6:d} unique documents) failed to parse\n'.format(len(success_url_list),
            len(set(success_url_list)), time_pulled, toc - tic, n_cpu, len(failure_url_list), len(set(failure_url_list))))

        self._update_data(time_pulled, data_parsed)

        return time_pulled, self.__getitem__(time_pulled)

    def __setitem__(self, idx_or_time_pulled: Union[int, str], df: pd.DataFrame):
        """
        before calling: time_pulled is in raw_data and data;        after calling: time_pulled is in raw_data and data (data[time_pulled] is updated)
        before calling: time_pulled is in raw_data but not in data; after calling: NA (not allowed to call this function, should call "parse_and_save")
        before calling: time_pulled is not in raw_data but in data; after calling: NA (this scenario will never happen, as we will always automatically insert raw_data[time_pulled])
        before calling: time_pulled is not in raw_data or data;     after calling: time_pulled is in raw_data and data (raw_data[time_pulled] and data[time_pulled] are inserted)

        If you load a dataframe from an Excel file and use that dataframe here to update self.data, please retain only
        the columns that are expected to be changed (and have been changed) in df, before passing df to this function

        This is because, when you saved the dataframe to the Excel file, the content in some cells could already be "silently"
        truncated to the character limit (32767 characters) of Excel, but you may not know. In this case, if you accidently
        retain these silently changed columns in df, the original self.data will also be silently & unexpectedly updated to
        the truncated content

        However, if you load a dataframe from a tsv file or another "HelpDoc" instance, nothing will be "silently" changed,
        so feel free to retain all columns in df, before passing df to this function
        """
        time_pulled = self.idx_to_time_pulled(idx_or_time_pulled)

        assert not (time_pulled in self.raw_data and time_pulled not in self.data)

        if time_pulled not in self.raw_data:
            self.raw_data[time_pulled] = None

        data_df = df.to_dict('records')

        self._update_data(time_pulled, data_df)

    # Functions to access and delete data ##############################################################################
    def __getitem__(self, idx_or_time_pulled: Union[int, str]) -> pd.DataFrame:
        time_pulled = self.idx_to_time_pulled(idx_or_time_pulled)

        df = copy.deepcopy(pd.DataFrame(self.data[time_pulled]))

        for col in df.columns:
            if df[col].dtype == object or df[col].isnull().all():
                df[col] = df[col].fillna('')

        return df

    def __delitem__(self, idx_or_time_pulled: Union[int, str]):
        """
        before calling: time_pulled is in raw_data and data;        after calling: time_pulled is in raw_data but not in data
        before calling: time_pulled is in raw_data but not in data; after calling: NA (not allowed to call this function, as time_pulled is not in data)
        before calling: time_pulled is not in raw_data but in data; after calling: NA (this scenario will never happen)
        before calling: time_pulled is not in raw_data or data;     after calling: NA (not allowed to call this function, as time_pulled is not in data)
        """
        time_pulled = self.idx_to_time_pulled(idx_or_time_pulled)

        del self.data[time_pulled]

    # Function to compare data pulled at different times ###############################################################
    def compare(self, idx_or_time_pulled1: Union[int, str], idx_or_time_pulled2: Union[int, str], col_compare: Tuple[str, ...] = None) -> pd.DataFrame:
        if str_to_datetime(self.idx_to_time_pulled(idx_or_time_pulled1)) > str_to_datetime(self.idx_to_time_pulled(idx_or_time_pulled2)):
            old, new = self.__getitem__(idx_or_time_pulled2), self.__getitem__(idx_or_time_pulled1)
        else:
            old, new = self.__getitem__(idx_or_time_pulled1), self.__getitem__(idx_or_time_pulled2)

        if col_compare is None:
            col_compare = tuple(col for col in unique_retain_order(list(old.columns) + list(new.columns)) if col not in {'time', 'url'})
        return compare_df(old, new, col_merge_on='url', col_compare=col_compare, display_diff_only=True)

    # Functions to check the validity of URLs in each document #########################################################
    def _is_external_url(self, url: str) -> bool:
        return all(s not in url.lower() for s in ['amazon', 'www.audible.com', 'www.primevideo.com', 'amzn.to'])

    def _check(self, url: str, filter_out: Callable[[str], Optional[str]] = lambda x: None, backend: str = 'requests', max_n_failure: int = 2) -> str:
        if url.lower().startswith('mailto:'):
            filter_out_reason = 'EmailAddress'
        elif self._is_external_url(url):
            filter_out_reason = 'ExternalWebpage'
        else:
            filter_out_reason = filter_out(url)

        if filter_out_reason is not None:
            return 'FilteredOut' + filter_out_reason
        else:
            # Unlike the case in "self._pull" where we pull only help documents, here we will pull much more various kinds of URLs so
            # we always want to set a time limit when using self._pull_single to avoid taking too much time
            # Due to unknown reason, it won't timeout if we use time_limit > 3, so here we use time_limit = 2 for safety
            _, error = self._pull_single(url, backend=backend, max_n_failure=max_n_failure, time_limit=2)
            return error

    def check(self, url_list: Union[List[str], str], filter_out: Callable[[str], Optional[str]] = lambda x: None,
              backend: str = 'requests', max_n_failure: int = 2, n_cpu: int = psutil.cpu_count(logical=True)) -> Union[List[str], str]:

        check_func = enable_multiprocessing(n_cpu=n_cpu)(
            single_to_list_input_decorator(
                functools.partial(self._check, filter_out=filter_out, backend=backend, max_n_failure=max_n_failure)
            )
        )

        return check_func(url_list)

    def check_and_save(self, idx_or_time_pulled: Union[int, str], filter_out: Callable[[str], Optional[str]] = lambda x: None,
              backend: str = 'requests', max_n_failure: int = 2, n_cpu: int = psutil.cpu_count(logical=True)) -> Tuple[str, pd.DataFrame]:
        time_pulled = self.idx_to_time_pulled(idx_or_time_pulled)

        url_in_doc_single = dict()
        for data_parsed in self.data[time_pulled]:
            for url in data_parsed['url_in_doc']:
                if url not in url_in_doc_single:
                    url_in_doc_single[url] = {'doc': [data_parsed['url']], 'error': 'NotChecked'}
                elif data_parsed['url'] not in url_in_doc_single[url]['doc']:
                    url_in_doc_single[url]['doc'].append(data_parsed['url'])

        url_list = [url for url, _ in url_in_doc_single.items()]
        url_already_pulled = {data_parsed['url'] for data_parsed in self.data[time_pulled] if len(data_parsed) > 2}
        tic = time.time()
        error_list = self.check(url_list, filter_out=lambda x: 'HelpDocAlreadyPulled' if self._get_url(x) in url_already_pulled else filter_out(x),
                                backend=backend, max_n_failure=max_n_failure, n_cpu=n_cpu)
        toc = time.time()

        n_valid_url = sum(error in {None, 'NotWebpage', 'TreatedAsRobot', 'TimeoutError'} or error.startswith('FilteredOut') for error in error_list)
        n_invalid_url = len(error_list) - n_valid_url
        self.logger.info('{0:d} unique URLs in documents pulled at {1:} checked with {2:} backend in {3:.2f} seconds using {4:d} CPUs, '
            'where {5:d} URLs are considered as valid and {6:d} URLs are considered as invalid (see below for the error distribution)'.format(
                len(url_list), time_pulled, backend, toc - tic, n_cpu, n_valid_url, n_invalid_url))
        pretty_log(pd.Series(error_list).value_counts(dropna=False), self.logger.info)

        for i, (_, v) in enumerate(url_in_doc_single.items()):
            v['error'] = error_list[i]
        self.url_in_doc[time_pulled] = url_in_doc_single

        return time_pulled, self.df_url_in_doc(time_pulled)

    def df_url_in_doc(self, idx_or_time_pulled: Union[int, str], row_filter: Union[str, Callable[[pd.Series], bool]] = None) -> pd.DataFrame:
        time_pulled = self.idx_to_time_pulled(idx_or_time_pulled)

        df = copy.deepcopy(pd.DataFrame.from_dict(self.url_in_doc[time_pulled], orient='index'))
        df.index.name = 'url_in_doc'
        df = df.reset_index(drop=False)

        for col in df.columns:
            if df[col].dtype == object or df[col].isnull().all():
                df[col] = df[col].fillna('')

        if row_filter is None:
            row_mask = pd.Series([True] * df.shape[0], index=df.index)
        elif row_filter == 'valid':
            row_mask = df['error'].map(lambda x: x in {'', 'NotWebpage', 'TreatedAsRobot', 'TimeoutError'} or x.startswith('FilteredOut'))
        elif row_filter == 'invalid':
            row_mask = df['error'].map(lambda x: not (x in {'', 'NotWebpage', 'TreatedAsRobot', 'TimeoutError'} or x.startswith('FilteredOut')))
        else:
            row_mask = df.apply(row_filter, axis=1)

        return df.loc[row_mask, :].reset_index(drop=True)

    def __call__(self, x_list: List[str], s3_bucket_name: str = None, s3_dir: str = None, max_n_raw_data: int = 2,
                 n_cpu_pull: int = 1, n_cpu_parse: int = 1, n_cpu_check: int = 1):
        # This is the final function that can be regularly called in production to automatically update the help document collection
        _, _ = self.pull_and_save(x_list, max_n_raw_data=max_n_raw_data, n_cpu=n_cpu_pull)
        _, df_data = self.parse_and_save(-1, n_cpu=n_cpu_parse)
        _, df_url_in_doc = self.check_and_save(-1, n_cpu=n_cpu_check)

        if s3_bucket_name is not None and s3_dir is not None:
            s3_dir = s3_dir.strip('/')
            if len(self.time_list()) >= 2:
                copy_s3_dir(s3_dir, s3_dir + '_' + self.time_list()[-2], s3_bucket_name=s3_bucket_name)
            df_data.to_csv(os.path.join('s3://{0:}/{1:}'.format(s3_bucket_name, s3_dir), 'df_data.tsv'), sep='\t', index=False)
            df_url_in_doc.to_csv(os.path.join('s3://{0:}/{1:}'.format(s3_bucket_name, s3_dir), 'df_url_in_doc.tsv'), sep='\t', index=False)

