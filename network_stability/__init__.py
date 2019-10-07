import time
import datetime
from os.path import splitext

import speedtest
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import socket


class NetworkTest(object):
    REMOTE_SERVER = "motherfuckingwebsite.com"

    def __init__(self):
        self.speed_data = pd.DataFrame()
        self.connection_data = pd.DataFrame()

    def connection_test(self, timeout=10):
        """
        Check connection only.
        :param timeout: Time before giving up connection.
        :return: boolean.
        """
        # Start time
        start = datetime.datetime.now()
        try:
            # Get host
            host = socket.gethostbyname(self.REMOTE_SERVER)
            # Connect to the host
            s = socket.create_connection((host, 80), timeout)
            s.close()
            # wait until timeout  to continue.
            while (datetime.datetime.now() - start).total_seconds() < timeout:
                time.sleep(0.1)
            # Calculate time elapsed
            duration = datetime.datetime.now() - start
            # Save results
            self.connection_data = self.connection_data.append({'duration': duration.total_seconds(),
                                                                'connection': True,
                                                                'time': start}, ignore_index=True)
            return True
        except (OSError, socket.gaierror):
            # wait until timeout  to continue.
            while (datetime.datetime.now() - start).total_seconds() < timeout:
                time.sleep(0.1)
            # Calculate time elapsed
            duration = datetime.datetime.now() - start
            self.connection_data = self.connection_data.append({'duration': duration.total_seconds(),
                                                                'connection': False,
                                                                'time': start}, ignore_index=True)
        return False

    def connection_test_interval(self, seconds=0, minutes=0, hours=0, days=0, timeout=10):
        """
        Test network connectivity for a time interval.
        :param seconds: duration in seconds.
        :param minutes: duration in minutes.
        :param hours: duration in hours.
        :param days: duration in days.
        :param timeout: timeout for the speed test.
        :return: Pandas DataFrame
        """
        print('Initializing test.')
        end = datetime.datetime.now() + datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        rows = []
        while end > datetime.datetime.now():
            rows.append(self.connection_test(timeout))
            delta_time = end - datetime.datetime.now()
            if delta_time.days < 0:
                delta_time = datetime.timedelta(0)
            print(f'\r{delta_time} remaining.', end='')
        print()

        return self.connection_data

    @staticmethod
    def add_prefix(dic, prefix):
        """
        Add a prefix for all keys in a dict
        :param dic: dictionary
        :param prefix: string
        :return: dict with new keys
        """
        return {f'{prefix}_{key}': item for key, item in dic.items()}

    def speed_test(self, timeout=60):
        """
        Make a speed test.
        :param timeout: timeout for the speed test.
        :return: results dict
        """
        start = datetime.datetime.now()
        try:
            s = speedtest.Speedtest(timeout=timeout)
            # Find best server
            s.get_servers()
            s.get_best_server()
            # Test download and upload
            s.download()
            s.upload()
        except(speedtest.ConfigRetrievalError, socket.timeout):
            # wait until timeout  to continue.
            while (datetime.datetime.now() - start).total_seconds() < timeout:
                time.sleep(0.1)
            return pd.DataFrame(data={'time': datetime.datetime.now()}, index=[0])
        # Transform results in a dict, to later transform it in a pandas DataFrame
        results = s.results.dict()

        # The results dict have two inner dicts for the client and server information. I will extract the data from
        # those keys and put in the same level as the main dict. This is done because I want to return all results in a
        # single row pandas DataFrame. Nested dictionaries would make it very confusing to work the data later.
        keys = ('client', 'server')
        for key in keys:
            # So first I have to add a prefix to the keys to avoid duplicate keys.
            results[key] = self.add_prefix(results[key], key)
            # And then update the upper level dictionary with the lower level dictionaries.
            results.update(results[key])
            # After extracting data from this key, it is no longer required.
            results.pop(key)

        # wait until timeout  to continue.
        while (datetime.datetime.now() - start).total_seconds() < timeout:
            time.sleep(0.1)
        # Add a key for the time elapsed
        duration = datetime.datetime.now() - start
        results['duration'] = duration.total_seconds()
        # Test time
        results['time'] = datetime.datetime.now()

        # Save results
        self.speed_data = self.speed_data.append(results, ignore_index=True)

        return self.speed_data

    def speed_test_interval(self, seconds=0, minutes=0, hours=0, days=0, timeout=60):
        """
        Test network speed for a time interval.
        :param seconds: duration in seconds.
        :param minutes: duration in minutes.
        :param hours: duration in hours.
        :param days: duration in days.
        :param timeout: timeout for the speed test.
        :return: Pandas DataFrame
        """
        print('Initializing test.')
        end = datetime.datetime.now() + datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        rows = []
        while end > datetime.datetime.now():
            rows.append(self.speed_test(timeout))
            delta_time = end - datetime.datetime.now()
            if delta_time.days < 0:
                delta_time = datetime.timedelta(0)
            print(f'\r{delta_time} remaining.', end='')
        print()

        return self.speed_data

    @staticmethod
    def export_results(dataframe, file_name):
        """
        Export results DataFrame to a tabular file:
        :param dataframe: Pandas dataframe with results.
        :param file_name: Name of the file, including extension.
        :return: DataFrame
        """
        if 'csv' in file_name[-4:]:
            dataframe.to_csv(file_name)
        elif 'xls' in file_name[-4:]:
            dataframe.to_excel(file_name)
        elif 'json' in file_name[-4:]:
            dataframe.to_json(file_name)
        else:
            ValueError(f"Extension '{splitext(file_name)[-1]}' not supported.")

        return dataframe

    def export_speed_results(self, file_name):
        """
        Export speed test results DataFrame to a tabular file:
        :param file_name: Name of the file, including extension.
        :return: DataFrame
        """
        return self.export_results(self.speed_data, file_name)

    def export_connection_results(self, file_name):
        """
        Export connection test results DataFrame to a tabular file:
        :param file_name: Name of the file, including extension.
        :return: DataFrame
        """
        return self.export_results(self.connection_data, file_name)

    def report_speed(self, file_name):
        """
        Report speed results in a figure.
        :param file_name: Figure file name.
        :return: Figure
        """

        # Convert from b to Gb.
        dw = self.speed_data['download'] / (1024 * 1024)
        up = self.speed_data['upload'] / (1024 * 1024)
        pg = self.speed_data['ping']
        x = self.speed_data.index

        # Calculate mean ignoring NaN
        dw_mean = np.nanmean(dw)
        up_mean = np.nanmean(up)
        pg_mean = np.nanmean(pg)
        # Calculate the duration
        duration = self.speed_data['duration'].sum()
        # Then prepare it to be printed.
        duration_str = f'{int(duration // 3600):02d}:{int(np.rint(duration % 3600 / 60)):02d}'
        # Calculate the percentage of time online.
        online = self.speed_data.dropna(subset=['download'])['duration'].sum() / self.speed_data['duration'].sum()

        try:
            # interpolate the download and upload to create a smooth line
            x_new = np.linspace(min(x), max(x), num=len(x) * 10, endpoint=True)
            f_dw = interp1d(x, dw, kind='cubic')
            f_up = interp1d(x, up, kind='cubic')
            dw_new = f_dw(x_new)
            up_new = f_up(x_new)
        except ValueError:
            x_new = x
            dw_new = dw
            up_new = up

        # color maps to plots
        div = plt.get_cmap('winter')
        div_r = plt.get_cmap('winter_r')

        # define marker based on the number of tests
        if len(x) == 1:
            marker = 'o'
        else:
            marker = '-'

        # plot time series
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_new, dw_new, marker, color=div(0.0), linewidth=2)
        ax.plot(x_new, up_new, marker, color=div(1.0), linewidth=2)

        # hide borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # hide ticks and x values
        ax.tick_params(axis='x', bottom=False)
        ax.tick_params(axis='y', left=False)
        ax.set_xticks([])
        # sets legend
        ax.legend(['Download', 'Upload'], loc='upper left')

        # Test duration
        fig.text(x=1, y=0.8, s='Test Duration', ha='center', va='bottom')
        fig.text(x=1, y=0.8, s=duration_str, ha='center', va='top', color=div(duration / 3600), size='xx-large')
        # Average download speed
        fig.text(x=1, y=0.65, s='Average Download Speed', ha='center', va='bottom')
        fig.text(x=1, y=0.65, s=f'{round(dw_mean, 1)} Gbps', ha='center', va='top', color=div(dw_mean / 35),
                 size='xx-large')
        # Average upload speed
        fig.text(x=1, y=0.5, s='Average Upload Speed', ha='center', va='bottom')
        fig.text(x=1, y=0.5, s=f'{round(up_mean, 1)} Gbps', ha='center', va='top', color=div(up_mean / 35),
                 size='xx-large')
        # Average ping
        fig.text(x=1, y=0.35, s='Average Ping', ha='center', va='bottom')
        fig.text(x=1, y=0.35, s=f'{int(pg_mean)} ms', ha='center', va='top', color=div_r(pg_mean / 100),
                 size='xx-large')
        # Online time
        fig.text(x=1, y=0.2, s='Online Time', ha='center', va='bottom')
        fig.text(x=1, y=0.2, s=f'{int(online * 100)} %', ha='center', va='top', color=div(online), size='xx-large')

        # Save figure
        plt.savefig(file_name, bbox_inches='tight')

        return fig

    def report_connection(self, file_name):
        """
        Report speed results in a figure.
        :param file_name: Figure file name.
        :return: Figure
        """
        conn = self.connection_data['connection']
        notconn = 1 - self.connection_data['connection']
        x = self.connection_data.index

        # Calculate the duration
        duration = self.connection_data['duration'].sum()
        # Then prepare it to be printed.
        duration_str = f'{int(duration // 3600):02d}:{int(np.rint(duration % 3600 / 60)):02d}'

        # Calculate the percentage of time online.
        online = self.connection_data[self.connection_data['connection'] == True]['duration'].sum() \
                 / self.connection_data['duration'].sum()

        # color maps to plots
        div = plt.get_cmap('winter')

        # plot time series
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.bar(x, conn, width=1, color=div(1.0))
        ax.bar(x, notconn, width=1, color=div(0.0))

        # hide borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # hide ticks and x values
        ax.tick_params(axis='x', bottom=False)
        ax.tick_params(axis='y', left=False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Set y axes limits because the max is 1. So the bars will only occupy half the figure. We can then use the
        # other half to put the annotations on.
        ax.set_ylim(0, 2)

        # sets legend
        ax.legend(['Online', 'Offline'], loc='upper left')

        # Test duration
        fig.text(x=0.5, y=0.75, s='Test Duration', ha='center', va='bottom')
        fig.text(x=0.5, y=0.75, s=duration_str, ha='center', va='top', color=div(duration / 3600), size='xx-large')

        # Online time
        fig.text(x=0.85, y=0.75, s='Online Time', ha='center', va='bottom')
        fig.text(x=0.85, y=0.75, s=f'{int(online * 100)} %', ha='center', va='top', color=div(online), size='xx-large')

        # Save figure
        plt.savefig(file_name, bbox_inches='tight')


if __name__ == '__main__':
    net = NetworkTest()
    net.connection_test_interval(minutes=1)
    net.export_connection_results('connection.csv')
    net.report_connection('connection.png')
    net.speed_test_interval(minutes=1)
    net.export_speed_results('speed.csv')
    net.report_speed('speed.png')
    print('Done!')
