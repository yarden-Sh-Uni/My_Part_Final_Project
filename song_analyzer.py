import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


class SongAnalyzer:
    def     __init__(self, min_seg_length_sec=5, max_seg_length_sec=40, min_width_sec=1,
                 number_of_levels=5, sensitivity_power=0.001, scale=None,normalize_values=False):
        self.min_seg_length_sec = min_seg_length_sec
        self.max_seg_length_sec = max_seg_length_sec
        self.min_width_sec = min_width_sec
        self.number_of_levels = number_of_levels
        self.sensitivity_power = sensitivity_power
    
        self.standard_values = {'bass':{'mu':0.038015, 'sigma':0.073480},
                                'drums':{'mu':0.028142, 'sigma':0.07326},
                                'vocals':{'mu':0.022812, 'sigma':0.049614},
                                'other':{'mu':0.037464, 'sigma':0.053378},
                                'main':{'mu':0.99669, 'sigma':1.229977}}
        self.scale = scale  # scales Relative, Normal, Percentile


        self.percentile_scale = {'bass': [0, 0.001157284, 0.001429282, 0.0018345797, 0.0023262554, 0.0028819158, 0.003494828, 0.0041640764, 0.004889756, 0.005669035, 0.0065031713, 0.007396182, 0.008347249, 0.00936502, 0.010437891, 0.011478838, 0.012122021, 0.012525667, 0.012854439, 0.013725911, 0.014990378, 0.016128056, 0.016993368, 0.017975546, 0.01951725, 0.021167342, 0.022897968, 0.024711492, 0.026596598, 0.028407376, 0.030470818, 0.032670952, 0.035004966, 0.037464187, 0.040052935, 0.042767517, 0.045635212, 0.04864127, 0.051818892, 0.055170633, 0.058723874, 0.06250961, 0.06657836, 0.07097216, 0.07580591, 0.08123309, 0.08738546, 0.0944075, 0.10260611, 0.11226575, 0.12392988, 0.1383674, 0.15509665, 0.17369953, 0.19332942, 0.21378435, 0.23563465, 0.2617035, 0.29739547, 0.35735264],
                                 'drums': [0, 0.0011123009, 0.0012339418, 0.001364873, 0.0015052524, 0.0016557821, 0.0018167943, 0.0019888014, 0.0021723525, 0.002368551, 0.0025787286, 0.002803017, 0.0030435654, 0.0033021488, 0.0035786494, 0.003876434, 0.004197322, 0.004543467, 0.0049170144, 0.0053222463, 0.005761604, 0.006238776, 0.0067598377, 0.0073265918, 0.007943939, 0.008612922, 0.009331483, 0.010083754, 0.0108035905, 0.011485541, 0.01204529, 0.0126626035, 0.013641676, 0.014733557, 0.015765801, 0.0168445, 0.018021353, 0.019478343, 0.02123136, 0.023187503, 0.025358694, 0.027759768, 0.030481339, 0.03358966, 0.037142046, 0.041228537, 0.04593939, 0.0514256, 0.057838745, 0.0654069, 0.07440406, 0.08513538, 0.09811039, 0.11407283, 0.13380778, 0.15914251, 0.19285963, 0.23914841, 0.30612633, 0.4133965],
                                 'vocals': [0, 0.0012548936, 0.0015783021, 0.0019592657, 0.0023963875, 0.0028873365, 0.0034296438, 0.0040250197, 0.0046678293, 0.005357623, 0.0060857506, 0.006843269, 0.0076302644, 0.008455195, 0.0092970375, 0.0101150405, 0.010881178, 0.011552881, 0.012218023, 0.012842127, 0.01334709, 0.014066377, 0.01473777, 0.015384736, 0.016064553, 0.017199371, 0.018482659, 0.019868195, 0.021359447, 0.022930874, 0.024597853, 0.026341362, 0.028200842, 0.030187126, 0.032328904, 0.034610085, 0.037031814, 0.03959623, 0.042316034, 0.04519497, 0.048245464, 0.051488522, 0.05493871, 0.05860678, 0.06252931, 0.06673269, 0.07125724, 0.07614963, 0.0814816, 0.08731328, 0.093744785, 0.100923575, 0.1090139, 0.11826754, 0.1290575, 0.14195952, 0.15802476, 0.17919424, 0.20944917, 0.26178294],
                                 'other': [0, 0.0012159692, 0.0014660307, 0.0017518019, 0.0020739958, 0.0024313275, 0.0028228816, 0.0032472152, 0.0037045958, 0.0041945763, 0.0047165235, 0.0052709244, 0.0058576637, 0.006478266, 0.007131314, 0.007820956, 0.008546062, 0.009294467, 0.010075849, 0.010915624, 0.011801757, 0.012735426, 0.013714738, 0.014744898, 0.015827501, 0.01696292, 0.018159458, 0.019416416, 0.020735618, 0.022122527, 0.023577422, 0.025111604, 0.026720988, 0.02841707, 0.030204238, 0.032086622, 0.03407658, 0.03617948, 0.038408276, 0.040762484, 0.043262158, 0.04591789, 0.048760142, 0.051806416, 0.055079475, 0.05860917, 0.062425993, 0.06657897, 0.07113121, 0.07614411, 0.08170158, 0.08793407, 0.09499398, 0.10310994, 0.11261358, 0.12400356, 0.1381113, 0.15654816, 0.18288808, 0.22793823],
                                 'main': [0, 0.0015581478, 0.0021928577, 0.0028938802, 0.003654058, 0.004469499, 0.0053401613, 0.0062646656, 0.0072414503, 0.008275367, 0.009369561, 0.010528434, 0.011754644, 0.013045382, 0.014407493, 0.015844528, 0.017361723, 0.01896102, 0.02064693, 0.02241943, 0.024284257, 0.02624285, 0.028305732, 0.03047591, 0.0327631, 0.035173804, 0.03771267, 0.04038921, 0.043206654, 0.046189725, 0.04933335, 0.05266592, 0.056192547, 0.059925035, 0.06389488, 0.06811552, 0.072609946, 0.077397704, 0.0825332, 0.08803107, 0.09392966, 0.10028723, 0.10715951, 0.11459597, 0.12266302, 0.13141489, 0.14096773, 0.1513802, 0.16277456, 0.17529559, 0.1890661, 0.20433073, 0.22137704, 0.24073666, 0.26311758, 0.28957248, 0.3222283, 0.36454862, 0.42488298, 0.52815664]
                                 }
        self.sample_rate = None
        self.ref_values = None
        self.normalize_values = normalize_values

    def run(self, song_name, show_sub_plots=False, separate=False, show_port=True):
        command = f"demucs --mp3 {song_name}.mp3"
        if separate:
            # Execute the command
            os.system(command)
        instruments = ['bass', 'drums', 'vocals', 'other', 'main']
        tracks = [f"separated/htdemucs/{song_name}/{inst}.mp3" if inst != "main" else f"{song_name}.mp3"
                  for inst in instruments]

        quantize_tracks = {}
        instruments_segments = {}
        for i, track in enumerate(tracks):
            y, self.sample_rate = librosa.load(track)
            y_power = np.abs(y)

            min_seg_length = self.sample_rate * self.min_seg_length_sec
            max_seg_length = self.sample_rate * self.max_seg_length_sec
            min_width = self.sample_rate * self.min_width_sec

            print(f"Processing {instruments[i]} - shape: {y.shape}, sample rate: {self.sample_rate}")
            print(f"Minimum  value : {y_power.min()}")
            print(f"Maximum  value : {y_power.max()}")

            self.ref_values = self.get_quantizer_vals(values=y_power, instrument=instruments[i])

            # Process segments and quantization
            segments = {}
            q = []

            s = self.remove_opening_silence(powers=y_power,instrument=instruments[i])
            e = s + min_seg_length
            q.extend([0] * s)

            while e < y_power.shape[0]:
                left, right = min_seg_length + s, max_seg_length + s
                section_power = self.quantize_form_list(y_power[s:s + min_seg_length].mean(), ref_list=self.ref_values)
                while left <= right:
                    mid = (left + right) // 2
                    if (self.quantize_form_list(np.median(y_power[mid - (min_width // 2):mid + (min_width // 2)]),
                                           ref_list=self.ref_values)
                        == section_power) and (
                    self.check_constant_range(values=y_power[s:mid + (min_width // 2)], val=section_power, step=min_width,
                                         ref=self.ref_values)):
                        left = mid + 1
                    else:
                        right = mid - 1

                mid = (s + e) // 2
                e = mid

                power = self.quantize_form_list(value=y_power[s:e].mean(),ref_list=self.ref_values)
                q.extend([power] * (e - s))
                segments[len(segments)] = {"start": self.get_sec_from_sr(s),
                                           "end": self.get_sec_from_sr(e),
                                           "power": power}
                s = e
                e = e + min_seg_length

            quantize_tracks[instruments[i]] = q

            instruments_segments[instruments[i]] = segments
            if show_sub_plots:
                # Match q length to y_power by zero padding
                if len(q) > y_power.shape[0]:
                    q = q[:-(len(q) - y_power.shape[0])]
                elif len(q) < y_power.shape[0]:
                    q.extend([0] * (y_power.shape[0] - len(q)))
                else:
                    pass

                y_norm = y_power / (((self.number_of_levels + 1) / 2) * y_power.mean())
                data = {
                    'x': range(y_power.shape[0]),
                    'Signal': y_norm,
                    'Quantize Signal': q
                }
                df = pd.DataFrame(data)
                df.plot(x='x', y='Signal', label='OrgSignal ', legend=True)
                df.plot(x='x', y='Quantize Signal', label='Quantize Signal', legend=True,
                        ax=plt.gca())  # Use the same axes
                ticks = self.get_ticks(length=y_power.shape[0])
                positions, labels = ticks
                plt.xticks(positions, labels)
                plt.title(f"{instruments[i]} - Quantize Signal Vs Original")
                plt.xlabel("Samples")
                plt.ylabel("Power")
                plt.grid(True)
                plt.show()

        if show_port:
            self.plot_quantized_tracks(quantize_tracks,  instruments)

        return instruments_segments

    def single_track_run(self, track, instrument, show_plot=True, segments_print=False):
        y, self.sample_rate = librosa.load(track)
        y_power = np.abs(y)

        min_seg_length = self.sample_rate * self.min_seg_length_sec
        max_seg_length = self.sample_rate * self.max_seg_length_sec
        min_width = self.sample_rate * self.min_width_sec

        print(f"Processing {instrument} - shape: {y.shape}, sample rate: {self.sample_rate}")
        print(f"Minimum  value : {y_power.min()}")
        print(f"Maximum  value : {y_power.max()}")

        self.ref_values = self.get_quantizer_vals(values=y_power, instrument=instrument)

        # Process segments and quantization
        segments = {}
        q = []

        s = self.remove_opening_silence(powers=y_power, instrument=instrument)
        e = s + min_seg_length
        q.extend([0] * s)

        while e < y_power.shape[0]:
            left, right = min_seg_length + s, max_seg_length + s
            section_power = self.quantize_form_list(y_power[s:s + min_seg_length].mean(), ref_list=self.ref_values)
            while left <= right:
                mid = (left + right) // 2
                if (self.quantize_form_list(np.median(y_power[mid - (min_width // 2):mid + (min_width // 2)]),
                                            ref_list=self.ref_values)
                    == section_power) and (
                        self.check_constant_range(values=y_power[s + min_seg_length:mid + (min_width // 2)], val=section_power,
                                                  step=min_width,
                                                  ref=self.ref_values)):
                    left = mid + 1
                else:
                    right = mid - 1

            mid = (s + e) // 2
            e = mid

            power = self.quantize_form_list(value=y_power[s:e].mean(), ref_list=self.ref_values)
            q.extend([power] * (e - s))
            segments[len(segments)] = {"start": self.get_sec_from_sr(s),
                                       "end": self.get_sec_from_sr(e),
                                       "power": power}
            s = e
            e = e + min_seg_length
        if segments_print:
            segments_cont = {}
            start = 0
            for i,sample in enumerate(q):
                if q[start] != sample:
                    segments_cont[len(segments_cont)] = {"start": self.get_sec_from_sr(start),
                                                         "end": self.get_sec_from_sr(i-1),
                                                         "power": q[start]}
                    start = i

            print(segments_cont)


        if show_plot:
            # Match q length to y_power by zero padding
            if len(q) > y_power.shape[0]:
                q = q[:-(len(q) - y_power.shape[0])]
            elif len(q) < y_power.shape[0]:
                q.extend([0] * (y_power.shape[0] - len(q)))
            else:
                pass

            y_norm = y_power / (((self.number_of_levels + 1) / 2) * y_power.mean())
            data = {
                'x': range(y_power.shape[0]),
                'Signal': y_norm,
                'Quantize Signal': q
            }
            df = pd.DataFrame(data)
            df.plot(x='x', y='Signal', label='OrgSignal ', legend=True)
            df.plot(x='x', y='Quantize Signal', label='Quantize Signal', legend=True,
                    ax=plt.gca())  # Use the same axes
            ticks = self.get_ticks(length=y_power.shape[0])
            positions, labels = ticks
            plt.xticks(positions, labels)
            plt.title(f"{instrument} - Quantize Signal Vs Original")
            plt.xlabel("Samples")
            plt.ylabel("Power")
            plt.grid(True)
            plt.show()

        return q

    def get_sec_from_sr(self, sample):
        return round(sample / self.sample_rate, 3)

    def calculate_standard_scale(self, instrument):
        """
        Generate a scale based on a normal distribution.

        Parameters:
        mu (float): Mean of the normal distribution
        sigma (float): Standard deviation of the normal distribution

        Returns:
        list: A sorted list of values representing the scale
        """
        mu = self.standard_values[instrument]['mu']
        sigma = self.standard_values[instrument]['sigma']
        deviations = np.linspace(-2, 2, self.number_of_levels)  # Adjust the range as needed
        scale_values = mu + deviations * sigma

        print(list(scale_values))

        return list(scale_values)

    def get_quantizer_vals(self, values, instrument):
        if self.scale == 'Normal':
            return self.calculate_standard_scale(instrument=instrument)
        elif self.scale == 'Percentile':
            step = len(self.percentile_scale[instrument])//self.number_of_levels
            return self.percentile_scale[instrument][::step]
        sorted_values = np.sort(values)
        index = self.bin_search_index(sorted_values, self.sensitivity_power)
        step = (len(values) - index) // self.number_of_levels
        ret_arr = [0]
        for i in range(1, self.number_of_levels):
            ret_arr.append(sorted_values[index + (i * step)])
        return ret_arr

    def quantize_form_list(self, value, ref_list):
        for i, ref in enumerate(ref_list):
            if value < ref:
                if self.normalize_values:
                    return (i-1)/self.number_of_levels
                return i - 1
        if self.normalize_values:
            return 1
        return self.number_of_levels

    def remove_opening_silence(self, powers, instrument):
        step = int(0.5 * self.sample_rate)
        step_w = int(self.min_width_sec * self.sample_rate)
        sample = step_w // 2
        while self.quantize_form_list(np.median(powers[sample - (step_w // 2):sample + (step_w // 2)]),
                                      ref_list=self.get_quantizer_vals(values=powers, instrument=instrument)) == 0:
            sample += step
        return sample

    def find_closing_silence(self, powers, instrument):
        step = int(0.5 * self.sample_rate)
        step_w = int(self.min_width_sec * self.sample_rate)
        sample = powers.shape[0] - step_w // 2
        while self.quantize_form_list(np.median(powers[sample - (step_w // 2):sample + (step_w // 2)]),
                                      ref_list=self.get_quantizer_vals(values=powers, instrument=instrument)) == 0:
            sample -= step
        return sample

    def check_constant_range(self, values, val, step, ref):
        i = step // 2
        while i < len(values):
            if self.quantize_form_list(np.median(values[i - (step // 2):i + (step // 2)]), ref_list=ref) != val:
                return False
            i += step
        return True

    @staticmethod
    def bin_search_index(values, ref):
        right = len(values)
        left = 0
        mid = (right + left) // 2
        while right > left:
            mid = (right + left) // 2
            if values[mid] < ref:
                left = mid + 1
            else:
                right = mid - 1
        return mid

    @staticmethod
    def plot_array(array, title="Plot of Array", x_ticks=None, log_scale=False):
        if not isinstance(array, np.ndarray) and not isinstance(array, list):
            raise TypeError("Input must be a numpy array or list.")

        plt.plot(array)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Value dB")
        if log_scale:
            plt.yscale("log")
        plt.grid(True)
        if x_ticks:
            positions, labels = x_ticks
            plt.xticks(positions, labels)
        plt.show()

    def plot_quantized_tracks(self, quantize_tracks, instruments):
        number_of_samples = max(len(quantize_tracks[i]) for i in instruments)

        # Pad tracks with zeros
        for instrument in instruments:
            if len(quantize_tracks[instrument]) < number_of_samples:
                quantize_tracks[instrument].extend([0] * (number_of_samples - len(quantize_tracks[instrument])))

        # Plotting results
        data = {'x': range(number_of_samples)}
        for instrument in instruments:
            data[instrument] = quantize_tracks[instrument]

        df = pd.DataFrame(data)
        fig, axes = plt.subplots(nrows=len(instruments), ncols=1, figsize=(6, 15))

        ticks = self.get_ticks(number_of_samples)
        positions, labels = ticks

        for i, ax in enumerate(axes):
            df.plot(x='x', y=instruments[i], ax=ax, title=f'{instruments[i]}')
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)

        plt.tight_layout()
        plt.ylabel('Power')
        plt.xlabel('Time')
        plt.show()

    def get_ticks(self, length):
        positions = []
        labels = []
        i = 0
        while i < length:
            positions.append(i)
            seconds = i // self.sample_rate
            labels.append(f"{seconds // 60}:{seconds % 60}")
            i += self.sample_rate * 10

        return positions, labels
