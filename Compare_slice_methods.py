
import librosa
import numpy as np
import matplotlib.pyplot as plt

def bin_search_index(values, ref=0.001):
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

def get_quantizer_vals(values,number_of_levels=10):
    sorted_values = np.sort(values)
    index = bin_search_index(sorted_values)
    step = (len(values) - index) // number_of_levels
    ret_arr = [0]
    for i in range(1, number_of_levels):
        ret_arr.append(sorted_values[index + (i * step)])
    return ret_arr

def quantize_form_list( value, ref_list):
    for i, ref in enumerate(ref_list):
        if value < ref:
            return i - 1
    return len(ref_list)


song_name = "Sweet_Boy"
instruments = ['bass', 'drums', 'vocals', 'other_instruments', 'full_track']

mp3_files = [

    f'separated/htdemucs/{song_name}/bass.mp3',
    f'separated/htdemucs/{song_name}/drums.mp3',
    f'separated/htdemucs/{song_name}/vocals.mp3',
    f'separated/htdemucs/{song_name}/other.mp3',
    f'{song_name}.mp3'
]

for file in mp3_files:
    y, sr = librosa.load(file, sr=None)  # y: waveform, sr: sample rate

    # Build an avg / mid value
    steps = sr // 2

    y_power = np.abs(y)
    length = (len(y_power) // steps)

    y_mid = np.zeros(length)
    for k in range(length):
        y_mid[k] = np.median(y_power[k * steps:(k + 1) * steps])
    reference_list = get_quantizer_vals(y_mid)
    for j, sample in enumerate(y_mid):
        y_mid[j] = quantize_form_list(sample, reference_list)

    y_avg = np.zeros(length)
    for k in range(length):
        y_avg[k] = np.mean(y_power[k * steps:(k + 1) * steps])
    # Quantize Values
    reference_list = get_quantizer_vals(y_avg)
    for j, sample in enumerate(y_avg):
        y_avg[j] = quantize_form_list(sample, reference_list)

    y_max = np.zeros(length)
    for k in range(length):
        y_max[k] = np.max(y_power[k * steps:(k + 1) * steps])
    reference_list = get_quantizer_vals(y_max)
    for j, sample in enumerate(y_max):
        y_max[j] = quantize_form_list(sample, reference_list)

    y_rms = np.zeros(length)
    for k in range(length):
        y_rms[k] = np.sqrt(np.mean((y_power[k * steps:(k + 1) * steps])))
    reference_list = get_quantizer_vals(y_rms)
    for j, sample in enumerate(y_rms):
        y_rms[j] = quantize_form_list(sample, reference_list)

    y_sliced =y_power[::steps]
    reference_list = get_quantizer_vals(y_sliced)
    for j, sample in enumerate(y_sliced):
        y_sliced[j] = quantize_form_list(sample, reference_list)


    x = np.linspace(start=0,stop=length,num=length)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Plot each array in its own subplot
    axes[0].plot(x, y_sliced[:-1], color='blue')
    axes[0].set_title('Sliced')

    axes[1].plot(x, y_avg, color='red')
    axes[1].set_title('Average')

    axes[2].plot(x, y_mid, color='green')
    axes[2].set_title('Median')

    axes[3].plot(x, y_rms, color='purple')
    axes[3].set_title('RMS')

    axes[4].plot(x, y_max, color='yellow')
    axes[4].set_title('Max')


    # Add overall labels
    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # Adjust layout so titles/labels don’t overlap
    plt.tight_layout()
    plt.show()