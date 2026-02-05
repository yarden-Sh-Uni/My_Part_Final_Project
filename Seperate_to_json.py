from operator import length_hint

import librosa
import json
import numpy as np
import os



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


song_name = "Mr.Brightside_The_Killers"
instruments = ['bass', 'drums', 'vocals', 'other_instruments', 'full_track']

methods = ["Sliced" , "Average" ,"Median", "Max", "RMS"]

method = "RMS"

# Separate Demucs
separate = False
command = f"demucs --mp3 {song_name}.mp3"
if separate:
    # Execute the command
    os.system(command)

# Output directory for CSV files
mp3_files = [

    f'separated/htdemucs/{song_name}/bass.mp3',
    f'separated/htdemucs/{song_name}/drums.mp3',
    f'separated/htdemucs/{song_name}/vocals.mp3',
    f'separated/htdemucs/{song_name}/other.mp3',
    f'{song_name}.mp3'
]
# Initialize the structure
audio_data = {"Instrument_Loudness_per_Sec": [], "Song_Parts_with_Timestamps": {}}

# Paress input file
audio_data["Song_Parts_with_Timestamps"] = \
    {
        0:{"Start_Timestamp":0,"Segments":"Intro","Type":"Vocals", "Lyrics":"One, two, three, four"},
        1:{"Start_Timestamp":3,"Segments":"Chorus","Type":"Vocals", "Lyrics":"Can we go home now? It's getting later,"
                                                                             " baby Can we go home now? You think it's "
                                                                             "time to give up We're on our own now "
                                                                             "No place to drive you crazy "
                                                                             "Don't share a home now I'm okay 'til tonight"},
        2:{"Start_Timestamp":26,"Segments":"Post-Chorus","Type":"Vocals", "Lyrics":"Ties get broken"
                                                                                   " You need space and you're not mad "
                                                                                   "when I'm not late Now I'm not there and "
                                                                                   "you're just fine But I'm okay 'til tonight"},
        3:{"Start_Timestamp":46,"Segments":"Verse 1","Type":"Vocals", "Lyrics":"I want you to go I wrote it in bold Yeah, "
                                                                               "I got a plan And you can't be involved "
                                                                               "And when it gets great Don't start to get mad "
                                                                               "'Cause I'ma lose time When you are free to have"
                                                                               " Flying out to the city They're throwing some money"
                                                                               " at me I know it's been shitty To see the attention"
                                                                               " I'm getting But it's been coming Since I've been singing,"
                                                                               " baby (Singing, baby) Now I'll just see you when I'm on business"
                                                                               " With everyone you can listen You can write from where you are living"},
        4:{"Start_Timestamp":90,"Segments":"Chorus","Type":"Vocals", "Lyrics":"Can we go home now? It's getting later,"
                                                                              " baby Can we go home now? "
                                                                              "You think it's time to give up"
                                                                              " We're on our own now "
                                                                              "No place to drive you crazy "
                                                                              "Don't share a home now I'm okay 'til tonight"},
        5:{"Start_Timestamp":113,"Segments":"Post-Chorus","Type":"Vocals", "Lyrics":"Ties get broken,"
                                                                                    " you need space And you're not mad when "
                                                                                    "I'm not late now I'm not there and you're just fine "
                                                                                    "But I'm okay 'til tonight"},
        6:{"Start_Timestamp":134,"Segments":"Coda","Type":"instrumental", "Lyrics":""}
    }

for i,file in enumerate(mp3_files):
    try:
        # Load audio file using librosa
        y, sr = librosa.load(file, sr=None)  # y: waveform, sr: sample rate

        # Build an avg / mid value
        steps = sr //2

        y_power = np.abs(y)
        length = len(y_power) // steps

        if method == "Median":
            y_mid = np.zeros(length)
            for k in range(length):
                y_mid[k] = np.median(y_power[k*steps:(k+1)*steps])
            reference_list = get_quantizer_vals(y_mid)
            y_norm = y_mid

        elif method == "Average":
            y_avg = np.zeros(length)
            for k in range(length):
                y_avg[k] = np.mean(y_power[k*steps:(k+1)*steps])
            reference_list = get_quantizer_vals(y_avg)
            y_norm = y_avg

        elif method == "Sliced":
            y_sliced = y_power[::steps]
            reference_list = get_quantizer_vals(y_sliced)
            y_norm = y_sliced

        elif method == "Max":
            y_max =np.zeros(length)
            for k in range(length):
                y_max[k] = np.max(y_power[k * steps:(k + 1) * steps])
            reference_list = get_quantizer_vals(y_max)
            y_norm = y_max

        elif method == "RMS":
            y_rms = np.zeros(length)
            for k in range(length):
                y_rms[k] = np.sqrt(np.mean((y_power[k * steps:(k + 1) * steps])))
            reference_list = get_quantizer_vals(y_rms)
            y_norm = y_rms

        else:
            print(f"Method {method} not implemented")
            y_sliced = y_power[::steps]
            reference_list = get_quantizer_vals(y_sliced)
            y_norm = y_sliced


        for j,sample in enumerate(y_norm):
            y_norm[j] = quantize_form_list(sample,reference_list)


        # Convert waveform to list (truncate for JSON size if needed)
        features = y_norm.tolist()

        # Add to structure
        audio_data["Instrument_Loudness_per_Sec"].append({
            "instrument": instruments[i],
            "levels": features
        })
    except Exception as e:
        print(f"Could not process {file}: {e}")

# Save to JSON file
with open(f"{song_name}_{method}_data.json", "w") as json_file:
    json.dump(audio_data, json_file, indent=4)

print(f"JSON file '{song_name}_{method}_data.json' created successfully with audio features.")

# Push to LLM

# Print Results
