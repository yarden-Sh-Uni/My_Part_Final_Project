from song_analyzer import SongAnalyzer

src_track = "imagine_dragons_radioactive.mp3"
song_name = "Mr.Brightside_The_Killers"
instrument = "main"

analyzer = SongAnalyzer(normalize_values=False, scale='Percentile')
# quantized_values = analyzer.single_track_run(track=src_track, instrument=instrument,
#                                                  show_plot=True, segments_print=True)

output=analyzer.run(song_name=song_name, show_port=False)

print(output)
# print(len(quantized_values))
# print(max(quantized_values))
# print(min(quantized_values))
# print(sum(quantized_values)/len(quantized_values))
