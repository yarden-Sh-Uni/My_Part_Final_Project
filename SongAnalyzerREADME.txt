Read me - SongAnalyzer UI

Files:

song_analyzer.py - holds all our code and logics for the analysis
class_tester.py - an easy to use with exmaple the two main function of the analyzer
interface_flow_test.py - UI


------How to run the UI?---------

Run the script:
"python .\interface_flow_test.py --track_path <track to analyze> <other arguments optional-see below>"

parser.add_argument('--scale', type=str,default='Relative', help='Scale to use') # scales Relative, Normal, Percentile
parser.add_argument('--min_seg_length_sec', type=int,default=5, help="Min Segment length in sec")
parser.add_argument('--max_seg_length_sec', type=int,default=40, help="Max Segment length in sec")
parser.add_argument('--instrument', type=str,default='main', help="Track's instrument")
parser.add_argument('--energy_levels', type=int,default=5, help="Number of levels")
parser.add_argument('--resolution', type=int,default=0.1, help="Display Resolution in sec")

In your browser go to "http://127.0.0.1:5000/"

After the code is running upload the audio file at "Load Audio"


---- Seperate Testing --------------
In the "class_tester.py" note to main funcitons:
1.run - full song analysis option to include speration
2.single_track_run - single track analysis
