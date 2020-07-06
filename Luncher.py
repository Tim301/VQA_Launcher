import sys, os, json, subprocess
from pathlib import Path

def get_input(input_path):
    workdir = Path(input_path)
    ref = list(workdir.glob('*.mxf'))  #List .mxf filepath in a array (pathlib type)
    comps= []
    trp_path = list(workdir.glob('*.trp'))[0] #Get trp filepath

    #Extract programs info into a json
    json_info = subprocess.run(['ffprobe', '-show_programs', '-of', 'json', '-v', 'quiet', '-i', trp_path], stdout=subprocess.PIPE) #Probe trp and pipe results
    json_info = json_info.stdout.decode('utf-8') #Just in case
    trp_info = json.loads(json_info) #Parse json into an object

    #Get all video steam id 
    stream_id = []
    for prg in trp_info['programs']:
        for strm in prg['streams']:
            if (strm['codec_type'] == 'video'):
                stream_id.append(strm['index'])
                break

    #Extract ts from trp
    for i, id in enumerate(stream_id):
        index = "0:" + str(id)
        pathout = os.fspath(workdir.cwd()) + "\comps_" + str(i+1) + ".ts" #Must use os.fspath() to convert pathlib type into str
        subprocess.run(['ffmpeg', '-y', '-nostats', '-loglevel', '0', '-i', trp_path, '-c:v', 'copy', '-map', index, pathout]) #Subprocess only accept str as type path
        comps.append(Path(pathout)) #Converts str path into pathlib type and adds it in comps[]
    
    print(ref)
    print(comps)
    
    #Lunch SET_ASYNC.exe for each programs
    for program, _ in enumerate(ref):
        print(ref[program])
        print(comps[program])
        #subprocess.run(['SET_ASYNC.exe', ref[program], comps[program]])

input_path = sys.argv[1]
get_input(input_path)