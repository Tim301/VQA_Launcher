#!/usr/bin/python3

import sys, os, json, time, subprocess, re, threading
from pathlib import Path
import shutil

def threadVQA(arg, i, key):
    global time
    #os.system('./VQA2 ' + arg)
    process = subprocess.run(['./VQA2 ' + arg], text=True, shell=True, capture_output=True)
    print("Thread " +str(i)+  " done in " + str(time.time() - start) + " s.")
    output = "################## "+ str(i) + " " + key +" ##################\n" + "Arg: "+ arg + "\n"+ process.stderr + process.stdout
    f = open("log.txt", "a")
    f.write(output)
    f.close()

def get_input(input_path):
    workdir = Path(input_path)
    ref = list(workdir.glob('*.mxf'))  #List .mxf filepath in a array (pathlib type)
    comps= []
    converted = []
    trp_path = list(workdir.glob('*.trp'))[0] #Get trp filepath
    #print(ref)
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

    comps_name=["service_1","service-2","service_3","service_4","service_5","service_6"]

    #Extract ts from trp
    for i, id in enumerate(stream_id):
        index = "0:" + str(id)
        pathout = os.fspath(workdir.cwd()) + "/TS/"+ comps_name[i] + ".ts" #Must use os.fspath() to convert pathlib type into str
        #subprocess.run(['ffmpeg', '-y', '-nostats', '-loglevel', '0','-i',trp_path, '-c:v', 'copy', '-map', index, pathout]) #Subprocess only accept str as type path
        subprocess.run(['ffmpeg', '-y', '-nostats', '-loglevel', '0','-ss', '00:00:05.00','-i',trp_path, '-c:v', 'copy', '-map', index, pathout]) #Subprocess only accept str as type path
        comps.append(Path(pathout)) #Converts str path into pathlib type and adds it in comps[]

    #Dictionary to map ref mxf with services from trp
    pidmap = {}
    pidmap["ref_1"] = [0,4]
    pidmap["ref_2"] = [1]
    pidmap["ref_3"] = [2]
    pidmap["ref_4"] = [3]
    pidmap["ref_5"] = [4]
    pidmap["ref_6"] = [5]

    #Lunch VQA.exe for each programs
    counter = 1
    thread_list = []
    for key in pidmap.keys():
        indexref = [i for i, elem in enumerate(ref) if key in str(os.fspath(elem))] # Search pidmap key in refpath
        #print("index: " )
        #print(indexref)
        refpath = str(os.fspath(ref[indexref[0]]))
        listcomps = []
        for indexcomp in pidmap[key]:
            listcomps.append(str(os.fspath(comps[indexcomp]))) # Match ref with comps though pidmap
        #print(listcomps)
        argjson = json.dumps({ "ref": refpath, "window": 300, "comps": listcomps })
        argjson = argjson.replace(' ', '\\ ').replace(',','\,').replace('"', '\\"').replace('\n', '\\n') # Convert json to escaped string
        print( "Start thread: " + str(counter) + "/" + str(len(pidmap)))
        threading.Thread(target=threadVQA,  args=(argjson,counter,key)).start()
        counter = counter + 1
        #print(argjson)

start = time.time()
if not (os.path.exists('./TS/')):
    os.system('mkdir TS')
    print('TS folder created')
os.system('touch log.txt')
os.system('cp /home/labo/Documents/Timothee/QT/build-VQA2-Desktop_Qt_5_15_0_GCC_64bit-Debug/VQA2 .')  #Copy VQA into current dir
if len(sys.argv) < 2:
    input_path = os.getcwd()
else:
    input_path = sys.argv[1]
print("Working dir:" + input_path)
os.chdir(input_path)
get_input(input_path)

