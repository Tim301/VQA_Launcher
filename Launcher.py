#!/usr/bin/python3

import sys, os, json, time, subprocess, re, threading
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

def threadVQA(arg, i, key):
    global time
    process = subprocess.run(['./VQA2 ' + arg], text=True, shell=True, capture_output=True)
    print("Thread " +str(i)+  " done in " + str(int(time.time() - start)) + " s.")
    output = "################## "+ str(i) + " " + key +" ##################\n" + "Arg: "+ arg + "\n"+ process.stderr + process.stdout
    f = open("log.txt", "a")
    f.write(output)
    f.close()

def printtime():
    global do_print
    while do_print:
        tmp = int(time.time() - start)
        work_time =time.strftime('%H:%M:%S', time.gmtime(tmp))
        print("Work in progress since " + work_time , end='\r')
        time.sleep(1)

def get_input(input_path):
    workdir = Path(input_path)
    ref = list(workdir.glob('*.mxf'))  #List .mxf filepath in a array (pathlib type)
    comps= []
    converted = []
    trp_path = list(workdir.glob('*.trp'))[0] #Get trp filepath
    print('Trp found: ' + os.fspath(trp_path))
    #Extract programs info into a json
    json_info = subprocess.run(['ffprobe', '-show_programs', '-of', 'json', '-v', 'quiet', '-i', trp_path], stdout=subprocess.PIPE) #Probe trp and pipe results
    json_info = json_info.stdout.decode('utf-8') #Just in case
    trp_info = json.loads(json_info) #Parse json into an object

    #Get all video steam id
    d = open('debug_programs.csv', "a")
    os.system('echo > debug_programs.csv')
    debug_text = "Programs_id;index_id;\n"
    stream_id = []
    for prg in trp_info['programs']:
        for strm in prg['streams']:
            if (strm['codec_type'] == 'video'):
                stream_id.append(strm['index'])
                break
        debug_text = debug_text + str(prg['program_id']) + ";" + str(strm['index']) + ";\n"
    d.write(debug_text)
    print(str(len(stream_id)) + " programs found")

    #Load config by json
    t = open("../config.json", "r")
    init_json = json.loads(t.read())
    comps_name=init_json["service_names"]

    print("Start extraction")
    #Extract ts from trp
    for i, id in enumerate(stream_id):
        index = "0:" + str(id)
        pathout = os.fspath(workdir.cwd()) + "/TS/"+ comps_name[i] + ".ts" #Must use os.fspath() to convert pathlib type into str
        subprocess.run(['ffmpeg', '-y', '-nostats', '-loglevel', '0','-ss', '00:00:05.0001','-i',trp_path, '-c:v', 'copy', '-map', index, pathout]) #Subprocess only accept str as type path
        #subprocess.run(['ffmpeg', '-y', '-nostats', '-loglevel', '0', '-i',trp_path, '-c:v', 'copy', '-map', index, pathout, "-copy_unknown"]) #Subprocess only accept str as type path
        comps.append(Path(pathout)) #Converts str path into pathlib type and adds it in comps[]
        print(str(i + 1) + "/" + str(len(stream_id)) + " programs extracted")

    #Dictionary to map ref mxf with services from trp
    pidmap = init_json["pidmap"]

    R_comps_name=[]
    listcsv = {}

    #Part I Start analyse with VQA + ffmpeg
    counter = 1
    threads = []
    for key in pidmap.keys():
        indexref = [i for i, elem in enumerate(ref) if key in str(os.fspath(elem))] # Search pidmap key in refpath
        #print("index: " )
        #print(indexref)
        refpath = str(os.fspath(ref[indexref[0]]))
        listcomps = []
        csvmap = []
        for indexcomp in pidmap[key]:
            csvmap.append(comps_name[indexcomp])
            listcomps.append(str(os.fspath(comps[indexcomp]))) # Match ref with comps though pidmap
        tmpcsv = {key:csvmap}
        listcsv.update(tmpcsv)
        #print(listcomps)
        argjson = json.dumps({ "ref": refpath, "window": 2500, "comps": listcomps })
        argjson = argjson.replace(' ', '\\ ').replace(',','\,').replace('"', '\\"').replace('\n', '\\n') # Convert json to escaped string
        print( "Start thread: " + str(counter) + "/" + str(len(pidmap)))
        t = threading.Thread(target=threadVQA,  args=(argjson,counter,key))
        threads.append(t)
        counter = counter + 1
        #print(argjson)

    show_time = threading.Thread(target=printtime)
    #Start threads
    for x in threads:
        x.start()

    show_time.start()

    #Wait end of all threads
    for x in threads:
        x.join()
    do_print = False


    #show_time.terminate()

    print("All threads completed")

    #Part II formating log file into csv
    print("Export log to csv")
    #print(listcsv)
    loglist = list(workdir.glob('*.log'))
    for csv in listcsv.keys():
        #print("##############")
        indexlog = []
        key_list = []
        #print(csv)
        for value in listcsv[csv]:
            #print(value)
            key_list.append(value)
            indexlog.append([i for i, elem in enumerate(loglist) if value in str(os.fspath(elem))]) # Search pidmap key in refpath

        #print(indexlog)
        csv_values = []
        csv_dict = {}
        for counter, index in enumerate(indexlog):
            #print("len: " + str(len(index)))
            name_dict = {}
            for i in index:
                path = os.fspath(loglist[i])
                if "PSNR" in path:
                    #print("PSNR: " + path)
                    csvfile = pd.read_csv(path, delimiter='\s+', header=None, dtype=str)
                    PSNR = csvfile[csvfile.columns[6]]
                    for i, value in enumerate(PSNR):
                        #print(type(value))
                        try:
                            PSNR[i] = value.replace("psnr_y:", "").replace(".", ",")
                        except:
                            strfloat = str(value)
                            PSNR[i] = strfloat.replace("psnr_y:", "").replace(".", ",")
                    csv_values.append(PSNR)

                if "SSIM" in path:
                    #print("SSIM: " + path)
                    csvfile = pd.read_csv(path, delimiter = '\s+', header = None, dtype=str)
                    SSIM = csvfile[csvfile.columns[1]]
                    for i, value in enumerate(SSIM):
                        #print(type(value))
                        try:
                            SSIM[i] = value.replace("Y:", "").replace(".", ",")
                        except:                 #In the case of value is not str, but why?
                            strfloat = str(value)
                            SSIM[i] = strfloat.replace("Y:", "").replace(".", ",");
                    csv_values.append(SSIM)
            #print("#########")
        #print(csv_values)
        f = open(csv + '.csv', "a")
        output = ""
        for i, name in enumerate(key_list):
            float_val = csv_values[i][0].replace(",",".")
            if float(float_val)<= 1:   #Detect SSIM or PSNR
                output = output + name + " SSIM:" + ";" + name + " PSNR:" + ";"
            else:
                output = output + name + " PSNR:" + ";" + name + " SSIM:" + ";"
        output = output + "\n"

        #print(len(csv_values))
        for i in range(min(map(len, csv_values))):
            for j in range(len(csv_values)):
                output = output + csv_values[j][i] + ";"
            output = output + "\n"
        f.write(output)
    os.system('mkdir LOG')
    os.system('mkdir CSV')
    os.system('mv *.log ./LOG/')
    os.system('mv *.csv ./CSV/')
    print("JOB COMPLETED")
    print("Done in " + str(time.time() - start) + " s.")
    exit()

start = time.time()
if not (os.path.exists('./TS/')):
    os.system('mkdir TS')
    print('TS folder created')
os.system('touch log.txt')
os.system('echo > log.txt')
os.system('cp /home/labo/Documents/Timothee/QT/build-VQA2-Desktop_Qt_5_15_0_GCC_64bit-Debug/VQA2 .')  # Needed to run a alias from .bashrc
if len(sys.argv) < 2:
    input_path = os.getcwd()
else:
    input_path = sys.argv[1]
print("Working dir:" + input_path)
os.chdir(input_path)
do_print = True
get_input(input_path)
store_csv = {}


