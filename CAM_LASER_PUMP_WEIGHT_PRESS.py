# -*- coding: utf-8 -*-
"Contains automation proposition for the push-pull setup, including weighting scale, camera, laser, pressure controller, syringe pusher. Made by clément 'Soda à la pêche' Petitjean with help of some LLM and documentations"
#%% import+variables+commands
import csv
import gc
import os
import shutil
import re
import numpy as np
import pylablib as pll
import time
import matplotlib.pyplot as plt
import cv2
import serial
import ctypes
from pylablib.devices import DCAM
#%% defining experimental parameters
texp = 1.
fps = 1.
interv = 1 / fps
biningfactor = 2
width, height = 2048, 2048
fluo_injection = 0.3
mean_vel = 0.5
inj_position = 4
medium = "homogeneous"
savingfolder = "D:/CLEMENT/exp/"
if os.path.exists(savingfolder):
    shutil.rmtree(savingfolder)
os.makedirs(savingfolder)
weight_log_file = savingfolder + "weight_data.csv"
SCALE_COM_PORT = 'COM21'
LASER_COM_PORT = 'COM8'
PUMP_COM_PORT = 'COM15'
OB1_COM_PORT = 'COM12'
BAUD_RATE_SCALE = 9600
BAUD_RATE_LASER = 9600
BAUD_RATE_PUMP = 9600
BAUD_RATE_OB1 = 115200
GREEN_LASER_ON = "L2 L=1"
GREEN_LASER_OFF = "L2 L=0"
RED_LASER_ON = "L1 L=1"
RED_LASER_OFF = "L1 L=0"
BLUE_LASER_ON = "L3 L=1"
BLUE_LASER_OFF = "L3 L=0"
PURPLE_LASER_ON = "L4 L=1"
PURPLE_LASER_OFF = "L4 L=0"
BLUE_LASER_POWER_ASK = "L3 ?C"
BLUE_LASER_POWER = "L3 C 100"
SHUTTER_OPEN = "SH1=1"
SHUTTER_CLOSE = "SH1=0"

def send_command(ser, command, wait=0.1):
    full_command = f"{command}\r\n"
    ser.write(full_command.encode())
    time.sleep(wait)
    return ser.read_all().decode(errors='ignore').strip()

def read_scale_weight(scale_ser):
    response = send_command(scale_ser, "IP")
    try:
        match = re.search(r'([-+]?\d*\.\d+|\d+)\s*g', response)
        return float(match.group()[0:-2])
    except:
        return 0.0

def pump_send(pump_ser, command, wait=0.1):
    full_cmd = f"{command}\r\n"
    pump_ser.write(full_cmd.encode())
    time.sleep(wait)
    return pump_ser.read_all().decode(errors='ignore').strip()

def pump_start(pump_ser, rate_ml_min):
    pump_send(pump_ser, f"SET RATE {rate_ml_min}")
    return pump_send(pump_ser, "RUN")

def pump_stop(pump_ser):
    return pump_send(pump_ser, "STOP")

def ob1_send(ob1_ser, command, wait=0.05):
    full_cmd = f"{command}\r\n"
    ob1_ser.write(full_cmd.encode())
    time.sleep(wait)
    return ob1_ser.read_all().decode(errors='ignore').strip()

def save_parameters():
    params_file = os.path.join(savingfolder, "parametres.txt")
    with open(params_file, "w") as f:
        f.write("=== PARAMÈTRES EXPÉRIMENTAUX ===\n")
        f.write(f"Chemin DLL : {pll.par['devices/dlls/dcamapi']}\n")
        f.write(f"Nombre de caméras détectées : {DCAM.get_cameras_number()}\n")
        f.write(f"Temps d'exposition (s) : {texp}\n")
        f.write(f"Facteur de binning : {biningfactor}\n")
        f.write(f"Résolution : {width} x {height}\n")
        f.write(f"Dossier : {savingfolder}\n")
        f.write(f"FPS : {fps}\n")
        f.write(f"Port balance : {SCALE_COM_PORT}\n")
        f.write(f"Port laser : {LASER_COM_PORT}\n")
        f.write(f"Port pompe : {PUMP_COM_PORT}\n")
        f.write(f"Port OB1 : {OB1_COM_PORT}\n")
        f.write(f"Baud scale : {BAUD_RATE_SCALE}\n")
        f.write(f"Baud laser : {BAUD_RATE_LASER}\n")
        f.write(f"Baud pompe : {BAUD_RATE_PUMP}\n")
        f.write(f"Baud OB1 : {BAUD_RATE_OB1}\n")
        f.write(f"Injection flow (ml/min) : {fluo_injection}\n")
        f.write(f"Mean flowrate (g/min) : {mean_vel}\n")
        f.write(f"Injection position : {inj_position}\n")
        f.write(f"Medium type : {medium}\n")
#%% defining protocols
def runpush(temps):
    # Initialization of hanamatsu cam
    pll.par["devices/dlls/dcamapi"] = "D:/CLEMENT/dcamnv/DCAMAPI/usb/Win/x64/dcamapi.dll"
    cam = DCAM.DCAMCamera()
    cam.set_exposure(texp)
    cam.set_roi(0, width, 0, height, hbin=biningfactor, vbin=biningfactor)
    ser = serial.Serial(LASER_COM_PORT, BAUD_RATE_LASER, timeout=1)
    scale_ser = serial.Serial(SCALE_COM_PORT, BAUD_RATE_SCALE, timeout=1)
    pump_ser = serial.Serial(PUMP_COM_PORT, BAUD_RATE_PUMP, timeout=1)
    ob1_ser = serial.Serial(OB1_COM_PORT, BAUD_RATE_OB1, timeout=1)
    #Weight CSV file
    with open(weight_log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image_Index", "Timestamp", "Weight"])

    send_command(ser, SHUTTER_OPEN)
    send_command(ser, BLUE_LASER_ON)
    send_command(ser, BLUE_LASER_POWER)

    ob1_send(ob1_ser, "SET_PRESS 2 100")
    pump_start(pump_ser, fluo_injection)

    save_parameters()
    j = 0
    start_time = time.time()
    #boucle while pour enregistrer les images
    while time.time() - start_time < temps:
        i = cam.grab(1)
        i = np.array(i)
        j += 1
        cv2.imwrite(savingfolder+str(j)+'.tif', i[0].astype('uint16'))
        weight = read_scale_weight(scale_ser)
        timestamp = time.time() - start_time
        with open(weight_log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([j, timestamp, weight])
        time.sleep(interv)
        gc.collect()

    pump_stop(pump_ser)
    ob1_send(ob1_ser, "SET_PRESS 2 0")

    send_command(ser, SHUTTER_CLOSE)

    ser.close()
    scale_ser.close()
    pump_ser.close()
    ob1_ser.close()

if __name__ == "__main__":
    #Put here the list of protocols to execute
    runpush(60)
