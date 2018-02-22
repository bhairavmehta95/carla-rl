from __future__ import print_function

import os
import signal
import subprocess

class CarlaServer():
    def __init__(self, port):
        if os.environ.get('CARLA_PATH') == None:
            print('Please set $CARLA_PATH to the *directory* that contains CarlaUE4.sh')

        devnull = open(os.devnull, 'w')
        carla_server_path = os.path.join(os.environ['CARLA_PATH'], 'CarlaUE4/Binaries/Linux/CarlaUE4 /Game/Maps/Town01 -carla-settings=CarlaSettings.ini -carla-server -fps=1 -world-port={}'.format(port))
        self.p = subprocess.Popen(carla_server_path.split(), stdout=devnull)    
        

    def __del__(self):
        pid = self.p.pid
        os.kill(pid, signal.SIGINT)

        if not self.p.poll():
            print("CARLA server successfully killed.")
        else:
            print("CARLA server still running, please kill manually.")
