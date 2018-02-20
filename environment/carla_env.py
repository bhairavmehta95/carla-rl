from __future__ import print_function

import os
import argparse
import subprocess

class CarlaEnv:
	def __init__(self):
		if os.environ.get('CARLA_PATH') == None:
			print('Please set $CARLA_PATH to the *directory* that contains CarlaUE4.sh')
			exit()

		# carla_server_path = os.path.join(os.environ['CARLA_PATH'], 'CarlaUE4.sh /Game/Maps/Town01 -carla-server -benchmark -fps=15')
		# self.p = subprocess.Popen(carla_server_path, shell=True)		

	def step(self):
		pass

	def reset(self):
		pass

	def __del__(self):
		pass
		# self.p.kill()
	

