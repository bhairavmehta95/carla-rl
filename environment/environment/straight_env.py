from __future__ import print_function

import argparse
import random
import time

from carla.client import make_carla_client
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.image_converter import to_rgb_array
from carla.planner.planner import Planner

import numpy as np
from gym import spaces

from carla_env import CarlaEnv
from carla_server import CarlaServer

from skimage.transform import resize

"""
Units: 
locations     cm
speed         km/h
acceleration  (km/h)/s
collisions    kg*cm/s
"""

# TODO:
"""
- Check units
- Check reward function
"""


class StraightDriveEnv(CarlaEnv):
	def __init__(self, client, frame_skip=1, cam_width=800, cam_height=600, town_string='Town01'):
		super(StraightDriveEnv, self).__init__()
		self.frame_skip = frame_skip
		self.client = client
		self._planner = Planner(town_string)

		camera0 = Camera('CameraRGB')
		camera0.set(CameraFOV=100)
		camera0.set_image_size(cam_height, cam_width)
		camera0.set_position(200, 0, 140)
		camera0.set_rotation(-15.0, 0, 0)

		self.start_goal_pairs = [[36, 40], [39, 35], [110, 114], [7, 3], [0, 4],
					[68, 50], [61, 59], [47, 64], [147, 90], [33, 87],
					[26, 19], [80, 76], [45, 49], [55, 44], [29, 107],
					[95, 104], [84, 34], [53, 67], [22, 17], [91, 148],
					[20, 107], [78, 70], [95, 102], [68, 44], [45, 69]]

		vehicles = 0
		pedestrians = 0
		weather = 1

		settings = CarlaSettings()
		settings.set(
			SynchronousMode=True,
			SendNonPlayerAgentsInfo=True,
			NumberOfVehicles=vehicles,
			NumberOfPedestrians=pedestrians,
			WeatherId=weather,
		)

		settings.randomize_seeds()
		settings.add_sensor(camera0)

		self.scene = self.client.load_settings(settings)

		img_shape = (cam_width, cam_height, 3)
		self.observation_space = spaces.Tuple(
			(spaces.Box(-np.inf, np.inf, (3,)), 
			spaces.Box(0, 255, img_shape))
		)
		self.action_space = spaces.Box(-np.inf, np.inf, shape=(3,))

		self.prev_state = np.array([0., 0., 0.])
		self.prev_collisions = np.array([0., 0., 0.])
		self.prev_intersections = np.array([0., 0.])


	def step(self, action):
		steer = np.clip(action[0], -1, 1)
		throttle = np.clip(action[1], 0, 1)
		brake = np.clip(action[2], 0, 1)
		for i in range(self.frame_skip):
			self.t += 1
			self.client.send_control(
				steer=steer,
				throttle=throttle,
				brake=brake,
				hand_brake=False,
				reverse=False)

		measurements, sensor_data = self.client.read_data()
		sensor_data = self._process_image(sensor_data['CameraRGB'])
		current_time = measurements.game_timestamp

		state, collisions, intersections, onehot = self._process_measurements(measurements)
		reward, dist_goal = self._calculate_reward(state, collisions, intersections)
		done = self._calculate_done(collisions, state, current_time)

		self.prev_state = np.array(state)
		self.prev_collisions = np.array(collisions)
		self.prev_intersections = np.array(intersections)

		measurement_obs = self._generate_obs(state, collisions, onehot, dist_goal)

		return (measurement_obs, sensor_data), reward, done, {}


	def reset(self):
		self._generate_start_goal_pair()

		print('Starting new episode...')
		# Blocking function until episode is ready
		self.client.start_episode(self.start_idx)

		measurements, sensor_data = self.client.read_data()
		sensor_data = self._process_image(sensor_data['CameraRGB'])
		
		state, collisions, intersections, onehot = self._process_measurements(measurements)
	
		self.prev_state = np.array(state)
		self.prev_collisions = np.array(collisions)
		self.prev_intersections = np.array(intersections)

		self.start_time = measurements.game_timestamp
		self.t = 0

		pos = np.array(state[0:2])
		dist_goal = np.linalg.norm(pos - self.goal)

		measurement_obs = self._generate_obs(state, collisions, onehot, dist_goal)

		return (measurement_obs, sensor_data)


	def _calculate_done(self, collisions, state, current_time):
		pos = np.array(state[0:2])
		dist_goal = np.linalg.norm(pos - self.goal)

		return self._is_timed_out(current_time) or self._is_goal(dist_goal)
		
		# Not described in paper, but should be there for safe driving
		return self._is_timed_out() and self._collision_on_step(dist_goal)


	def _calculate_planner_onehot(self, measurements):
		# return np.array([0, 0, 0, 0, 1]) # TODO need to debug
		print(type(self.end_point.location), type(self.end_point.orientation))
		val = self._planner.get_next_command(measurements.location, measurements.orientation, 
			self.end_point.location, self.end_point.orientation)

		if val == 0.0:
			return np.array([1, 0, 0, 0, 0])

		onehot = np.zeros(5)
		val = int(val) - 1
		onehot[val] = 1

		return onehot


	def _calculate_reward(self, state, collisions, intersections):
		pos = np.array(state[0:2])
		dist_goal = np.linalg.norm(pos - self.goal)
		dist_goal_prev = np.linalg.norm(self.prev_state[0:2] - self.goal)

		speed = state[2]
	
		# TODO: Check this?
		r = 1000 * (dist_goal_prev - dist_goal) / 10 + 0.05 * (speed - self.prev_state[2]) \
			- 2 * (sum(intersections) - sum(self.prev_intersections))

		return r, dist_goal


	def _calculate_timeout(self):
		self.timeout_t = ((self.timeout_dist / 100000.0) / 10.0) * 3600.0 + 10.0


	def _collision_on_step(self, collisions):
		return sum(collisions) > 0


	def _generate_obs(self, state, collisions, onehot, dist_goal):
		speed = state[2]
		collisions = np.sum(collisions)

		return np.concatenate((np.array([speed, dist_goal, collisions]), np.array(onehot)))


	def _generate_start_goal_pair(self):
		# Choose one player start at random.
		self.position_index = np.random.randint(0, len(self.start_goal_pairs) - 1)
		self.start_idx = self.start_goal_pairs[self.position_index][0]
		self.goal_idx = self.start_goal_pairs[self.position_index][1]

		start_point = self.scene.player_start_spots[self.start_idx]
		end_point = self.scene.player_start_spots[self.goal_idx]
		self.end_point = end_point

		self.goal = [end_point.location.x / 100, end_point.location.y / 100] # cm -> m      

		self.timeout_dist = self._planner.get_shortest_path_distance(
			[start_point.location.x, start_point.location.y, 22], 
			[start_point.orientation.x, start_point.orientation.y, 22],
			[end_point.location.x, end_point.location.y, 22],
			[end_point.orientation.x, end_point.orientation.y, 22]
		)

		self._calculate_timeout()


	def _is_goal(self, distance):
		return distance < 2.0


	def _is_timed_out(self, current_time):
		return (current_time - self.start_time) > (self.timeout_t * 1000)


	def _process_image(self, carla_raw_img):
		return resize(to_rgb_array(carla_raw_img), (84,84))


	def _process_measurements(self, measurements):
		player_measurements = measurements.player_measurements

		pos_x = player_measurements.transform.location.x / 100 # cm -> m
		pos_y = player_measurements.transform.location.y / 100
		speed = player_measurements.forward_speed

		col_cars = player_measurements.collision_vehicles
		col_ped = player_measurements.collision_pedestrians
		col_other = player_measurements.collision_other

		other_lane = player_measurements.intersection_otherlane
		offroad = player_measurements.intersection_offroad

		onehot = self._calculate_planner_onehot(player_measurements.transform)
	
		return np.array([pos_x, pos_y, speed]), np.array([col_cars, col_ped, col_other]), np.array([other_lane, offroad]), onehot


if __name__ == '__main__':
	host = 'localhost'
	port = 2000
	c = CarlaServer()
	while True:
		try:
			with make_carla_client(host, port) as client:
				s = StraightDriveEnv(client)
				s.reset()

				while True:
					obs, r, done, _ = s.step([0., 0.2, 0.])
					print(r, 'is the reward')
			if done:
				s.reset()
		
		except TCPConnectionError as error:
			print(error)
			time.sleep(1) 
