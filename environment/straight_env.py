from __future__ import print_function

import argparse
import random
import time

from carla.client import make_carla_client
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError

import numpy as np
from gym import spaces

from carla_env import CarlaEnv

class StraightDriveEnv(CarlaEnv):
    def __init__(self, client, frame_skip=1, cam_width=800, cam_height=600):
        super().__init__()

        self.frame_skip = frame_skip
        self.client = client

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
        print(self.client)
        steer = np.clip(-1, 1, action[0])
        throttle = np.clip(0, 1, action[1])
        brake = np.clip(0, 1, action[2])

        for i in range(self.frame_skip):
            self.client.send_control(
                steer=steer,
                throttle=throttle,
                brake=brake,
                hand_brake=False,
                reverse=False)

        measurements, sensor_data = self.client.read_data()
        sensor_data = sensor_data.data
        state, collisions, intersections = self._process_measurements(measurements)
        reward = self._calculate_reward(state, collisions, intersections)
        done = self._calculate_done(collisions)

        self.prev_state = np.array(state)
        self.prev_collisions = np.array(collisions)
        self.prev_intersections = np.array(intersections)

        print((state, sensor_data), reward, collisions)
        return (state, sensor_data), reward, done, {}


    def reset(self):
        self._generate_start_goal_pair()

        print('Starting new episode...')
        # Blocking function until episode is ready
        self.client.start_episode(self.start_idx)
        measurements, sensor_data = self.client.read_data()
        sensor_data = sensor_data.data
        state, collisions, intersections = self._process_measurements(measurements)
    
        self.prev_state = np.array(state)
        self.prev_collisions = np.array(collisions)
        self.prev_intersections = np.array(intersections)

        return (state, sensor_data)


    def _calculate_reward(self, state, collisions, intersections):
        pos = np.array(state[0:2])
        dist_goal = np.linalg.norm(pos - self.goal)
        dist_goal_prev = np.linalg.norm(self.prev_state[0:2] - self.goal)

        speed = state[2]

        # TODO: Check this?
        r = (dist_goal_prev - dist_goal) / 1000 + 0.05 * (speed - self.prev_state[2]) \
            - 2 * (sum(intersections) - sum(self.prev_intersections))

        return r


    def _calculate_timeout(self, distance):
        pass

    def _collision_on_step(self, collisions):
        return sum(collisions) > 0


    def _generate_start_goal_pair(self):
        # Choose one player start at random.
        self.position_index = np.random.randint(0, len(self.start_goal_pairs) - 1)
        self.start_idx = self.start_goal_pairs[self.position_index][0]
        self.goal_idx = self.start_goal_pairs[self.position_index][1]

        self.goal = self.scene.player_start_spots[self.goal_idx]


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

        return np.array([pos_x, pos_y, speed]), np.array([col_cars, col_ped, col_other]), np.array([other_lane, offroad])


if __name__ == '__main__':
    host = 'localhost'
    port = 2000

    with make_carla_client(host, port) as client:
        s = StraightDriveEnv(client)

        while True:
            s.step(np.random.rand(3))

