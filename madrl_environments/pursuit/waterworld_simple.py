import numpy as np
import scipy.spatial.distance as ssd
from gym import spaces
from gym.utils import seeding

from madrl_environments import AbstractMAEnv, Agent
from rltools.util import EzPickle


class Archea(Agent):

    def __init__(self, idx, radius, _n_sensors, sensor_range, speed_features=True):
        self._idx = idx
        self._radius = radius
        # Number of observation coordinates from each sensor
        self._position = None
        self._velocity = None
        self._n_sensors = _n_sensors
        self._sensor_range = sensor_range
        # Number of observation coordinates from each sensor
        self._sensor_obscoord = 3
        if speed_features:
            self._sensor_obscoord += 1
        self._obscoord_from_sensors = self._n_sensors * self._sensor_obscoord
        self._obs_dim = self._obscoord_from_sensors + 1  #+ 1  #2 for type, 1 for id

        # Sensors
        angles_K = np.linspace(0., 2. * np.pi, self._n_sensors + 1)[:-1]
        sensor_vecs_K_2 = np.c_[np.cos(angles_K), np.sin(angles_K)]
        self._sensors = sensor_vecs_K_2
        # Sensors
        # angles_K = np.linspace(0., 2. * np.pi, self._n_sensors + 1)[:-1]
        # sensor_vecs_K_2 = np.c_[np.cos(angles_K), np.sin(angles_K)]
        # self._sensors = sensor_vecs_K_2

    @property
    def observation_space(self):
        return spaces.Box(low=-10, high=10, shape=(self._obs_dim,))

    @property
    def action_space(self):
        return spaces.Box(low=-1, high=1, shape=(2,))

    @property
    def position(self):
        assert self._position is not None
        return self._position

    @property
    def velocity(self):
        assert self._velocity is not None
        return self._velocity

    def set_position(self, x_2):
        assert x_2.shape == (2,)
        self._position = x_2

    def set_velocity(self, v_2):
        assert v_2.shape == (2,)
        self._velocity = v_2

    @property
    def sensors(self):
        assert self._sensors is not None
        return self._sensors

    def sensed(self, obj_pos, obj_vel, obj_rad, speed=False, same=False):
        """Whether `obj` would be sensed by the pursuers"""
        relpos_obj = obj_pos - np.expand_dims(self.position, 0) 
        reldist = np.linalg.norm(relpos_obj)
        relpos_obj = relpos_obj * (1 - obj_rad/reldist)
        sensorvals = self.sensors.dot(relpos_obj.T)
        sensorvals[(sensorvals< 0) | (sensorvals > self._sensor_range) | ((
            relpos_obj**2).sum(axis=1)[None, :] - sensorvals**2 > self._radius**2)] = np.inf
        if same:
            sensorvals[:, self._idx - 1] = np.inf
        sensedmask = np.isfinite(np.array(sensorvals))
        sensed_distfeatures = np.zeros((self._n_sensors,1))
        sensed_distfeatures[sensedmask] = sensorvals[sensedmask]
        if speed:
            relvel = obj_vel - np.expand_dims(self.velocity, 0)
            sensorvals = self.sensors.dot(relvel.T)
            sensed_speedfeatures = np.zeros((self._n_sensors,1))
            sensed_speedfeatures[sensedmask] = sensorvals[sensedmask]
            return sensed_distfeatures, sensed_speedfeatures

        return sensed_distfeatures

class WaterWorld(AbstractMAEnv, EzPickle):

    def __init__(self, radius=0.015, obstacle_radius=0.2, obstacle_loc=np.array([0.5, 0.5]), ev_speed=0.01, 
                  n_sensors = 30, sensor_range=0.2, action_scale=0.01,food_reward=10, 
                  encounter_reward=.05, control_penalty= -5, speed_features=True, **kwargs):
        EzPickle.__init__(self, radius, obstacle_radius,
                          obstacle_loc, ev_speed,
                          action_scale, food_reward, encounter_reward,
                          control_penalty, speed_features, **kwargs)

        self.obstacle_radius = obstacle_radius
        self.obstacle_loc = obstacle_loc
        self.ev_speed = ev_speed
        self.n_sensors = n_sensors
        self.sensor_range = np.ones(1) * sensor_range
        self.radius = radius
        self.action_scale = action_scale
        self.food_reward = food_reward
        self.encounter_reward = encounter_reward
        self.control_penalty = control_penalty
        self.n_obstacles = 1
        self._speed_features = speed_features
        self.seed()
        self._pursuer = Archea(1, self.radius, self.n_sensors, self.sensor_range, speed_features=True) 
        self._evader  = Archea(1, self.radius, self.n_sensors, self.sensor_range,speed_features=True)
        self._food  = Archea(1, self.radius * 0.75, self.n_sensors, self.sensor_range, speed_features=True)
        self._pursuers = [self._pursuer]

    @property
    def reward_mech(self):
        return self._reward_mech

    @property
    def timestep_limit(self):
        return 1000

    @property
    def agents(self):
        return self._pursuers

    def get_param_values(self):
        return self.__dict__

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def _respawn(self, objx_2, radius):

        while ssd.euclidean(objx_2, self.obstaclesx_No_2) <= radius * 2 + self.obstacle_radius:
            # print objx_2, self.obstaclesx_No_2, ssd.euclidean(objx_2, self.obstaclesx_No_2), radius
            objx_2 = self.np_random.rand(2)
        return objx_2

    def reset(self):

        self._timesteps = 0
        # Initialize obstacles
        if self.obstacle_loc is None:
            self.obstaclesx_No_2 = self.np_random.rand(self.n_obstacles, 2)
        else:
            self.obstaclesx_No_2 = self.obstacle_loc
        self.obstaclesv_No_2 = np.zeros((self.n_obstacles, 2))

        # Initialize pursuers
        self._pursuer.set_position(self.np_random.rand(2))
        # Avoid spawning where the obstacles lie
        self._pursuer.set_position(self._respawn(self._pursuer.position, self._pursuer._radius))
        self._pursuer.set_velocity(np.zeros(2))

        # Initialize evaders
        self._evader.set_position(self.np_random.rand(2))
        self._evader.set_position(self._respawn(self._evader.position, self._evader._radius))
        self._evader.set_velocity(self.np_random.rand(2) - 0.5 * self.ev_speed)  # TODO policies

        self._food.set_position(self.np_random.rand(2))
        self._food.set_position(self._respawn(self._food.position, self._food._radius))
        # (self.np_random.rand(2) - 0.5)
        # Initialize poisons
        return self.step(np.zeros((1, 2)))[0]

    @property
    def is_terminal(self):
        if self._timesteps >= self.timestep_limit:
            return True
        return False

    def get_evaders_velocity(self):


        # if any(self._evader.position != np.clip(self._evader.position, 0, 1)):
        #         vel = -1 * self._evader.velocity
        D = 0.1
        E = 0.05
        ev_speed = 1
        evfromobst = ssd.euclidean(self._evader.position, self.obstaclesx_No_2)
        evfromfood = ssd.euclidean(self._food.position, self._evader.position)
        foodfromobst = ssd.euclidean(self._food.position, self.obstaclesx_No_2)
        vel = ((self._food.position - self._evader.position)/evfromfood) * self.action_scale
        is_colliding_evader = evfromobst <= self._evader._radius + self.obstacle_radius
        if evfromobst < self._evader._radius + self.obstacle_radius + E and evfromfood > foodfromobst:
            vel = self._evader.velocity
            rel_dis = (self._evader.position - self.obstaclesx_No_2)/evfromobst
            vel[0] = rel_dis[1] * self.action_scale
            vel[1] = -rel_dis[0] * self.action_scale 

        rel_pursuer = self._pursuer.position - self._evader.position
        evfrompur = ssd.euclidean(self._pursuer.position, self._evader.position)
        if np.linalg.norm(rel_pursuer) < D and np.linalg.norm(rel_pursuer) > 0.01:
            vel = self._evader.velocity - (rel_pursuer/evfrompur) * self.action_scale 
        return vel

    def get_partial_observation(self):
        sensed_obdistfeatures = self._pursuer.sensed(self.obstaclesx_No_2,0,self.obstacle_radius,speed=False)
        sensed_evdistfeatures, sensed_evvelfeatures = self._pursuer.sensed(self._evader.position,self._evader.velocity,self._evader._radius,speed=True)
        sensed_fooddistfeatures = self._pursuer.sensed(self._food.position,0,self._food._radius,speed=False)
        if self._speed_features:
            sensor_features = np.c_[sensed_obdistfeatures, sensed_evdistfeatures, sensed_fooddistfeatures,sensed_evvelfeatures]
        else:
            sensor_features = np.c_[sensed_obdistfeatures, sensed_evdistfeatures, sensed_fooddistfeatures]
        sensor_features = sensor_features.ravel()
        pursuerfromev = ssd.euclidean(self._pursuer.position,self._evader.position)
        is_colliding_ev_pursuer = pursuerfromev <= self._pursuer._radius + self._evader._radius
        obs = np.append(sensor_features, is_colliding_ev_pursuer)  
        # print np.shape(obs)
        obslist = []
        obslist.append(obs)
        return obslist


    def get_full_observation(self):
        pursuers_pos = np.array(self._pursuer.position)
        evaders_pos = np.array(self._evader.position)
        food_pos = np.array(self._food.position)
        obs_pos = np.array(self.obstaclesx_No_2)
        pursuer_rad = self._pursuer._radius
        evader_rad = self._evader._radius
        food_rad = self._food._radius
        obs_rad = self.obstacle_radius
        pursuer_vel = np.array(self._pursuer.velocity)
        evader_vel = np.array(self._evader.velocity)
        obs = []
        obs.extend(np.concatenate([pursuers_pos,evaders_pos, food_pos, obs_pos]))
        obs.extend([pursuer_rad, evader_rad, food_rad, obs_rad])
        obs.extend(np.concatenate([pursuer_vel, evader_vel]))
        obs = np.array(obs)
        obslist = []
        obslist.append(obs)
        return obslist

    def step(self, action):
 
        action = np.asarray(action)
        action = action.reshape(2)
        action = action * self.action_scale
        reward = 0

        # Penalize large actions
        reward += self.control_penalty * np.sum(action**2)
        
        self._pursuer.set_velocity(self._pursuer.velocity + action)
        probable_position = self._pursuer.position + self._pursuer.velocity
        # Bounce pursuer on hitting an obstacle
        pursuerfromobst = ssd.euclidean(probable_position, self.obstaclesx_No_2)
        is_colliding_pursuer = pursuerfromobst <= self._pursuer._radius + self.obstacle_radius
        if is_colliding_pursuer:
            current_dist = ssd.euclidean(self._pursuer.position, self.obstaclesx_No_2)
            displacement = self._pursuer.velocity * ((current_dist - self.obstacle_radius- self._pursuer._radius)/current_dist)
            self._pursuer.set_position(self._pursuer.position + displacement)
            self._pursuer.set_velocity(-1 * self._pursuer.velocity)
        else:
            self._pursuer.set_position(probable_position)

        # Pursuer stop on hitting a wall
        clippedx_2 = np.clip(self._pursuer.position, 0, 1)
        vel_2 = self._pursuer.velocity
        vel_2[self._pursuer.position != clippedx_2] = 0
        self._pursuer.set_velocity(vel_2)
        self._pursuer.set_position(clippedx_2)   

        self._evader.set_velocity(self.get_evaders_velocity())
        probable_position_ev = self._evader.position + self._evader.velocity
        # Bounce evader on hitting an obstacle
        evfromobst = ssd.euclidean(probable_position_ev, self.obstaclesx_No_2)
        is_colliding_evader = evfromobst <= self._evader._radius + self.obstacle_radius
        if is_colliding_evader:
            # print("Collision")
            current_dist = ssd.euclidean(self._evader.position, self.obstaclesx_No_2)
            displacement = self._evader.velocity * ((current_dist - self.obstacle_radius- self._evader._radius)/current_dist)
            self._evader.set_position(self._evader.position + displacement)
            self._evader.set_velocity(-1 * self._evader.velocity)
        else:
            self._evader.set_position(probable_position_ev)

        clippedx_2 = np.clip(self._evader.position, 0, 1)
        vel_2 = self._evader.velocity
        vel_2[self._evader.position != clippedx_2] = 0
        self._evader.set_velocity(vel_2)
        self._evader.set_position(clippedx_2)

        # print(self._evader.velocity,self._evader.position, self._food.position)
        # check if evader caught its food
        evfromfood = ssd.euclidean(self._evader.position, self._food.position)
        is_colliding_food = evfromfood <= self._evader._radius + self._food._radius
        if is_colliding_food:
            self._food.set_position(self.np_random.rand(2))
            self._food.set_position(self._respawn(self._food.position, self._food._radius))    
                
        # Find collisions
        # Evaders
        pursuerfromev = ssd.euclidean(self._pursuer.position,self._evader.position)
        is_colliding_ev_pursuer = pursuerfromev <= self._pursuer._radius + self._evader._radius
        evcaught = False
        if is_colliding_ev_pursuer:
            evcaught = True
            reward += self.food_reward
            self._evader.set_position(self.np_random.rand(2))
            self._evader.set_position(self._respawn(self._evader.position, self._evader._radius))

        # Update reward based on these collisions 
        obslist = self.get_partial_observation()
        # print self._pursuer.observation_space.shape
        self._timesteps += 1
        done = self.is_terminal
        info = dict(evcaught=evcaught)
        rlist = np.array([reward])
        # print(rlist)
        return obslist, rlist, done, info

    def render(self, screen_size=600, rate=10, mode='human'):
        import cv2
        img = np.empty((screen_size, screen_size, 3), dtype=np.uint8)
        img[...] = 255
        # Obstacles

        color = (128, 128, 0)
        cv2.circle(img,
                   tuple((self.obstaclesx_No_2 * screen_size).astype(int)),
                   int(self.obstacle_radius * screen_size), color, -1, lineType=cv2.LINE_AA)
        
        for k in range(self._pursuer._n_sensors):
            color = (0, 0, 0)
            cv2.line(img,
                     tuple((self._pursuer.position * screen_size).astype(int)),
                     tuple(((self._pursuer.position + self._pursuer._sensor_range * self._pursuer.sensors[k]) *
                            screen_size).astype(int)), color, 1, lineType=cv2.LINE_AA)
            cv2.circle(img,
                       tuple((self._pursuer.position * screen_size).astype(int)),
                       int(self._pursuer._radius * screen_size), (255, 0, 0), -1, lineType=cv2.LINE_AA)
        # # Pursuer
        # cv2.circle(img,
        #           tuple((self._pursuer.position * screen_size).astype(int)),
        #           int(self._pursuer._radius * screen_size), (255, 0, 0), -1, lineType=cv2.LINE_AA)
        # Evader
        color = (0, 255, 0)
        cv2.circle(img,
                   tuple((self._evader.position * screen_size).astype(int)),
                   int(self._evader._radius * screen_size), color, -1, lineType=cv2.LINE_AA)

        # Food
        color = (0, 0, 255)
        cv2.circle(img,
                   tuple((self._food.position * screen_size).astype(int)),
                   int(self._food._radius * screen_size), color, -1, lineType=cv2.LINE_AA)

        opacity = 0.4
        bg = np.ones((screen_size, screen_size, 3), dtype=np.uint8) * 255
        cv2.addWeighted(bg, opacity, img, 1 - opacity, 0, img)
        cv2.imshow('Waterworld', img)
        cv2.waitKey(rate)
        return np.asarray(img)[..., ::-1]


if __name__ == '__main__':
    env = WaterWorld()
    obs = env.reset()
    while True:
        obs, rew, _, _ = env.step(env.np_random.randn(2) * .01)
        print(obs)
        # print obs
        if rew.sum() > 0:
            print(rew)
        env.render()