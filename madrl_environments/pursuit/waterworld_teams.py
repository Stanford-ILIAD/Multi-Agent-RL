import numpy as np
import scipy.spatial.distance as ssd
from gym import spaces
from gym.utils import seeding

from madrl_environments import AbstractMAEnv, Agent
from rltools.util import EzPickle


class Archea(Agent):

    def __init__(self, idx, radius, n_sensors, sensor_range, addid=True, speed_features=True):
        self._idx = idx
        self._radius = radius
        self._n_sensors = n_sensors
        self._sensor_range = sensor_range
        # Number of observation coordinates from each sensor
        self._sensor_obscoord = 5
        if speed_features:
            self._sensor_obscoord += 4
        self._obscoord_from_sensors = self._n_sensors * self._sensor_obscoord
        self._obs_dim = self._obscoord_from_sensors + 3 #+ 1  #2 for type, 1 for id
        if addid:
            self._obs_dim += 1

        self._position = None
        self._velocity = None
        # Sensors
        angles_K = np.linspace(0., 2. * np.pi, self._n_sensors + 1)[:-1]
        sensor_vecs_K_2 = np.c_[np.cos(angles_K), np.sin(angles_K)]
        self._sensors = sensor_vecs_K_2

    @property
    def observation_space(self):
        return spaces.Box(low=-10, high=10, shape=(self._obs_dim,), dtype=np.float32)

    @property
    def action_space(self):
        return spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

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

    def sensed(self, objx_N_2, same=False):
        """Whether `obj` would be sensed by the pursuers"""
        relpos_obj_N_2 = objx_N_2 - np.expand_dims(self.position, 0)
        sensorvals_K_N = self.sensors.dot(relpos_obj_N_2.T)
        sensorvals_K_N[(sensorvals_K_N < 0) | (sensorvals_K_N > self._sensor_range) | ((
            relpos_obj_N_2**2).sum(axis=1)[None, :] - sensorvals_K_N**2 > self._radius**2)] = np.inf
        if same:
            sensorvals_K_N[:, self._idx - 1] = np.inf

        return sensorvals_K_N


class MAWaterWorldTeams(AbstractMAEnv, EzPickle):

    def __init__(self, n_pursuers1, n_pursuers2, n_evaders1, n_evaders2, n_coop=2, n_poison=10, radius=0.015,
                 obstacle_radius=0.2, obstacle_loc=np.array([0.5, 0.5]), ev_speed=0.01,
                 poison_speed=0.01, n_sensors=30, sensor_range=0.2, action_scale=0.01,
                 poison_reward=-1., food_reward=1., encounter_reward=.05, control_penalty=-.5,
                 collision_penalty=-1, reward_mech='local', addid=True, speed_features=True, **kwargs):
        EzPickle.__init__(self, n_pursuers1, n_pursuers2, n_evaders1, n_evaders2, n_coop, n_poison, radius, obstacle_radius,
                          obstacle_loc, ev_speed, poison_speed, n_sensors, sensor_range,
                          action_scale, poison_reward, food_reward, encounter_reward,
                          control_penalty, reward_mech, addid, speed_features, **kwargs)
        self.n_pursuers1 = n_pursuers1
        self.n_pursuers2 = n_pursuers2
        self.n_evaders1 = n_evaders1
        self.n_evaders2 = n_evaders2
        self.n_coop = n_coop
        self.n_poison = n_poison
        self.obstacle_radius = obstacle_radius
        self.obstacle_loc = obstacle_loc
        self.poison_speed = poison_speed
        self.radius = radius
        self.ev_speed = ev_speed
        self.n_sensors = n_sensors
        self.sensor_range1 = np.ones(self.n_pursuers1) * sensor_range
        self.sensor_range2 = np.ones(self.n_pursuers2) * sensor_range
        self.action_scale = action_scale
        self.poison_reward = poison_reward
        self.food_reward = food_reward
        self.control_penalty = control_penalty
        self.collision_penalty = collision_penalty
        self.encounter_reward = encounter_reward

        self.n_obstacles = 1
        self._reward_mech = reward_mech
        self._addid = addid
        self._speed_features = speed_features
        self.seed()
        self._pursuers1 = [
            Archea(npu + 1, self.radius, self.n_sensors, self.sensor_range1[npu], addid=self._addid,
                   speed_features=self._speed_features) for npu in range(self.n_pursuers1)
        ]
        self._pursuers2 = [
            Archea(npu + 1, self.radius, self.n_sensors, self.sensor_range2[npu], addid=self._addid,
                   speed_features=self._speed_features) for npu in range(self.n_pursuers2)
        ]
        self._evaders1 = [
            Archea(nev + 1, self.radius * 2, self.n_pursuers1, self.sensor_range1.mean() / 2)
            for nev in range(self.n_evaders1)
        ]
        self._evaders2 = [
            Archea(nev + 1, self.radius * 2, self.n_pursuers1, self.sensor_range1.mean() / 2)
            for nev in range(self.n_evaders2)
        ]
        self._poisons = [
            Archea(npo + 1, self.radius * 3 / 4, self.n_poison, 0) for npo in range(self.n_poison)
        ]

    @property
    def reward_mech(self):
        return self._reward_mech

    @property
    def timestep_limit(self):
        return 1000

    @property
    def agents(self):
        return self._pursuers1 + self._pursuers2

    def get_param_values(self):
        return self.__dict__

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def _respawn(self, objx_2, radius):
        while ssd.cdist(objx_2[None, :], self.obstaclesx_No_2) <= radius * 2 + self.obstacle_radius:
            objx_2 = self.np_random.rand(2)
        return objx_2

    def reset(self):
        self._timesteps = 0
        # Initialize obstacles
        if self.obstacle_loc is None:
            self.obstaclesx_No_2 = self.np_random.rand(self.n_obstacles, 2)
        else:
            self.obstaclesx_No_2 = self.obstacle_loc[None, :]
        self.obstaclesv_No_2 = np.zeros((self.n_obstacles, 2))

        # Initialize pursuers
        for pursuer in self._pursuers1:
            pursuer.set_position(self.np_random.rand(2))
            # Avoid spawning where the obstacles lie
            pursuer.set_position(self._respawn(pursuer.position, pursuer._radius))
            pursuer.set_velocity(np.zeros(2))
        for pursuer in self._pursuers2:
            pursuer.set_position(self.np_random.rand(2))
            # Avoid spawning where the obstacles lie
            pursuer.set_position(self._respawn(pursuer.position, pursuer._radius))
            pursuer.set_velocity(np.zeros(2))

        # Initialize evaders
        for evader in self._evaders1:
            evader.set_position(self.np_random.rand(2))
            evader.set_position(self._respawn(evader.position, evader._radius))
            evader.set_velocity((self.np_random.rand(2) - 0.5) * self.ev_speed)  # TODO policies

        for evader in self._evaders2:
            evader.set_position(self.np_random.rand(2))
            evader.set_position(self._respawn(evader.position, evader._radius))
            evader.set_velocity((self.np_random.rand(2) - 0.5) * self.ev_speed)
        # Initialize poisons
        for poison in self._poisons:
            poison.set_position(self.np_random.rand(2))
            poison.set_position(self._respawn(poison.position, poison._radius))
            poison.set_velocity((self.np_random.rand(2) - 0.5) * self.ev_speed)

        return self.step(np.zeros((self.n_pursuers1 + self.n_pursuers2, 2)))[0]

    @property
    def is_terminal(self):
        if self._timesteps >= self.timestep_limit:
            return True
        return False

    def _caught(self, is_colliding_N1_N2, n_coop):
        """ Checke whether collision results in catching the object

        This is because you need `n_coop` agents to collide with the object to actually catch it
        """
        # number of N1 colliding with given N2
        n_collisions_N2 = is_colliding_N1_N2.sum(axis=0)
        is_caught_cN2 = np.where(n_collisions_N2 >= n_coop)[0]

        # number of N2 colliding with given N1
        who_collisions_N1_cN2 = is_colliding_N1_N2[:, is_caught_cN2]
        who_caught_cN1 = np.where(who_collisions_N1_cN2 >= 1)[0]

        return is_caught_cN2, who_caught_cN1

    def _closest_dist(self, pursuers, closest_obj_idx_Np_K, sensorvals_Np_K_N):
        """Closest distances according to `idx`"""
        sensorvals = []
        # print sensorvals_Np_K_N.shape
        for inp in range(len(pursuers)):
            sensorvals.append(sensorvals_Np_K_N[inp, ...][np.arange(self.n_sensors),
                                                          closest_obj_idx_Np_K[inp, ...]])
        return np.c_[sensorvals]

    def _extract_speed_features(self, pursuers, objv_N_2, closest_obj_idx_N_K, sensedmask_obj_Np_K):
        sensorvals = []
        for pursuer in pursuers:
            sensorvals.append(
                pursuer.sensors.dot((objv_N_2 - np.expand_dims(pursuer.velocity, 0)).T))
        sensed_objspeed_Np_K_N = np.c_[sensorvals]

        sensed_objspeedfeatures_Np_K = np.zeros((len(pursuers), self.n_sensors))

        sensorvals = []
        for inp in range(len(pursuers)):
            sensorvals.append(sensed_objspeed_Np_K_N[inp, :, :][np.arange(self.n_sensors),
                                                                closest_obj_idx_N_K[inp, :]])
        sensed_objspeedfeatures_Np_K[sensedmask_obj_Np_K] = np.c_[sensorvals][sensedmask_obj_Np_K]

        return sensed_objspeedfeatures_Np_K

    def hit_walls(self, particles):
        for npu, particle in enumerate(particles):
            clippedx_2 = np.clip(particle.position, 0, 1)
            vel_2 = particle.velocity
            vel_2[particle.position != clippedx_2] = 0
            particle.set_velocity(vel_2)
            particle.set_position(clippedx_2)

    def rebound_obstacles(self, particles):
        obstacle_coll_Npo = np.zeros(len(particles))
        for npo, particle in enumerate(particles):
            distfromobst_No = ssd.cdist(np.expand_dims(particle.position, 0), self.obstaclesx_No_2)
            is_colliding_No = distfromobst_No <= particle._radius + self.obstacle_radius
            obstacle_coll_Npo[npo] = is_colliding_No.sum()
            if obstacle_coll_Npo[npo] > 0:
                particle.set_velocity(-1 * particle.velocity)

    def caught_particles(self, pursuers, particles, n_coop):
        pursuers_ = np.array([pursuer.position for pursuer in pursuers])
        particles_  = np.array([evader.position for evader in particles])
        dists_ = ssd.cdist(pursuers_, particles_)
        is_colliding_ = dists_ <= np.asarray([
            pursuer._radius + evader._radius for pursuer in pursuers
            for evader in particles
        ]).reshape(len(pursuers), len(particles))

        # num_collisions depends on how many needed to catch an evader
        particle_caught, which_pursuer_caught_particle = self._caught(is_colliding_, n_coop)
        return particle_caught, which_pursuer_caught_particle

    def encounter_particles(self, pursuers, particles):
        pursuers_ = np.array([pursuer.position for pursuer in pursuers])
        particles_  = np.array([evader.position for evader in particles])
        dists_ = ssd.cdist(pursuers_, particles_)
        is_colliding_ = dists_ <= np.asarray([
            pursuer._radius + evader._radius for pursuer in pursuers
            for evader in particles
        ]).reshape(len(pursuers), len(particles))

        # num_collisions depends on how many needed to catch an evader
        particle_encountered, which_pursuer_encounterd_particle = self._caught(is_colliding_, 1)
        return particle_encountered, which_pursuer_encounterd_particle

    def collect_sensed_features(self, pursuers, evaders, other_team):

        evadersx_Ne_2 = np.array([evader.position for evader in evaders])
        poisonx_Npo_2 = np.array([poison.position for poison in self._poisons])
        pursuersx_Np_2 = np.array([pursuer.position for pursuer in pursuers])
        other_teamx_Np_2 = np.array([pursuer.position for pursuer in other_team])
        sensorvals_Np_K_No = np.array(
            [pursuer.sensed(self.obstaclesx_No_2) for pursuer in pursuers])

        # print sensorvals_Np_K_No.shape, len(pursuers)
        # Evaders
        sensorvals_Np_K_Ne = np.array([pursuer.sensed(evadersx_Ne_2) for pursuer in pursuers])

        # Other team 
        sensorvals_Np_K_Np2 = np.array([pursuer.sensed(other_teamx_Np_2) for pursuer in pursuers])

        # Poison
        sensorvals_Np_K_Npo = np.array(
            [pursuer.sensed(poisonx_Npo_2) for pursuer in pursuers])

        # Allies
        sensorvals_Np_K_Np = np.array(
            [pursuer.sensed(pursuersx_Np_2, same=True) for pursuer in pursuers])

        # dist features

        # obstacle
        closest_ob_idx_Np_K = np.argmin(sensorvals_Np_K_No, axis=2)
        closest_ob_dist_Np_K = self._closest_dist(pursuers, closest_ob_idx_Np_K, sensorvals_Np_K_No)
        sensedmask_ob_Np_K = np.isfinite(closest_ob_dist_Np_K)
        sensed_obdistfeatures_Np_K = np.zeros((len(pursuers), self.n_sensors))
        sensed_obdistfeatures_Np_K[sensedmask_ob_Np_K] = closest_ob_dist_Np_K[sensedmask_ob_Np_K]
        # Evaders
        closest_ev_idx_Np_K = np.argmin(sensorvals_Np_K_Ne, axis=2)
        closest_ev_dist_Np_K = self._closest_dist(pursuers, closest_ev_idx_Np_K, sensorvals_Np_K_Ne)
        sensedmask_ev_Np_K = np.isfinite(closest_ev_dist_Np_K)
        sensed_evdistfeatures_Np_K = np.zeros((len(pursuers), self.n_sensors))
        sensed_evdistfeatures_Np_K[sensedmask_ev_Np_K] = closest_ev_dist_Np_K[sensedmask_ev_Np_K]
        # Poison
        closest_po_idx_Np_K = np.argmin(sensorvals_Np_K_Npo, axis=2)
        closest_po_dist_Np_K = self._closest_dist(pursuers, closest_po_idx_Np_K, sensorvals_Np_K_Npo)
        sensedmask_po_Np_K = np.isfinite(closest_po_dist_Np_K)
        sensed_podistfeatures_Np_K = np.zeros((len(pursuers), self.n_sensors))
        sensed_podistfeatures_Np_K[sensedmask_po_Np_K] = closest_po_dist_Np_K[sensedmask_po_Np_K]
        # Allies
        closest_pu_idx_Np_K = np.argmin(sensorvals_Np_K_Np, axis=2)
        closest_pu_dist_Np_K = self._closest_dist(pursuers, closest_pu_idx_Np_K, sensorvals_Np_K_Np)
        sensedmask_pu_Np_K = np.isfinite(closest_pu_dist_Np_K)
        sensed_pudistfeatures_Np_K = np.zeros((len(pursuers), self.n_sensors))
        sensed_pudistfeatures_Np_K[sensedmask_pu_Np_K] = closest_pu_dist_Np_K[sensedmask_pu_Np_K]

        # Other team
        closest_ot_idx_Np_K = np.argmin(sensorvals_Np_K_Np2, axis=2)
        closest_ot_dist_Np_K = self._closest_dist(pursuers, closest_ot_idx_Np_K, sensorvals_Np_K_Np2)
        sensedmask_ot_Np_K = np.isfinite(closest_ot_dist_Np_K)
        sensed_otdistfeatures_Np_K = np.zeros((len(pursuers), self.n_sensors))
        sensed_otdistfeatures_Np_K[sensedmask_pu_Np_K] = closest_ot_dist_Np_K[sensedmask_pu_Np_K]

        # speed features
        pursuersv_Np_2 = np.array([pursuer.velocity for pursuer in pursuers])
        evadersv_Ne_2 = np.array([evader.velocity for evader in evaders])
        poisonv_Npo_2 = np.array([poison.velocity for poison in self._poisons])
        other_teamv_Np_2 = np.array([pursuer.velocity for pursuer in other_team])
        # Evaders

        sensed_evspeedfeatures_Np_K = self._extract_speed_features(pursuers, evadersv_Ne_2,
                                                                   closest_ev_idx_Np_K,
                                                                   sensedmask_ev_Np_K)
        # Poison
        sensed_pospeedfeatures_Np_K = self._extract_speed_features(pursuers, poisonv_Npo_2,
                                                                   closest_po_idx_Np_K,
                                                                   sensedmask_po_Np_K)
        # Allies
        sensed_puspeedfeatures_Np_K = self._extract_speed_features(pursuers, pursuersv_Np_2,
                                                                   closest_pu_idx_Np_K,
                                                                   sensedmask_pu_Np_K)

        # Other team
        sensed_otspeedfeatures_Np_k = self._extract_speed_features(pursuers, other_teamv_Np_2,
                                                                    closest_ot_idx_Np_K,
                                                                    sensedmask_ot_Np_K)


        if self._speed_features:
            sensorfeatures = np.c_[sensed_obdistfeatures_Np_K, sensed_evdistfeatures_Np_K,
                                          sensed_evspeedfeatures_Np_K, sensed_podistfeatures_Np_K,
                                          sensed_pospeedfeatures_Np_K, sensed_pudistfeatures_Np_K,
                                          sensed_puspeedfeatures_Np_K, sensed_otdistfeatures_Np_K,
                                          sensed_otspeedfeatures_Np_k]
        else:
            sensorfeatures = np.c_[sensed_obdistfeatures_Np_K, sensed_evdistfeatures_Np_K,
                                          sensed_podistfeatures_Np_K, sensed_pudistfeatures_Np_K,
                                          sensed_otdistfeatures_Np_K]
        return sensorfeatures

    def respawn_dead(self, particles, particle_caught):   
        if particle_caught.size:
            for particlecaught in particle_caught:
                particles[particlecaught].set_position(self.np_random.rand(2))
                particles[particlecaught].set_position(
                    self._respawn(particles[particlecaught].position, particles[particlecaught]
                                  ._radius))
                particles[particlecaught].set_velocity(
                    (self.np_random.rand(2,) - 0.5) * self.poison_speed)

    def get_obs_list(self,pursuers, evaders, other_team, sensorfeatures):
        pursuersx_Np_2 = np.array([pursuer.position for pursuer in pursuers])
        evadersx_Ne_2 = np.array([evader.position for evader in evaders])
        otherteamx_Np_2 = np.array([pursuer.position for pursuer in other_team])
        poisonx_Npo_2 = np.array([poison.position for poison in self._poisons])
        dists_ev = ssd.cdist(pursuersx_Np_2, evadersx_Ne_2)
        is_colliding_ev_Np_Ne = dists_ev <= np.asarray([
            pursuer._radius + evader._radius for pursuer in pursuers
            for evader in evaders
        ]).reshape(len(pursuers), len(evaders))

        dists_po = ssd.cdist(pursuersx_Np_2, poisonx_Npo_2)
        is_colliding_po_Np_Npo = dists_po <= np.asarray([
            pursuer._radius + poison._radius for pursuer in pursuers
            for poison in self._poisons
        ]).reshape(len(pursuers), self.n_poison)

        dists_ot = ssd.cdist(pursuersx_Np_2, otherteamx_Np_2)
        is_colliding_ot_Np_Np = dists_ot <= np.asarray([
            pursuer._radius + pursuer2._radius for pursuer in pursuers
            for pursuer2 in other_team
        ]).reshape(len(pursuers), len(other_team))

        obslist = []
        for inp in range(len(pursuers)):
            if self._addid:
                obslist.append(
                    np.concatenate([
                        sensorfeatures[inp, ...].ravel(), [
                            float((is_colliding_ev_Np_Ne[inp, :]).sum() > 0), float((
                                is_colliding_po_Np_Npo[inp, :]).sum() > 0), float((
                                is_colliding_ot_Np_Np[inp,:]).sum() > 0)
                        ], [inp + 1]
                    ]))
            else:
                obslist.append(
                    np.concatenate([
                        sensorfeatures[inp, ...].ravel(), [
                            float((is_colliding_ev_Np_Ne[inp, :]).sum() > 0), float((
                                is_colliding_po_Np_Npo[inp, :]).sum() > 0), float((
                                is_colliding_ot_Np_Np[inp, :]).sum() > 0)
                        ]
                    ]))
        return obslist

    def step(self, action_Np2_team):
        # print len(action_Np2_team)
        action_Np2_team = np.asarray(action_Np2_team)
        # print action_Np2_team.shape
        action_Np_2_team = action_Np2_team.reshape((self.n_pursuers1 + self.n_pursuers2, 2))
        action_Np_2_team1 = action_Np_2_team[0:self.n_pursuers1][:]
        action_Np_2_team2 = action_Np_2_team[self.n_pursuers1:self.n_pursuers1+self.n_pursuers2][:]
        # print action_Np_2_team.shape, action_Np_2_team1.shape, action_Np_2_team2.shape
        action_Np_2_team1 = action_Np_2_team1.reshape((self.n_pursuers1, 2))
        action_Np_2_team2 = action_Np_2_team2.reshape((self.n_pursuers2, 2))
        
        actions_Np_2_team1 = action_Np_2_team1 * self.action_scale
        actions_Np_2_team2 = action_Np_2_team2 * self.action_scale
        rewards_team1 = np.zeros((self.n_pursuers1,))
        rewards_team2 = np.zeros((self.n_pursuers2,))
        
        # print action_Np_2_team.shape, action_Np_2_team1.shape, action_Np_2_team2.shape
        assert action_Np_2_team1.shape == (self.n_pursuers1, 2)
        assert action_Np_2_team2.shape == (self.n_pursuers2, 2)

        for npu, pursuer in enumerate(self._pursuers1):
            pursuer.set_velocity(pursuer.velocity + actions_Np_2_team1[npu])
            pursuer.set_position(pursuer.position + pursuer.velocity)

        for npu, pursuer in enumerate(self._pursuers2):
            pursuer.set_velocity(pursuer.velocity + actions_Np_2_team2[npu])
            pursuer.set_position(pursuer.position + pursuer.velocity)

        # Penalize large actions
        if self.reward_mech == 'global':
            rewards_team1 += self.control_penalty * (actions_Np_2_team1**2).sum()
            rewards_team2 += self.control_penalty * (actions_Np_2_team2**2).sum()
        else:
            rewards_team1 += self.control_penalty * (actions_Np_2_team1**2).sum(axis=1)
            rewards_team2+= self.control_penalty * (actions_Np_2_team2**2).sum(axis=1)

        # Players stop on hitting a wall
        self.hit_walls(self._pursuers1)
        self.hit_walls(self._pursuers2)

        #Players rebound on hitting obstacles
        self.rebound_obstacles(self._pursuers1)
        self.rebound_obstacles(self._pursuers2)
        self.rebound_obstacles(self._evaders1)
        self.rebound_obstacles(self._evaders2)
        self.rebound_obstacles(self._poisons)

        # Find collisions
        ev_caught1, who_caught_ev1 = self.caught_particles(self._pursuers1,self._evaders1, self.n_coop)
        ev_caught2, who_caught_ev2 = self.caught_particles(self._pursuers2,self._evaders2, self.n_coop)
        po_caught1, who_caught_po1 = self.caught_particles(self._pursuers1,self._poisons, 1)
        po_caught2, who_caught_po2 = self.caught_particles(self._pursuers2,self._poisons, 1)
        ot_caught1, who_caught_ot1 = self.caught_particles(self._pursuers1, self._pursuers2, 1)
        ot_caught2, who_caught_ot2 = self.caught_particles(self._pursuers2, self._pursuers1, 1)

        # Find sensed objects
        sensorfeatures1 = self.collect_sensed_features(self._pursuers1, self._evaders1, self._pursuers2)
        sensorfeatures2 = self.collect_sensed_features(self._pursuers2, self._evaders2, self._pursuers1)
        # Obstacles
        
        # Process collisions
        # If object collided with required number of players, reset its position and velocity
        # Effectively the same as removing it and adding it back
        self.respawn_dead(self._evaders1, ev_caught1)
        self.respawn_dead(self._evaders2, ev_caught2)
        self.respawn_dead(self._poisons, po_caught1)
        self.respawn_dead(self._poisons, po_caught2)

        ev_encounters1, who_encountered_ev1 = self.encounter_particles(self._pursuers1, self._evaders1)
        ev_encounters2, who_encountered_ev2 = self.encounter_particles(self._pursuers2, self._evaders2)
        # Update reward based on these collisions
        if self.reward_mech == 'global':
            rewards_team1 += (
                (len(ev_caught1) * self.food_reward) + (len(po_caught1) * self.poison_reward) +
                (len(ev_encounters1) * self.encounter_reward) + len(ot_caught1) * self.collision_penalty)
            rewards_team2 += (
                (len(ev_caught2) * self.food_reward) + (len(po_caught2) * self.poison_reward) +
                (len(ev_encounters2) * self.encounter_reward) + len(ot_caught2) * self.collision_penalty)
        else:
            rewards_team1[who_caught_ev1] += self.food_reward
            rewards_team1[who_caught_po1] += self.poison_reward
            rewards_team1[who_encountered_ev1] += self.encounter_reward
            rewards_team2[who_caught_ev2] += self.food_reward
            rewards_team2[who_caught_po2] += self.poison_reward
            rewards_team2[who_encountered_ev2] += self.encounter_reward 
            rewards_team1[who_caught_ot1] += self.collision_penalty
            rewards_team2[who_caught_ot2] += self.collision_penalty
        for evader in self._evaders1:
            # Move objects
            evader.set_position(evader.position + evader.velocity)
            # Bounce object if it hits a wall
            if all(evader.position != np.clip(evader.position, 0, 1)):
                evader.set_velocity(-1 * evader.velocity)

        for evader in self._evaders2:
            # Move objects
            evader.set_position(evader.position + evader.velocity)
            # Bounce object if it hits a wall
            if all(evader.position != np.clip(evader.position, 0, 1)):
                evader.set_velocity(-1 * evader.velocity)

        for poison in self._poisons:
            # Move objects
            poison.set_position(poison.position + poison.velocity)
            # Bounce object if it hits a wall
            if all(poison.position != np.clip(poison.position, 0, 1)):
                poison.set_velocity(-1 * poison.velocity)

        
        obslist1 = self.get_obs_list(self._pursuers1, self._evaders1, self._pursuers2, sensorfeatures1)
        obslist2 = self.get_obs_list(self._pursuers2, self._evaders2, self._pursuers1, sensorfeatures2)
        assert all([
            obs.shape == agent.observation_space.shape for obs, agent in zip(obslist1, self._pursuers1)
        ])
        assert all([
            obs.shape == agent.observation_space.shape for obs, agent in zip(obslist2, self._pursuers2)
        ])
        self._timesteps += 1
        done = self.is_terminal
        info = dict(evcatches1=len(ev_caught1), evcatches2=len(ev_caught2),pocatches1=len(po_caught1),pocatches2=len(po_caught2), otcatches1=len(ot_caught1), otcatches2=len(ot_caught2))
        obslist = np.concatenate((obslist1,obslist2))
        rewards_team = np.concatenate((rewards_team1,rewards_team2))
        # print "SHAPE", obslist.shape, rewards_team.shape
        return obslist, rewards_team, done, info

    def render(self, screen_size=800, rate=10, mode='human'):
        import cv2
        img = np.empty((screen_size, screen_size, 3), dtype=np.uint8)
        img[...] = 255
        # Obstacles
        for iobs, obstaclex_2 in enumerate(self.obstaclesx_No_2):
            assert obstaclex_2.shape == (2,)
            color = (128, 128, 0)
            cv2.circle(img,
                       tuple((obstaclex_2 * screen_size).astype(int)),
                       int(self.obstacle_radius * screen_size), color, -1, lineType=cv2.LINE_AA)
        # Pursuers
        for pursuer in self._pursuers1:
            for k in range(pursuer._n_sensors):
                color = (0, 0, 0)
                cv2.line(img,
                         tuple((pursuer.position * screen_size).astype(int)),
                         tuple(((pursuer.position + pursuer._sensor_range * pursuer.sensors[k]) *
                                screen_size).astype(int)), color, 1, lineType=cv2.LINE_AA)
                cv2.circle(img,
                           tuple((pursuer.position * screen_size).astype(int)),
                           int(pursuer._radius * screen_size), (255, 0, 0), -1, lineType=cv2.LINE_AA)
        for pursuer in self._pursuers2:
            for k in range(pursuer._n_sensors):
                color = (0, 0, 0)
                cv2.line(img,
                         tuple((pursuer.position * screen_size).astype(int)),
                         tuple(((pursuer.position + pursuer._sensor_range * pursuer.sensors[k]) *
                                screen_size).astype(int)), color, 1, lineType=cv2.LINE_AA)
                cv2.circle(img,
                           tuple((pursuer.position * screen_size).astype(int)),
                           int(pursuer._radius * screen_size), (0, 255, 0), -1, lineType=cv2.LINE_AA)
        # Evaders
        for evader in self._evaders1:
            color = (255, 0, 0)
            cv2.circle(img,
                       tuple((evader.position * screen_size).astype(int)),
                       int(evader._radius * screen_size), color, -1, lineType=cv2.LINE_AA)

        for evader in self._evaders2:
            color = (0, 255, 0)
            cv2.circle(img,
                       tuple((evader.position * screen_size).astype(int)),
                       int(evader._radius * screen_size), color, -1, lineType=cv2.LINE_AA)

        # Poison
        for poison in self._poisons:
            color = (0, 0, 255)
            cv2.circle(img,
                       tuple((poison.position * screen_size).astype(int)),
                       int(poison._radius * screen_size), color, -1, lineType=cv2.LINE_AA)

        opacity = 0.4
        bg = np.ones((screen_size, screen_size, 3), dtype=np.uint8) * 255
        cv2.addWeighted(bg, opacity, img, 1 - opacity, 0, img)
        cv2.imshow('Waterworld', img)
        cv2.waitKey(rate)
        return np.asarray(img)[..., ::-1]


if __name__ == '__main__':
    env = MAWaterWorldTeams(2, 2, 10, 10, obs_loc=None)
    obs = env.reset()
    while True:
        obs, rew, _, _ = env.step(env.np_random.randn(8) * .5)
        # raw_input()
        if rew.sum() > 0:
            print(rew)
        env.render()
