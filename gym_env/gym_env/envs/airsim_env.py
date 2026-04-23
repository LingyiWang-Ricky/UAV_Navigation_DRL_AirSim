import gym
from gym import spaces
import airsim
from configparser import NoOptionError
import keyboard

import torch as th
import numpy as np
import math
import cv2
import json
from pathlib import Path

from .dynamics.multirotor_simple import MultirotorDynamicsSimple
from .dynamics.multirotor_airsim import MultirotorDynamicsAirsim
from .dynamics.fixedwing_simple import FixedwingDynamicsSimple
# from .lgmd.LGMD import LGMD

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal


class AirsimGymEnv(gym.Env, QtCore.QThread):
    # pyqt signal for visualization
    action_signal = pyqtSignal(int, np.ndarray)
    state_signal = pyqtSignal(int, np.ndarray)
    attitude_signal = pyqtSignal(int, np.ndarray, np.ndarray)
    reward_signal = pyqtSignal(int, float, float)
    pose_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    lgmd_signal = pyqtSignal(float, float, np.ndarray)

    def __init__(self) -> None:
        super().__init__()
        np.set_printoptions(formatter={'float': '{: 4.2f}'.format},
                            suppress=True)
        th.set_printoptions(profile="short", sci_mode=False, linewidth=1000)
        print("init airsim-gym-env.")
        self.model = None
        self.data_path = None
        self.lgmd = None

    def set_config(self, cfg):
        """get config from .ini file
        """
        self.cfg = cfg
        self.env_name = cfg.get('options', 'env_name')
        self.dynamic_name = cfg.get('options', 'dynamic_name')
        self.keyboard_debug = cfg.getboolean('options', 'keyboard_debug')
        self.generate_q_map = cfg.getboolean('options', 'generate_q_map')
        self.step_log = cfg.getboolean('options', 'step_log', fallback=False)
        self.perception_type = cfg.get('options', 'perception')
        self.num_uavs = cfg.getint('options', 'num_uavs', fallback=1)
        self.task_type = cfg.get('options', 'task_type', fallback='goal_nav')
        self.uav_start_separation = cfg.getfloat('options', 'uav_start_separation', fallback=10.0)
        self.catch_distance = cfg.getfloat('options', 'catch_distance', fallback=5.0)
        self.dual_policy = cfg.getboolean('options', 'dual_policy', fallback=False)
        self.control_role = cfg.get('options', 'control_role', fallback='all')
        self.opponent_model = None
        uav_names_raw = cfg.get('options', 'uav_names', fallback='Drone1,Drone2')
        self.uav_names = [name.strip() for name in uav_names_raw.split(',') if name.strip()]
        if len(self.uav_names) < self.num_uavs:
            self.uav_names += [f"Drone{i+1}" for i in range(len(self.uav_names), self.num_uavs)]
        self._active_uav_idx = None
        self._resolve_uav_names_with_airsim()
        discovered_count = len(getattr(self, 'discovered_uav_names', []))
        if self.num_uavs > 1 and discovered_count > 0 and discovered_count < self.num_uavs:
            raise ValueError(
                f"Configured num_uavs={self.num_uavs}, but AirSim/settings only provide {discovered_count} vehicles: "
                f"{self.discovered_uav_names}. Please add missing UAVs in settings.json."
            )
        print(f"UAV setup -> num_uavs={self.num_uavs}, uav_names={self.uav_names[:self.num_uavs]}, start_separation={self.uav_start_separation}")

        # create LGMD agent
        if self.perception_type == 'lgmd':
            self.lgmd = LGMD(type='origin',  p_threshold=50, s_threshold=0, Ki=2, i_layer_size=3, activate_coeff=1, use_on_off=True)
            self.split_out_last = np.array([0, 0, 0, 0, 0])

        print('Environment: ', self.env_name, "Dynamics: ", self.dynamic_name,
              'Perception: ', self.perception_type)
        if self.num_uavs <= 1 and self.dynamic_name in ['Multirotor', 'SimpleMultirotor']:
            print('[Warning] num_uavs <= 1. Multi-UAV mode is NOT enabled. Check config path and [options] num_uavs.')
        else:
            print(f'[Info] Multi-UAV mode enabled: {self.num_uavs} UAVs -> {self.uav_names[:self.num_uavs]}')

        # set dynamics
        if self.dynamic_name == 'SimpleFixedwing':
            self.dynamic_models = [FixedwingDynamicsSimple(cfg)]
        elif self.dynamic_name == 'SimpleMultirotor':
            self.dynamic_models = [
                MultirotorDynamicsSimple(cfg, vehicle_name=self.uav_names[i] if self.num_uavs > 1 else "")
                for i in range(self.num_uavs)
            ]
        elif self.dynamic_name == 'Multirotor':
            self.dynamic_models = [
                MultirotorDynamicsAirsim(cfg, vehicle_name=self.uav_names[i] if self.num_uavs > 1 else "")
                for i in range(self.num_uavs)
            ]
        else:
            raise Exception("Invalid dynamic_name!", self.dynamic_name)
        self.dynamic_model = self.dynamic_models[0]

        # pursuit settings: first N-1 are pursuers, last one is evader by default
        self.evader_index = cfg.getint('options', 'evader_index', fallback=max(self.num_uavs - 1, 0))
        self.pursuer_indices = [i for i in range(self.num_uavs) if i != self.evader_index]
        if self.task_type == 'pursuit_2v1' and self.num_uavs < 3:
            raise ValueError('task_type=pursuit_2v1 requires num_uavs >= 3')
        self.prev_pursuit_distances = None

        if self.num_uavs > 1 and self.dynamic_name in ['Multirotor', 'SimpleMultirotor']:
            for i, model in enumerate(self.dynamic_models):
                model_name = getattr(model, 'vehicle_name', f'UAV-{i+1}')
                print(f"[UAV Mapping] idx={i} -> vehicle_name='{model_name}'")

        # set start and goal position according to different environment
        if self.env_name == 'NH_center':
            start_position = [0, 0, 5]
            goal_rect = [-128, -128, 128, 128]  # rectangular goal pose
            goal_distance = 90
            self.dynamic_model.set_start(
                start_position, random_angle=math.pi*2)
            self.dynamic_model.set_goal(random_angle=math.pi*2, rect=goal_rect)
            self.work_space_x = [-140, 140]
            self.work_space_y = [-140, 140]
            self.work_space_z = [0.5, 20]
            self.max_episode_steps = 1000
        elif self.env_name == 'NH_tree':
            start_position = [110, 180, 5]
            goal_distance = 90
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model.set_goal(distance=90, random_angle=0)
            self.work_space_x = [start_position[0],
                                 start_position[0] + goal_distance + 10]
            self.work_space_y = [
                start_position[1] - 30, start_position[1] + 30]
            self.work_space_z = [0.5, 10]
            self.max_episode_steps = 400
        elif self.env_name == 'City':
            start_position = [40, -30, 40]
            goal_position = [280, -200, 40]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-100, 350]
            self.work_space_y = [-300, 100]
            self.work_space_z = [0, 100]
            self.max_episode_steps = 400
        elif self.env_name == 'City_400':
            # note: the start and end points will be covered by update_start_and_goal_pose_random function
            start_position = [0, 0, 50]
            goal_position = [280, -200, 50]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-220, 220]
            self.work_space_y = [-220, 220]
            self.work_space_z = [0, 100]
            self.max_episode_steps = 800
        elif self.env_name == 'Tree_200':
            # note: the start and end points will be covered by
            # update_start_and_goal_pose_random function
            start_position = [0, 0, 8]
            goal_position = [280, -200, 50]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-100, 100]
            self.work_space_y = [-100, 100]
            self.work_space_z = [0, 100]
            self.max_episode_steps = 600
        elif self.env_name == 'SimpleAvoid':
            start_position = [0, 0, 5]
            goal_distance = 50
            self.dynamic_model.set_start(
                start_position, random_angle=math.pi*2)
            self.dynamic_model.set_goal(
                distance=goal_distance, random_angle=math.pi*2)
            self.work_space_x = [
                start_position[0] - goal_distance - 10, start_position[0] + goal_distance + 10]
            self.work_space_y = [
                start_position[1] - goal_distance - 10, start_position[1] + goal_distance + 10]
            self.work_space_z = [0.5, 50]
            self.max_episode_steps = 400
        elif self.env_name == 'Forest':
            start_position = [0, 0, 10]
            goal_position = [280, -200, 50]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-100, 100]
            self.work_space_y = [-100, 100]
            self.work_space_z = [0, 100]
            self.max_episode_steps = 300
        elif self.env_name == 'Trees':
            start_position = [0, 0, 5]
            goal_distance = 70
            self.dynamic_model.set_start(
                start_position, random_angle=math.pi*2)
            self.dynamic_model.set_goal(
                distance=goal_distance, random_angle=math.pi*2)
            self.work_space_x = [
                start_position[0] - goal_distance - 10, start_position[0] + goal_distance + 10]
            self.work_space_y = [
                start_position[1] - goal_distance - 10, start_position[1] + goal_distance + 10]
            self.work_space_z = [0.5, 50]
            self.max_episode_steps = 500
        else:
            raise Exception("Invalid env_name!", self.env_name)

        if self.num_uavs > 1:
            for i, dynamic_model in enumerate(self.dynamic_models[1:], start=1):
                start_position = list(self.dynamic_model.start_position)
                start_position[1] += i * self.uav_start_separation
                dynamic_model.start_position = start_position
                dynamic_model.start_random_angle = self.dynamic_model.start_random_angle
                dynamic_model.goal_position = list(self.dynamic_model.goal_position)
                dynamic_model.goal_distance = self.dynamic_model.goal_distance
                dynamic_model.goal_random_angle = self.dynamic_model.goal_random_angle
                dynamic_model.goal_rect = self.dynamic_model.goal_rect

        self.client = self.dynamic_model.client
        self.state_feature_length = self.dynamic_model.state_feature_length
        self.cnn_feature_length = self.cfg.getint('options', 'cnn_feature_num')

        # training state
        self.episode_num = 0
        self.total_step = 0
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.previous_distance_from_des_point = 0

        # other settings
        self.crash_distance = cfg.getint('environment', 'crash_distance')
        self.accept_radius = cfg.getint('environment', 'accept_radius')

        self.max_depth_meters = cfg.getint('environment', 'max_depth_meters')
        self.screen_height = cfg.getint('environment', 'screen_height')
        self.screen_width = cfg.getint('environment', 'screen_width')

        self.trajectory_list = []

        # observation space vector or image
        if self.perception_type == 'vector' or self.perception_type == 'lgmd':
            self.observation_space = spaces.Box(low=0, high=1,
                                                shape=(1,
                                                       self.num_uavs * (self.cnn_feature_length + self.state_feature_length)),
                                                dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(self.screen_height,
                                                       self.screen_width, 2 * self.num_uavs),
                                                dtype=np.uint8)

        self.base_action_space = self.dynamic_model.action_space
        if self.num_uavs == 1:
            self.action_space = self.base_action_space
        else:
            self.action_space = spaces.Box(
                low=np.tile(self.base_action_space.low, self.num_uavs),
                high=np.tile(self.base_action_space.high, self.num_uavs),
                dtype=np.float32)

        if self.dual_policy and self.task_type == 'pursuit_2v1' and self.num_uavs > 1:
            self._update_action_space_for_role()

        self.reward_type = None
        try:
            self.reward_type = cfg.get('options', 'reward_type')
            print('Reward type: ', self.reward_type)
        except NoOptionError:
            self.reward_type = None


    def _resolve_uav_names_with_airsim(self):
        """Validate configured UAV names and auto-remap using AirSim/settings candidates."""
        if self.num_uavs <= 1 or self.dynamic_name not in ['Multirotor', 'SimpleMultirotor']:
            return

        configured_names = self.uav_names[:self.num_uavs]
        candidates = []

        # 1) Query live AirSim vehicles (best source).
        try:
            probe_client = airsim.MultirotorClient()
            probe_client.confirmConnection()
            if hasattr(probe_client, 'listVehicles'):
                candidates = list(probe_client.listVehicles())
            elif hasattr(probe_client, 'simListVehicles'):
                candidates = list(probe_client.simListVehicles())
        except Exception as e:
            print(f"[Warning] Failed to query AirSim vehicle list: {e}")

        # 2) Fallback to known settings json files if AirSim API does not provide vehicle list.
        if not candidates:
            settings_candidates = [
                Path('airsim_settings/settings_multirotor.json'),
                Path.home() / 'Documents' / 'AirSim' / 'settings.json',
            ]
            for sp in settings_candidates:
                try:
                    if not sp.exists():
                        continue
                    data = json.loads(sp.read_text(encoding='utf-8'))
                    vehicles = data.get('Vehicles', {})
                    if isinstance(vehicles, dict):
                        keys = [k for k in vehicles.keys() if isinstance(k, str) and k.strip()]
                        if keys:
                            candidates = keys
                            print(f"[Info] Loaded UAV names from settings: {sp}")
                            break
                except Exception as e:
                    print(f"[Warning] Failed to parse settings file {sp}: {e}")

        if not candidates:
            self.discovered_uav_names = []
            print('[Warning] Could not discover UAV names from AirSim or settings. Keep config uav_names as-is.')
            return

        # keep order + uniqueness
        ordered_candidates = []
        for name in candidates:
            if name not in ordered_candidates:
                ordered_candidates.append(name)
        candidates = ordered_candidates
        self.discovered_uav_names = candidates

        missing_names = [name for name in configured_names if name not in candidates]
        print(f"UAV name discovery -> candidates={candidates}, configured={configured_names}")

        if missing_names:
            print(f"[Warning] Configured UAV names not found in discovered names: {missing_names}.")
            if len(candidates) >= self.num_uavs:
                self.uav_names = candidates[:self.num_uavs]
                print(f"[Fix] Auto-remap uav_names to discovered order: {self.uav_names}")
            else:
                print(f"[Warning] Discovered UAV count {len(candidates)} < num_uavs={self.num_uavs}.")
        elif len(candidates) < self.num_uavs:
            print(f"[Warning] Discovered UAV count {len(candidates)} < num_uavs={self.num_uavs}.")
        else:
            print('[Info] Configured uav_names validated with discovered UAV names.')

    def reset(self):
        # reset state (with retry for invalid spawn states)
        max_reset_retries = 6
        reset_ok = False
        for attempt in range(max_reset_retries):
            if self.task_type == 'pursuit_2v1' and self.num_uavs > 1:
                self._randomize_pursuit_start_positions()

            if self.dynamic_name == 'Multirotor':
                # reset AirSim world only once to avoid repeated global resets when using multiple UAVs
                self.client.reset()
                for dynamic_model in self.dynamic_models:
                    dynamic_model.reset(do_client_reset=False)
            else:
                for dynamic_model in self.dynamic_models:
                    dynamic_model.reset()

            if self.task_type != 'pursuit_2v1' or self.num_uavs <= 1:
                reset_ok = True
                break

            reset_ok = self._is_valid_reset_state()
            if reset_ok:
                break

            print(f"[Reset-Retry] invalid spawn state detected (attempt {attempt+1}/{max_reset_retries}), re-randomizing...")

        if not reset_ok and self.task_type == 'pursuit_2v1' and self.num_uavs > 1:
            print('[Warning] reset retries exhausted; continuing with current spawn state.')

        self.episode_num += 1
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.goal_distance_list = []
        for dynamic_model in self.dynamic_models:
            dynamic_model.goal_distance = dynamic_model.get_distance_to_goal_2d()
            self.goal_distance_list.append(dynamic_model.goal_distance)
        self.previous_distance_from_des_point = float(np.mean(self.goal_distance_list))

        self.trajectory_list = []
        self.last_action_split_list = None
        self.last_position_list = None
        self.episode_uav_rewards = np.zeros(self.num_uavs, dtype=np.float32)

        if self.task_type == 'pursuit_2v1':
            self._update_pursuit_goals()
            self._update_pursuit_distance_cache()

        obs = self.get_obs()

        if self.num_uavs > 1:
            self._check_multi_uav_binding()

        self.last_obs = obs
        return obs

    def _randomize_pursuit_start_positions(self):
        """Randomize episode starts for pursuit task with wider spread and min-distance constraint."""
        if self.num_uavs <= 1:
            return

        margin = 20.0
        min_pair_dist = max(self.uav_start_separation * 1.2, 10.0)
        z = float(self.dynamic_models[0].start_position[2]) if len(self.dynamic_models[0].start_position) > 2 else 5.0

        sampled_xy = []
        for i in range(self.num_uavs):
            chosen = None
            for _ in range(300):
                sx = np.random.uniform(self.work_space_x[0] + margin, self.work_space_x[1] - margin)
                sy = np.random.uniform(self.work_space_y[0] + margin, self.work_space_y[1] - margin)
                if all(np.hypot(sx - px, sy - py) >= min_pair_dist for px, py in sampled_xy):
                    chosen = (sx, sy)
                    break
            if chosen is None:
                # fallback: still keep inside workspace
                sx = np.random.uniform(self.work_space_x[0] + margin, self.work_space_x[1] - margin)
                sy = np.random.uniform(self.work_space_y[0] + margin, self.work_space_y[1] - margin)
                chosen = (sx, sy)
            sampled_xy.append(chosen)

        for i, model in enumerate(self.dynamic_models):
            sx, sy = sampled_xy[i]
            model.start_position = [float(sx), float(sy), z]

    def _is_valid_reset_state(self):
        """Check whether freshly reset UAV states are valid for training."""
        try:
            positions = [np.asarray(model.get_position(), dtype=np.float32) for model in self.dynamic_models]

            # 1) workspace and altitude sanity
            for pos in positions:
                if pos[0] < self.work_space_x[0] or pos[0] > self.work_space_x[1]:
                    return False
                if pos[1] < self.work_space_y[0] or pos[1] > self.work_space_y[1]:
                    return False
                if pos[2] < max(self.work_space_z[0], 0.8):
                    return False

            # 2) pairwise spawn separation (avoid overlap at episode start)
            min_pair_dist = max(self.uav_start_separation * 0.6, 4.0)
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    if np.linalg.norm(positions[i] - positions[j]) < min_pair_dist:
                        return False

            # 3) immediate collision check
            for model in self.dynamic_models:
                try:
                    collision_info = model.client.simGetCollisionInfo(vehicle_name=getattr(model, 'vehicle_name', ''))
                except TypeError:
                    collision_info = model.client.simGetCollisionInfo()
                if collision_info.has_collided:
                    return False

            return True
        except Exception:
            return False

    def _check_multi_uav_binding(self):
        if self.num_uavs <= 1:
            return
        try:
            pos_list = [np.asarray(model.get_position(), dtype=np.float32) for model in self.dynamic_models]
            print('[BindingCheck] positions after reset:', pos_list)
            if len(pos_list) >= 2:
                all_same = all(np.allclose(pos_list[0], p, atol=1e-2) for p in pos_list[1:])
                if all_same:
                    print('[Warning] All UAV positions are identical right after reset. This usually means vehicle_name binding is incorrect.')
        except Exception as e:
            print(f"[Warning] _check_multi_uav_binding failed: {e}")

    def step(self, action):
        if self.dual_policy and self.task_type == 'pursuit_2v1' and self.num_uavs > 1 and self.control_role in ['pursuer', 'evader']:
            action = self._compose_full_action_for_dual_policy(action)

        # set action
        if self.num_uavs == 1 and self.dynamic_name == 'SimpleFixedwing':
            # add step to calculate pitch flap deg Fixed wing only
            self.dynamic_model.set_action(action, self.step_num)
        elif self.num_uavs == 1:
            self.dynamic_model.set_action(action)
        else:
            action, action_split_list = self._split_multi_uav_action(action)
            position_ue4 = []
            for i, dynamic_model in enumerate(self.dynamic_models):
                action_i = action_split_list[i]
                dynamic_model.set_action(action_i)
                position_ue4.append(dynamic_model.get_position())
            self.trajectory_list.append(position_ue4)
            self.last_action_split_list = action_split_list
            self.last_position_list = position_ue4

            action_pos_map = self.get_uav_action_position_map(action_split_list, position_ue4)

            # runtime warning for common misconfiguration: both names mapped to same vehicle
            if len(position_ue4) >= 2:
                ref = np.asarray(position_ue4[0], dtype=np.float32)
                same_pose = True
                for p in position_ue4[1:]:
                    if not np.allclose(ref, np.asarray(p, dtype=np.float32), atol=1e-3):
                        same_pose = False
                        break
                if same_pose and self.step_num % 20 == 0:
                    print("[Warning] Multi-UAV positions are identical at this step. Check AirSim vehicle_name mapping.")

        if self.num_uavs == 1:
            position_ue4 = self.dynamic_model.get_position()
            self.trajectory_list.append(position_ue4)

        if self.task_type == 'pursuit_2v1' and self.num_uavs > 1:
            self._update_pursuit_goals()

        # get new obs
        obs = self.get_obs()
        self.last_obs = obs
        done = self.is_done()
        info = {
            'is_success': self.is_in_desired_pose(),
            'is_crash': self.is_crashed(),
            'is_not_in_workspace': self.is_not_inside_workspace(),
            'step_num': self.step_num
        }
        if self.num_uavs > 1:
            info['uav_action_position_map'] = self.get_uav_action_position_map(action_split_list, position_ue4)
        if done:
            self.print_episode_summary(info)

        # ----------------compute reward---------------------------
        if self.num_uavs == 1 and self.dynamic_name == 'SimpleFixedwing':
            # reward = self.compute_reward_fixedwing(done, action)
            reward = self.compute_reward_final_fixedwing(done, action)
        elif self.num_uavs > 1:
            if self.task_type == 'pursuit_2v1':
                reward = self.compute_pursuit_reward(done)
                if self.dual_policy and self.control_role == 'pursuer':
                    reward = self.last_pursuer_reward
                elif self.dual_policy and self.control_role == 'evader':
                    reward = self.last_evader_reward
            else:
                reward = self.compute_multi_uav_reward(done, action)
        elif self.reward_type == 'reward_with_action':
            reward = self.compute_reward_with_action(done, action)
        elif self.reward_type == 'reward_new':
            reward = self.compute_reward_multirotor_new(done, action)
        elif self.reward_type == 'reward_lqr':
            reward = self.compute_reward_lqr(done, action)
        elif self.reward_type == 'reward_final':
            reward = self.compute_reward_final(done, action)
        else:
            reward = self.compute_reward(done, action)

        self.cumulated_episode_reward += reward

        if self.num_uavs > 1:
            if self.task_type == 'pursuit_2v1':
                pursuer_rewards = [self.last_pursuer_reward for _ in self.pursuer_indices]
                role_rewards = pursuer_rewards + [self.last_evader_reward]
                self.last_multi_uav_reward_list = role_rewards
            if hasattr(self, 'last_multi_uav_reward_list') and len(self.last_multi_uav_reward_list) >= self.num_uavs:
                self.episode_uav_rewards += np.asarray(self.last_multi_uav_reward_list[:self.num_uavs], dtype=np.float32)
        else:
            self.episode_uav_rewards[0] += float(reward)

        # ----------------print info---------------------------
        if self.step_log:
            self.print_train_info_airsim(action, obs, reward, info)

        if self.cfg.get('options', 'dynamic_name') == 'SimpleFixedwing':
            self.set_pyqt_signal_fixedwing(action, reward, done)
        else:
            self.set_pyqt_signal_multirotor(action, reward)

        if self.keyboard_debug:
            action_copy = np.copy(action)
            action_copy[-1] = math.degrees(action_copy[-1])
            state_copy = np.copy(self.dynamic_model.state_raw)

            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print(
                '=============================================================================')
            print('episode', self.episode_num, 'step',
                  self.step_num, 'total step', self.total_step)
            print('action', action_copy)
            print('state', state_copy)
            print('state_norm', self.dynamic_model.state_norm)
            print('reward {:.3f} {:.3f}'.format(
                reward, self.cumulated_episode_reward))
            print('done', done)
            keyboard.wait('a')

        if self.generate_q_map and (self.cfg.get('options', 'algo') == 'TD3' or self.cfg.get('options', 'algo') == 'SAC'):
            if self.model is not None:
                with th.no_grad():
                    # get q-value for td3
                    obs_copy = obs.copy()
                    if self.perception_type != 'vector':
                        obs_copy = obs_copy.swapaxes(0, 1)
                        obs_copy = obs_copy.swapaxes(0, 2)
                    q_value_current = self.model.critic(th.from_numpy(obs_copy[tuple(
                        [None])]).float().cuda(), th.from_numpy(action[None]).float().cuda())
                    q_1 = q_value_current[0].cpu().numpy()[0]
                    q_2 = q_value_current[1].cpu().numpy()[0]

                    q_value = min(q_1, q_2)[0]

                    self.visual_log_q_value(q_value, action, reward)

        self.step_num += 1
        self.total_step += 1

        return obs, reward, done, info

    def _split_multi_uav_action(self, action):
        """Normalize multi-UAV action and split it per UAV.

        Accept action in either flat shape ``(num_uavs * action_dim,)`` or
        matrix shape ``(num_uavs, action_dim)``.
        """
        action_dim = self.dynamic_models[0].action_space.shape[0]
        action_arr = np.asarray(action, dtype=np.float32)

        if action_arr.ndim == 2:
            if action_arr.shape == (self.num_uavs, action_dim):
                action_arr = action_arr.reshape(-1)
            elif action_arr.shape == (action_dim, self.num_uavs):
                action_arr = action_arr.T.reshape(-1)
            else:
                raise ValueError(
                    f"Invalid multi-uav action shape {action_arr.shape}, "
                    f"expected ({self.num_uavs}, {action_dim}) or flat vector."
                )
        elif action_arr.ndim != 1:
            raise ValueError(f"Invalid multi-uav action ndim {action_arr.ndim}, expected 1 or 2.")

        expected_size = self.num_uavs * action_dim
        if action_arr.size != expected_size:
            raise ValueError(f"Invalid multi-uav action size {action_arr.size}, expected {expected_size}.")

        action_split_list = [action_arr[i*action_dim:(i+1)*action_dim] for i in range(self.num_uavs)]

        return action_arr, action_split_list

    def _update_action_space_for_role(self):
        action_dim = self.base_action_space.shape[0]
        if self.control_role == 'pursuer':
            count = len(self.pursuer_indices)
        elif self.control_role == 'evader':
            count = 1
        else:
            count = self.num_uavs
        self.action_space = spaces.Box(
            low=np.tile(self.base_action_space.low, count),
            high=np.tile(self.base_action_space.high, count),
            dtype=np.float32)

    def set_control_role(self, role):
        self.control_role = role
        if self.dual_policy and self.task_type == 'pursuit_2v1' and self.num_uavs > 1:
            self._update_action_space_for_role()

    def set_opponent_model(self, model):
        self.opponent_model = model

    def _sample_or_predict_opponent_action(self, expected_agent_count=1):
        action_dim = self.base_action_space.shape[0]
        expected_size = expected_agent_count * action_dim
        if self.opponent_model is None or not hasattr(self, 'last_obs'):
            return np.concatenate([self.base_action_space.sample() for _ in range(expected_agent_count)], axis=0)
        try:
            action, _ = self.opponent_model.predict(self.last_obs, deterministic=False)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.size >= expected_size:
                return action[:expected_size]
        except Exception:
            pass
        return np.concatenate([self.base_action_space.sample() for _ in range(expected_agent_count)], axis=0)

    def _compose_full_action_for_dual_policy(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action_dim = self.base_action_space.shape[0]
        full_action = np.zeros(self.num_uavs * action_dim, dtype=np.float32)

        if self.control_role == 'pursuer':
            pursuer_action = action.reshape(len(self.pursuer_indices), action_dim)
            for k, idx in enumerate(self.pursuer_indices):
                full_action[idx*action_dim:(idx+1)*action_dim] = pursuer_action[k]
            evader_act = self._sample_or_predict_opponent_action(expected_agent_count=1)
            full_action[self.evader_index*action_dim:(self.evader_index+1)*action_dim] = evader_act[:action_dim]
        elif self.control_role == 'evader':
            evader_act = action[:action_dim]
            full_action[self.evader_index*action_dim:(self.evader_index+1)*action_dim] = evader_act
            pursuer_actions = self._sample_or_predict_opponent_action(expected_agent_count=len(self.pursuer_indices))
            pursuer_actions = pursuer_actions.reshape(len(self.pursuer_indices), action_dim)
            for k, idx in enumerate(self.pursuer_indices):
                full_action[idx*action_dim:(idx+1)*action_dim] = pursuer_actions[k]
        else:
            full_action = action

        return full_action

    def _update_pursuit_goals(self):
        if self.task_type != 'pursuit_2v1' or self.num_uavs <= 1:
            return

        evader_pos = np.asarray(self.dynamic_models[self.evader_index].get_position(), dtype=np.float32)

        # pursuers chase evader
        for idx in self.pursuer_indices:
            self.dynamic_models[idx].goal_position = evader_pos.tolist()

        # evader goal: move away from pursuers centroid
        pursuer_positions = [np.asarray(self.dynamic_models[idx].get_position(), dtype=np.float32)
                             for idx in self.pursuer_indices]
        pursuer_center = np.mean(pursuer_positions, axis=0)
        away_vec = evader_pos - pursuer_center
        norm = np.linalg.norm(away_vec[:2])
        if norm < 1e-3:
            away_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            norm = 1.0
        # For evader policy, use pursuer centroid as dynamic reference so state features
        # directly encode threat geometry (distance/yaw to pursuers).
        evader_goal = np.array([
            np.clip(pursuer_center[0], self.work_space_x[0], self.work_space_x[1]),
            np.clip(pursuer_center[1], self.work_space_y[0], self.work_space_y[1]),
            np.clip(evader_pos[2], self.work_space_z[0], self.work_space_z[1])
        ], dtype=np.float32)
        self.dynamic_models[self.evader_index].goal_position = evader_goal.tolist()

    def _get_pursuit_distances(self):
        evader_pos = np.asarray(self.dynamic_models[self.evader_index].get_position(), dtype=np.float32)
        distances = []
        for idx in self.pursuer_indices:
            pursuer_pos = np.asarray(self.dynamic_models[idx].get_position(), dtype=np.float32)
            distances.append(float(np.linalg.norm(pursuer_pos - evader_pos)))
        return distances

    def _update_pursuit_distance_cache(self):
        if self.task_type != 'pursuit_2v1':
            return
        self.prev_pursuit_distances = self._get_pursuit_distances()

    def compute_pursuit_reward(self, done):
        if self.prev_pursuit_distances is None:
            self._update_pursuit_distance_cache()

        curr_distances = self._get_pursuit_distances()
        prev_distances = self.prev_pursuit_distances
        caught = any(d <= self.catch_distance for d in curr_distances)

        pursuer_rewards = []
        closest_dist = min(curr_distances)
        near_catch_bonus = 0.0
        if closest_dist < 1.5 * self.catch_distance:
            near_catch_bonus = 2.0 * (1.5 * self.catch_distance - closest_dist) / max(self.catch_distance, 1e-3)

        for prev_d, curr_d in zip(prev_distances, curr_distances):
            # dense chase shaping + urgency penalty
            r = 3.0 * (prev_d - curr_d) - 0.05 + near_catch_bonus
            if caught:
                r += 30.0
            pursuer_rewards.append(r)

        prev_mean = float(np.mean(prev_distances))
        curr_mean = float(np.mean(curr_distances))
        evader_reward = 3.0 * (curr_mean - prev_mean) + 0.05
        if caught:
            evader_reward -= 30.0

        self.prev_pursuit_distances = curr_distances

        self.last_pursuer_reward = float(np.mean(pursuer_rewards))
        self.last_evader_reward = float(evader_reward)

        # terminal shaping to discourage collision-heavy policies
        if done and not caught:
            if self.is_crashed() or self.is_not_inside_workspace():
                self.last_pursuer_reward -= 20.0
                self.last_evader_reward -= 20.0
            elif self.step_num >= self.max_episode_steps:
                # timeout means evader survived
                self.last_pursuer_reward -= 5.0
                self.last_evader_reward += 10.0

        # single scalar for centralized policy / fallback
        return float(self.last_pursuer_reward + self.last_evader_reward)

    def get_uav_action_position_map(self, action_split_list=None, position_list=None):
        if self.num_uavs <= 1:
            return None

        if action_split_list is None:
            action_split_list = getattr(self, 'last_action_split_list', None)
        if position_list is None:
            position_list = getattr(self, 'last_position_list', None)
        if action_split_list is None or position_list is None:
            return None

        mapping = {}
        for i in range(min(len(action_split_list), len(position_list), self.num_uavs)):
            uav_name = self.uav_names[i] if i < len(self.uav_names) else f"Drone{i+1}"
            mapping[uav_name] = {
                'action': np.asarray(action_split_list[i]).tolist(),
                'position': np.asarray(position_list[i]).tolist()
            }
        return mapping

# ! -------------------------get obs------------------------------------------
    def get_obs(self):
        if self.num_uavs > 1:
            self.min_distance_to_obstacles_all = []
            if self.perception_type == 'vector':
                obs_all = [self.get_obs_vector_single(dynamic_model) for dynamic_model in self.dynamic_models]
                return np.concatenate(obs_all, axis=1)
            obs_all = [self.get_obs_image_single(dynamic_model) for dynamic_model in self.dynamic_models]
            return np.concatenate(obs_all, axis=2)

        if self.perception_type == 'vector':
            obs = self.get_obs_vector()
        elif self.perception_type == 'lgmd':
            obs = self.get_obs_lgmd()
        else:
            obs = self.get_obs_image()

        return obs

    def get_obs_image_single(self, dynamic_model):
        image = self.get_depth_image(client=dynamic_model.client, vehicle_name=getattr(dynamic_model, 'vehicle_name', ''))
        self.min_distance_to_obstacles_all.append(image.min())
        image_resize = cv2.resize(image, (self.screen_width,
                                          self.screen_height))
        image_scaled = np.clip(
            image_resize, 0, self.max_depth_meters) / self.max_depth_meters * 255
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)

        state_feature_array = np.zeros((self.screen_height, self.screen_width))
        state_feature = dynamic_model._get_state_feature()
        state_feature_array[0, 0:self.state_feature_length] = state_feature

        image_with_state = np.array([image_uint8, state_feature_array])
        image_with_state = image_with_state.swapaxes(0, 2)
        image_with_state = image_with_state.swapaxes(0, 1)
        return image_with_state

    def get_obs_vector_single(self, dynamic_model):
        image = self.get_depth_image(client=dynamic_model.client, vehicle_name=getattr(dynamic_model, 'vehicle_name', ''))
        self.min_distance_to_obstacles_all.append(image.min())
        image_scaled = np.clip(image, 0, self.max_depth_meters) / self.max_depth_meters * 255
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)

        image_obs = image_uint8
        split_row = 1
        split_col = 5

        v_split_list = np.vsplit(image_obs, split_row)

        split_final = []
        for i in range(split_row):
            h_split_list = np.hsplit(v_split_list[i], split_col)
            for j in range(split_col):
                split_final.append(h_split_list[j].max())

        img_feature = np.array(split_final) / 255.0
        state_feature = dynamic_model._get_state_feature() / 255
        feature_all = np.concatenate((img_feature, state_feature), axis=0)
        feature_all = np.reshape(feature_all, (1, len(feature_all)))
        return feature_all

    def get_obs_image(self):
        # Normal mode: get depth image then transfer to matrix with state
        # 1. get current depth image and transfer to 0-255  0-20m 255-0m
        image = self.get_depth_image()  # 0-6550400.0 float 32
        image_resize = cv2.resize(image, (self.screen_width,
                                          self.screen_height))
        self.min_distance_to_obstacles = image.min()
        # switch 0 and 255
        image_scaled = np.clip(
            image_resize, 0, self.max_depth_meters) / self.max_depth_meters * 255
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)

        # 2. get current state (relative_pose, velocity)
        state_feature_array = np.zeros((self.screen_height, self.screen_width))
        state_feature = self.dynamic_model._get_state_feature()
        state_feature_array[0, 0:self.state_feature_length] = state_feature

        # 3. generate image with state
        image_with_state = np.array([image_uint8, state_feature_array])
        image_with_state = image_with_state.swapaxes(0, 2)
        image_with_state = image_with_state.swapaxes(0, 1)

        return image_with_state

    def get_depth_gray_image(self):
        # get depth and rgb image
        # scene vision image in png format
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True),
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        ])

        # check observation
        while responses[0].width == 0:
            print("get_image_fail...")
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, True),
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            ])

        # get depth image
        depth_img = airsim.list_to_2d_float_array(
            responses[0].image_data_float,
            responses[0].width, responses[0].height)
        depth_meter = depth_img * 100

        # get gary image
        img_1d = np.fromstring(responses[1].image_data_uint8, dtype=np.uint8)
        # reshape array to 4 channel image array H X W X 3
        img_rgb = img_1d.reshape(responses[1].height, responses[1].width, 3)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        # cv2.imshow('test', img_rgb)
        # cv2.waitKey(1)

        return depth_meter, img_gray

    def get_depth_image(self, client=None, vehicle_name=''):
        if client is None:
            client = self.client

        try:
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)
            ], vehicle_name=vehicle_name)
        except TypeError:
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)
            ])

        # check observation
        while responses[0].width == 0:
            print("get_image_fail...")
            try:
                responses = client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)
                ], vehicle_name=vehicle_name)
            except TypeError:
                responses = client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)
                ])

        depth_img = airsim.list_to_2d_float_array(
            responses[0].image_data_float, responses[0].width,
            responses[0].height)

        depth_meter = depth_img * 100

        return depth_meter

    def get_obs_vector(self):

        image = self.get_depth_image()  # 0-6550400.0 float 32
        self.min_distance_to_obstacles = image.min()

        image_scaled = np.clip(image, 0, self.max_depth_meters) / self.max_depth_meters * 255
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)

        image_obs = image_uint8
        split_row = 1
        split_col = 5

        v_split_list = np.vsplit(image_obs, split_row)

        split_final = []
        for i in range(split_row):
            h_split_list = np.hsplit(v_split_list[i], split_col)
            for j in range(split_col):
                split_final.append(h_split_list[j].max())

        img_feature = np.array(split_final) / 255.0

        state_feature = self.dynamic_model._get_state_feature() / 255

        feature_all = np.concatenate((img_feature, state_feature), axis=0)

        self.feature_all = feature_all

        feature_all = np.reshape(feature_all, (1, len(feature_all)))

        return feature_all

    def get_obs_lgmd(self):
        # get depth and gray image
        depth_meter, img_gray = self.get_depth_gray_image()
        self.min_distance_to_obstacles = depth_meter.min()

        self.lgmd.update(img_gray)

        split_col_num = 5
        s_layer = self.lgmd.s_layer  # (192, 320)
        s_layer_split = np.hsplit(s_layer, split_col_num)  # (192, 109)

        lgmd_out_list = []
        activate_coeff = 0.5
        for i in range(split_col_num):
            s_layer_activated_sum = abs(np.sum(s_layer_split[i]))
            Kf = -(s_layer_activated_sum * activate_coeff) / (192*64)  # 0 - 1
            a = np.exp(Kf)
            lgmd_out_norm = (1 / (1 + a) - 0.5) * 2
            lgmd_out_list.append(lgmd_out_norm)

        # show iamges
        heatmapshow = None
        heatmapshow = cv2.normalize(s_layer, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        cv2.imshow('gray image', img_gray)
        cv2.imshow('depth image', np.clip(depth_meter, 0, 255)/255)
        cv2.imshow('s-layer', heatmapshow)
        cv2.waitKey(1)

        # update LGMD
        split_final = np.array(lgmd_out_list)
        
        filter_coeff = 0.8
        split_final_filter = filter_coeff * split_final + (1-filter_coeff) * self.split_out_last
        self.split_out_last = split_final_filter

        img_feature = np.array(split_final_filter)

        state_feature = self.dynamic_model._get_state_feature() / 255

        feature_all = np.concatenate((img_feature, state_feature), axis=0)

        self.feature_all = feature_all

        feature_all = np.reshape(feature_all, (1, len(feature_all)))

        return feature_all

    def _get_active_dynamic_model(self):
        if self._active_uav_idx is None:
            return self.dynamic_model
        return self.dynamic_models[self._active_uav_idx]

    def _get_active_min_distance_to_obstacles(self):
        if self.num_uavs <= 1:
            return getattr(self, 'min_distance_to_obstacles', float(self.max_depth_meters))

        idx = self._active_uav_idx if self._active_uav_idx is not None else 0
        depth_all = getattr(self, 'min_distance_to_obstacles_all', None)
        if depth_all is not None and len(depth_all) > idx:
            return float(depth_all[idx])

        # fallback for unexpected flow
        return getattr(self, 'min_distance_to_obstacles', float(self.max_depth_meters))

    def compute_multi_uav_reward(self, done, action):
        action = np.asarray(action)
        action_dim = self.dynamic_models[0].action_space.shape[0]
        reward_list = []
        for i in range(self.num_uavs):
            self._active_uav_idx = i
            action_i = action[i*action_dim:(i+1)*action_dim]
            if self.reward_type == 'reward_with_action':
                reward_i = self.compute_reward_with_action(done, action_i)
            elif self.reward_type == 'reward_new':
                reward_i = self.compute_reward_multirotor_new(done, action_i)
            elif self.reward_type == 'reward_lqr':
                reward_i = self.compute_reward_lqr(done, action_i)
            elif self.reward_type == 'reward_final':
                reward_i = self.compute_reward_final(done, action_i)
            else:
                reward_i = self.compute_reward(done, action_i)
            reward_list.append(reward_i)

        self._active_uav_idx = None
        self.last_multi_uav_reward_list = reward_list
        return float(np.mean(reward_list))
# ! ---------------------calculate rewards-------------------------------------

    def compute_reward(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = -10

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now) / \
                self.dynamic_model.goal_distance * \
                500  # normalized to 100 according to goal_distance
            self.previous_distance_from_des_point = distance_now

            reward_obs = 0
            action_cost = 0

            # add yaw_rate cost
            yaw_speed_cost = 0.1 * \
                abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad

            if self.dynamic_model.navigation_3d:
                # add action and z error cost
                v_z_cost = 0.1 * \
                    ((abs(action[1]) / self.dynamic_model.v_z_max)**2)
                z_err_cost = 0.05 * \
                    ((abs(
                        self.dynamic_model.state_raw[1]) / self.dynamic_model.max_vertical_difference)**2)
                action_cost += (v_z_cost + z_err_cost)

            action_cost += yaw_speed_cost

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = 0.1 * abs(yaw_error / 180)

            reward = reward_distance - reward_obs - action_cost - yaw_error_cost
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_final(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = -10
        dynamic_model = self._get_active_dynamic_model()

        if self.env_name == 'NH_center':
            distance_reward_coef = 500
        else:
            distance_reward_coef = 50

        if not done:
            # 1 - goal reward
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = distance_reward_coef * (self.previous_distance_from_des_point - distance_now) /                 dynamic_model.goal_distance   # normalized to 100 according to goal_distance
            self.previous_distance_from_des_point = distance_now

            # 2 - Position punishment
            current_pose = dynamic_model.get_position()
            goal_pose = dynamic_model.goal_position
            x = current_pose[0]
            y = current_pose[1]
            z = current_pose[2]
            x_g = goal_pose[0]
            y_g = goal_pose[1]
            z_g = goal_pose[2]

            punishment_xy = np.clip(self.getDis(
                x, y, 0, 0, x_g, y_g) / 10, 0, 1)
            punishment_z = 0.5 * np.clip((z - z_g)/5, 0, 1)

            punishment_pose = punishment_xy + punishment_z

            min_depth = self._get_active_min_distance_to_obstacles()
            if min_depth < 10:
                punishment_obs = 1 - np.clip((min_depth - self.crash_distance) / 5, 0, 1)
            else:
                punishment_obs = 0

            punishment_action = 0

            # add yaw_rate cost
            yaw_speed_cost = abs(action[-1]) / dynamic_model.yaw_rate_max_rad

            if dynamic_model.navigation_3d:
                # add action and z error cost
                v_z_cost = ((abs(action[1]) / dynamic_model.v_z_max)**2)
                z_err_cost = (
                    (abs(dynamic_model.state_raw[1]) / dynamic_model.max_vertical_difference)**2)
                punishment_action += (v_z_cost + z_err_cost)

            punishment_action += yaw_speed_cost

            yaw_error = dynamic_model.state_raw[2]
            yaw_error_cost = abs(yaw_error / 90)

            reward = reward_distance - 0.1 * punishment_pose - 0.2 *                 punishment_obs - 0.1 * punishment_action - 0.5 * yaw_error_cost
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_final_fixedwing(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = -10

        if not done:
            # 1 - goal reward
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = 300 * (self.previous_distance_from_des_point - distance_now) / \
                self.dynamic_model.goal_distance   # normalized to 100 according to goal_distance
            self.previous_distance_from_des_point = distance_now

            # 2 - Position punishment
            current_pose = self.dynamic_model.get_position()
            goal_pose = self.dynamic_model.goal_position
            x = current_pose[0]
            y = current_pose[1]
            x_g = goal_pose[0]
            y_g = goal_pose[1]

            punishment_xy = np.clip(self.getDis(
                x, y, 0, 0, x_g, y_g) / 50, 0, 1)
            # punishment_z = 0.5 * np.clip((z - z_g)/5, 0, 1)

            punishment_pose = punishment_xy

            if self.min_distance_to_obstacles < 20:
                punishment_obs = 1 - np.clip((self.min_distance_to_obstacles - self.crash_distance) / 15, 0, 1)
            else:
                punishment_obs = 0

            # action cost
            punishment_action = abs(action[0]) / self.dynamic_model.roll_rate_max

            yaw_error = self.dynamic_model.state_raw[1]
            yaw_error_cost = abs(yaw_error / 90)

            reward = reward_distance - 0.1 * punishment_pose - 0.5 * \
                punishment_obs - 0.1 * punishment_action - 0.1 * yaw_error_cost
            # reward = reward

            # print("r_dist: {:.2f} p_pose: {:.2f} p_obs: {:.2f} p_action: {:.2f}, p_yaw_e: {:.2f}".format(reward_distance, punishment_pose, punishment_obs, punishment_action, yaw_error_cost))
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_test(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -100
        reward_outside = -10

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now) / \
                self.dynamic_model.goal_distance * \
                100  # normalized to 100 according to goal_distance
            self.previous_distance_from_des_point = distance_now

            reward_obs = 0
            action_cost = 0

            # add yaw_rate cost
            yaw_speed_cost = 0.1 * \
                abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad

            if self.dynamic_model.navigation_3d:
                # add action and z error cost
                v_z_cost = 0.1 * abs(action[1]) / self.dynamic_model.v_z_max
                z_err_cost = 0.05 * \
                    abs(self.dynamic_model.state_raw[1]) / \
                    self.dynamic_model.max_vertical_difference
                action_cost += (v_z_cost + z_err_cost)

            action_cost += yaw_speed_cost

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = 0.1 * abs(yaw_error / 180)

            reward = reward_distance - reward_obs - action_cost - yaw_error_cost
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_fixedwing(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -50
        reward_outside = -10

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now) / \
                self.dynamic_model.goal_distance * \
                300  # normalized to 100 according to goal_distance
            self.previous_distance_from_des_point = distance_now

            # 只有action cost和obs cost
            # 由于没有速度控制，所以前面那个也取消了
            # action_cost = 0
            # obs_cost = 0

            # relative_yaw_cost = abs(
            #     (self.dynamic_model.state_norm[0]/255-0.5) * 2)
            # action_cost = abs(action[0]) / self.dynamic_model.roll_rate_max

            # obs_punish_distance = 15
            # if self.min_distance_to_obstacles < obs_punish_distance:
            #     obs_cost = 1 - (self.min_distance_to_obstacles -
            #                     self.crash_distance) / (obs_punish_distance -
            #                                             self.crash_distance)
            #     obs_cost = 0.5 * obs_cost ** 2
            # reward = reward_distance - (2 * relative_yaw_cost + 0.5 * action_cost + obs_cost)

            action_cost = abs(action[0]) / self.dynamic_model.roll_rate_max

            yaw_error_deg = self.dynamic_model.state_raw[1]
            yaw_error_cost = 0.1 * abs(yaw_error_deg / 180)

            reward = reward_distance - action_cost - yaw_error_cost
        else:
            if self.is_in_desired_pose():
                yaw_error_deg = self.dynamic_model.state_raw[1]
                reward = reward_reach * (1 - abs(yaw_error_deg / 180))
                # reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_multirotor_new(self, done, action):
        reward = 0
        reward_reach = 100
        reward_crash = -100
        reward_outside = 0

        if not done:
            dynamic_model = self._get_active_dynamic_model()
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point -
                               distance_now) / dynamic_model.goal_distance * 5
            self.previous_distance_from_des_point = distance_now

            state_cost = 0
            action_cost = 0
            obs_cost = 0

            yaw_error_deg = dynamic_model.state_raw[1]

            relative_yaw_cost = abs(yaw_error_deg/180)
            action_cost = abs(action[1]) / dynamic_model.yaw_rate_max_rad

            obs_punish_dist = 5
            min_depth = self._get_active_min_distance_to_obstacles()
            if min_depth < obs_punish_dist:
                obs_cost = 1 - (min_depth -
                                self.crash_distance) / (obs_punish_dist - self.crash_distance)
                obs_cost = 0.5 * obs_cost ** 2
            reward = - (2 * relative_yaw_cost + 0.5 * action_cost)
        else:
            if self.is_in_desired_pose():
                # 到达之后根据yaw偏差对reward进行scale
                reward = reward_reach * \
                    (1 - abs(self.dynamic_model.state_norm[1]))
                # reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_with_action(self, done, action):
        reward = 0
        reward_reach = 50
        reward_crash = -50
        reward_outside = -10

        step_cost = 0.01  # 10 for max 1000 steps

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now) / \
                self.dynamic_model.goal_distance * \
                10  # normalized to 100 according to goal_distance
            self.previous_distance_from_des_point = distance_now

            reward_obs = 0
            action_cost = 0

            # add action cost
            # speed 0-8  cruise speed is 4, punish for too fast and too slow
            v_xy_cost = 0.02 * abs(action[0]-5) / 4
            yaw_rate_cost = 0.02 * \
                abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad
            if self.dynamic_model.navigation_3d:
                v_z_cost = 0.02 * abs(action[1]) / self.dynamic_model.v_z_max
                action_cost += v_z_cost
            action_cost += (v_xy_cost + yaw_rate_cost)

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = 0.05 * abs(yaw_error/180)

            reward = reward_distance - reward_obs - action_cost - yaw_error_cost
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_lqr(self, done, action):
        # 模仿matlab提供的mix reward的思想设计
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = 0

        if not done:
            action_cost = 0
            # add yaw_rate cost
            yaw_speed_cost = 0.2 * \
                ((action[-1] / self.dynamic_model.yaw_rate_max_rad) ** 2)

            if self.dynamic_model.navigation_3d:
                # add action and z error cost
                v_z_cost = 0.1 * ((action[1] / self.dynamic_model.v_z_max)**2)
                z_err_cost = 0.1 * \
                    ((self.dynamic_model.state_raw[1] /
                      self.dynamic_model.max_vertical_difference)**2)
                action_cost += (v_z_cost + z_err_cost)

            action_cost += yaw_speed_cost

            yaw_error_clip = min(
                max(-60, self.dynamic_model.state_raw[2]), 60) / 60
            yaw_error_cost = 1.0 * (yaw_error_clip**2)

            reward = - (action_cost + yaw_error_cost)

            # print('r: {:.2f} y_r: {:.2f} y_e: {:.2f} z_r: {:.2f} z_e: {:.2f}'.format(reward, yaw_speed_cost, yaw_error_cost, v_z_cost, z_err_cost))
        else:
            if self.is_in_desired_pose():
                yaw_error_clip = min(
                    max(-30, self.dynamic_model.state_raw[2]), 30) / 30
                reward = reward_reach * (1 - yaw_error_clip**2)
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

# ! ------------------ is done-----------------------------------------------

    def is_done(self):
        episode_done = False

        is_not_inside_workspace_now = self.is_not_inside_workspace()
        has_reached_des_pose = self.is_in_desired_pose()
        too_close_to_obstable = self.is_crashed()

        # We see if we are outside the Learning Space
        episode_done = is_not_inside_workspace_now or\
            has_reached_des_pose or\
            too_close_to_obstable or\
            self.step_num >= self.max_episode_steps

        return episode_done

    def is_not_inside_workspace(self):
        """
        Check if the Drone is inside the Workspace defined
        """
        is_not_inside = False
        model_list = self.dynamic_models
        if self._active_uav_idx is not None:
            model_list = [self.dynamic_models[self._active_uav_idx]]

        for model in model_list:
            current_position = model.get_position()
            if current_position[0] < self.work_space_x[0] or current_position[0] > self.work_space_x[1] or \
                current_position[1] < self.work_space_y[0] or current_position[1] > self.work_space_y[1] or \
                    current_position[2] < self.work_space_z[0] or current_position[2] > self.work_space_z[1]:
                is_not_inside = True
                break

        return is_not_inside

    def is_in_desired_pose(self):
        if self.task_type == 'pursuit_2v1' and self.num_uavs > 1:
            distances = self._get_pursuit_distances()
            return any(d <= self.catch_distance for d in distances)

        if self._active_uav_idx is not None:
            return self.get_distance_to_goal_3d() < self.accept_radius

        in_desired_pose = True
        for i in range(self.num_uavs):
            self._active_uav_idx = i
            if self.get_distance_to_goal_3d() >= self.accept_radius:
                in_desired_pose = False
                break
        self._active_uav_idx = None

        return in_desired_pose

    def is_crashed(self):
        is_crashed = False
        model_indices = list(range(self.num_uavs))
        if self._active_uav_idx is not None:
            model_indices = [self._active_uav_idx]

        for i in model_indices:
            dynamic_model = self.dynamic_models[i]
            try:
                collision_info = dynamic_model.client.simGetCollisionInfo(vehicle_name=getattr(dynamic_model, 'vehicle_name', ''))
            except TypeError:
                collision_info = dynamic_model.client.simGetCollisionInfo()
            min_distance = self.min_distance_to_obstacles if self.num_uavs == 1 else self.min_distance_to_obstacles_all[i]
            penetration_depth = getattr(collision_info, 'penetration_depth', 0.0)
            collision_hit = collision_info.has_collided or penetration_depth > 1e-3
            if self.task_type == 'pursuit_2v1' and self.num_uavs > 1:
                # In pursuit, other UAVs can appear in depth view and trigger false crash by depth threshold.
                # Keep collision/wall impact as crash.
                crashed_now = collision_hit
            else:
                crashed_now = collision_hit or min_distance < self.crash_distance
            if crashed_now:
                is_crashed = True
                break

        return is_crashed

# ! ----------- useful functions-------------------------------------------
    def get_distance_to_goal_3d(self):
        if self._active_uav_idx is None:
            current_pose = self.dynamic_model.get_position()
            goal_pose = self.dynamic_model.goal_position
        else:
            dynamic_model = self.dynamic_models[self._active_uav_idx]
            current_pose = dynamic_model.get_position()
            goal_pose = dynamic_model.goal_position
        relative_pose_x = current_pose[0] - goal_pose[0]
        relative_pose_y = current_pose[1] - goal_pose[1]
        relative_pose_z = current_pose[2] - goal_pose[2]

        return math.sqrt(pow(relative_pose_x, 2) + pow(relative_pose_y, 2) + pow(relative_pose_z, 2))

    def getDis(self, pointX, pointY, lineX1, lineY1, lineX2, lineY2):
        '''
        Get distance between Point and Line
        Used to calculate position punishment
        '''
        a = lineY2-lineY1
        b = lineX1-lineX2
        c = lineX2*lineY1-lineX1*lineY2
        dis = (math.fabs(a*pointX+b*pointY+c))/(math.pow(a*a+b*b, 0.5))

        return dis
# ! -----------used for plot or show states------------------------------------------------------------------

    def print_episode_summary(self, info):
        steps = max(1, self.step_num + 1)
        avg_uav_rewards = (self.episode_uav_rewards / steps).tolist() if hasattr(self, 'episode_uav_rewards') else []
        summary = {
            'episode': self.episode_num,
            'steps': self.step_num,
            'total_step': self.total_step,
            'avg_uav_rewards': avg_uav_rewards,
            'episode_reward': float(self.cumulated_episode_reward),
            'is_success': info.get('is_success'),
            'is_crash': info.get('is_crash'),
            'is_not_in_workspace': info.get('is_not_in_workspace')
        }
        avg_rewards = summary['avg_uav_rewards']
        reward_msg = ' '.join([f"UAV{i+1}_ep_avg_reward={avg_rewards[i]:.4f}" for i in range(len(avg_rewards))])
        print(f"EP {summary['episode']} done | steps={summary['steps']} total_reward={summary['episode_reward']:.4f} "
              f"success={summary['is_success']} crash={summary['is_crash']} out={summary['is_not_in_workspace']} | {reward_msg}")

    def print_train_info_airsim(self, action, obs, reward, info):
        # if self.perception_type == 'split' or self.perception_type == 'lgmd':
        #     feature_all = self.feature_all
        # elif self.perception_type == 'vector':
        #     feature_all = self.feature_all
        # else:
        #     if self.cfg.get('options', 'algo') == 'TD3' or self.cfg.get('options', 'algo') == 'SAC':
        #         feature_all = self.model.actor.features_extractor.feature_all
        #     elif self.cfg.get('options', 'algo') == 'PPO':
        #         feature_all = self.model.policy.features_extractor.feature_all

        # self.client.simPrintLogMessage('feature_all: ', str(feature_all))

        msg_train_info = "EP: {} Step: {} Total_step: {}".format(
            self.episode_num, self.step_num, self.total_step)

        self.client.simPrintLogMessage('Train: ', msg_train_info)
        self.client.simPrintLogMessage('Action: ', str(action))
        self.client.simPrintLogMessage('reward: ', "{:4.4f} total: {:4.4f}".format(
            reward, self.cumulated_episode_reward))
        self.client.simPrintLogMessage('Info: ', str(info))
        if self.num_uavs > 1:
            action_position_map = info.get('uav_action_position_map', None)
            if action_position_map is not None:
                self.client.simPrintLogMessage('UAV Action-Pos: ', str(action_position_map))
        self.client.simPrintLogMessage(
            'Feature_norm: ', str(self.dynamic_model.state_norm))
        self.client.simPrintLogMessage(
            'Feature_raw: ', str(self.dynamic_model.state_raw))
        self.client.simPrintLogMessage(
            'Min_depth: ', str(self.min_distance_to_obstacles if self.num_uavs == 1 else self.min_distance_to_obstacles_all))

    def set_pyqt_signal_fixedwing(self, action, reward, done):
        """
        emit signals for pyqt plot
        """
        step = int(self.total_step)
        # action: v_xy, v_z, roll

        action_plot = np.array([10, 0, math.degrees(action[0])])

        state = self.dynamic_model.state_raw  # distance, relative yaw, roll

        # state out 6: d_xy, d_z, yaw_error, v_xy, v_z, roll
        # state in  3: d_xy, yaw_error, roll
        state_output = np.array([state[0], 0, state[1], 10, 0, state[2]])

        self.action_signal.emit(step, action_plot)
        self.state_signal.emit(step, state_output)

        # other values
        self.attitude_signal.emit(step, np.asarray(self.dynamic_model.get_attitude(
        )), np.asarray(self.dynamic_model.get_attitude_cmd()))
        self.reward_signal.emit(step, reward, self.cumulated_episode_reward)
        self.pose_signal.emit(np.asarray(self.dynamic_model.goal_position), np.asarray(
            self.dynamic_model.start_position), np.asarray(self.dynamic_model.get_position()), np.asarray(self.trajectory_list))

        # lgmd_signal = pyqtSignal(float, float, np.ndarray)  min_dist, lgmd_out, lgmd_split
        self.lgmd_signal.emit(self.min_distance_to_obstacles, 0,  self.feature_all[:-1])

    def set_pyqt_signal_multirotor(self, action, reward):
        step = int(self.total_step)
        action = np.asarray(action)

        if self.num_uavs > 1:
            action, action_split = self._split_multi_uav_action(action)
            action_output_list = []
            state_output_list = []
            attitude_real_list = []
            attitude_cmd_list = []
            for i, model in enumerate(self.dynamic_models):
                action_i = action_split[i]
                state_i = model.state_raw
                if model.navigation_3d:
                    action_output_i = action_i
                    state_output_i = state_i
                else:
                    action_output_i = np.array([action_i[0], 0, action_i[1]])
                    state_output_i = np.array([state_i[0], 0, state_i[2], state_i[3], 0, state_i[5]])
                action_output_list.append(action_output_i)
                state_output_list.append(state_output_i)
                attitude_real_list.append(model.get_attitude())
                attitude_cmd_list.append(model.get_attitude_cmd())

            action_output = np.asarray(action_output_list)
            state_output = np.asarray(state_output_list)
            attitude_real = np.asarray(attitude_real_list)
            attitude_cmd = np.asarray(attitude_cmd_list)
        else:
            dynamic_model_plot = self.dynamic_models[0]
            # transfer 2D state and action to 3D
            state = dynamic_model_plot.state_raw
            if dynamic_model_plot.navigation_3d:
                action_output = action
                state_output = state
            else:
                action_output = np.array([action[0], 0, action[1]])
                state_output = np.array([state[0], 0, state[2], state[3], 0, state[5]])
            attitude_real = np.asarray(dynamic_model_plot.get_attitude())
            attitude_cmd = np.asarray(dynamic_model_plot.get_attitude_cmd())

        self.action_signal.emit(step, action_output)
        self.state_signal.emit(step, state_output)

        # other values
        self.attitude_signal.emit(step, attitude_real, attitude_cmd)
        self.reward_signal.emit(step, reward, self.cumulated_episode_reward)

        if self.num_uavs > 1:
            traj_plot = np.asarray(self.trajectory_list, dtype=np.float32)
            current_pose = np.asarray([model.get_position() for model in self.dynamic_models], dtype=np.float32)
            goal_pose = np.asarray([model.goal_position for model in self.dynamic_models], dtype=np.float32)
            start_pose = np.asarray([model.start_position for model in self.dynamic_models], dtype=np.float32)
        else:
            traj_plot = np.asarray(self.trajectory_list)
            current_pose = np.asarray(self.dynamic_model.get_position())
            goal_pose = np.asarray(dynamic_model_plot.goal_position)
            start_pose = np.asarray(dynamic_model_plot.start_position)

        self.pose_signal.emit(goal_pose, start_pose, current_pose, traj_plot)

    def visual_log_q_value(self, q_value, action, reward):
        '''
        Create grid map (map_size = work_space)
        Log Q value and the best action in grid map
        At any grid position, record:
        1. Q value
        2. action 0
        3. action 1
        4. steps
        5. reward
        Save image every 10k steps
        Used only for 2D explanation
        '''

        # create init array if not exist
        map_size_x = self.work_space_x[1] - self.work_space_x[0]
        map_size_y = self.work_space_y[1] - self.work_space_y[0]
        if not hasattr(self, 'q_value_map'):
            self.q_value_map = np.full((9, map_size_x+1, map_size_y+1), np.nan)

        # record info
        position = self.dynamic_model.get_position()
        pose_x = position[0]
        pose_y = position[1]

        index_x = int(np.round(pose_x) + self.work_space_x[1])
        index_y = int(np.round(pose_y) + self.work_space_y[1])

        # check if index valid
        if index_x in range(0, map_size_x) and index_y in range(0, map_size_y):
            self.q_value_map[0, index_x, index_y] = q_value
            self.q_value_map[1, index_x, index_y] = action[0]
            self.q_value_map[2, index_x, index_y] = action[-1]
            self.q_value_map[3, index_x, index_y] = self.total_step
            self.q_value_map[4, index_x, index_y] = reward
            self.q_value_map[5, index_x, index_y] = q_value
            self.q_value_map[6, index_x, index_y] = action[0]
            self.q_value_map[7, index_x, index_y] = action[-1]
            self.q_value_map[8, index_x, index_y] = reward
        else:
            print(
                'Error: X:{} and Y:{} is outside of range 0~mapsize (visual_log_q_value)')

        # save array every record_step steps
        record_step = self.cfg.getint('options', 'q_map_save_steps')
        if (self.total_step+1) % record_step == 0:
            if self.data_path is not None:
                np.save(
                    self.data_path + '/q_value_map_{}'.format(self.total_step+1), self.q_value_map)
                # refresh 5 6 7 8 to record period data
                self.q_value_map[5, :, :] = np.nan
                self.q_value_map[6, :, :] = np.nan
                self.q_value_map[7, :, :] = np.nan
                self.q_value_map[8, :, :] = np.nan
