from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import Road
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.traffic_light.traffic_light import TrafficLightNetwork, TrafficLightState, TrafficLightRule


class IntersectionTrafficLightEnv(AbstractEnv):
    ACTIONS: dict[int, str] = {0: "SLOWER", 1: "IDLE", 2: "FASTER"}
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                    },
                    "absolute": True,
                    "flatten": False,
                    "observe_intentions": False,
                },
                "action": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": True,
                    # "target_speeds": [0, 4.5, 9],
                    "target_speeds": [0, 1, 2],
                },
                "duration": 30,  # [s]
                "destination": None, #"center_oo1",
                "controlled_vehicles": 1,
                "initial_vehicle_count": 10,
                "spawn_probability": 1,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "scaling": 5.5 * 1.3,
                "collision_reward": -5,
                "high_speed_reward": 1,
                "arrived_reward": 1,
                "reward_speed_range": [7.0, 9.0],
                "normalize_reward": False,
                "offroad_terminal": False,
            }
        )
        return config

    def _reward(self, action: int) -> float:
        """Aggregated reward, for cooperative agents."""
        return sum(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        ) / len(self.controlled_vehicles)

    def _rewards(self, action: int) -> dict[str, float]:
        """Multi-objective rewards, for cooperative agents."""
        agents_rewards = [
            self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles
        ]
        return {
            name: sum(agent_rewards[name] for agent_rewards in agents_rewards)
            / len(agents_rewards)
            for name in agents_rewards[0].keys()
        }

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """Per-agent reward signal."""
        rewards = self._agent_rewards(action, vehicle)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        reward = self.config["arrived_reward"] if rewards["arrived_reward"] else reward
        reward *= rewards["on_road_reward"]
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], self.config["arrived_reward"]],
                [0, 1],
            )
        return reward

    def _agent_rewards(self, action: int, vehicle: Vehicle) -> dict[str, float]:
        """Per-agent per-objective reward signal."""
        scaled_speed = utils.lmap(
            vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": vehicle.crashed,
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "arrived_reward": self.has_arrived(vehicle),
            "on_road_reward": vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        return (
            any(vehicle.crashed for vehicle in self.controlled_vehicles)
            or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles)
            or (self.config["offroad_terminal"] and not self.vehicle.on_road)
        )

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed or self.has_arrived(vehicle)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        )
        info["agents_terminated"] = tuple(
            self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles
        )
        return info

    def _reset(self) -> None:
        self._make_road_and_traffic_lights()
        self._make_vehicles(self.config["initial_vehicle_count"])

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        self._clear_vehicles()
        self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, terminated, truncated, info

    def _make_road_and_traffic_lights(self) -> None:
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """

        from highway_env.road.my_util import MyUtil

        lane_width = AbstractLane.DEFAULT_WIDTH
        left_turn_radius = lane_width + 5  # [m}
        right_turn_radius = left_turn_radius + lane_width  # [m}
        outer_distance = left_turn_radius + lane_width / 2
        access_length = 50 + 20 # [m]

        road_net = RoadNetwork()
        traffic_net = TrafficLightNetwork(self)

        intersection_length = 2 * (outer_distance + access_length)
        bridge_length = 10  # [m]

        MyUtil.make_intersection_v3(
            intersection_name="center",
            road_net=road_net,
            traffic_net=traffic_net,
            access_length=access_length,
            num_lanes=2,
        )

        # start = np.array([
        #     -intersection_length/2, 
        #     -lane_width / 2,
        # ])
        # end = np.array([
        #     -(intersection_length/2 + bridge_length), 
        #     -lane_width / 2,
        # ])
        # bridge_center_to_west = (
        #     "center" + "_" + "o1",
        #     "west" + "_" + "o3",
        #     StraightLane(
        #         start, 
        #         end, 
        #         line_types=[LineType.STRIPED, LineType.CONTINUOUS], 
        #         priority=3, 
        #         speed_limit=10
        #     ),
        # )
        # road_net.add_lane(*bridge_center_to_west)

        # start = np.array([
        #     -(intersection_length/2 + bridge_length), 
        #     lane_width / 2,
        # ])
        # end = np.array([
        #     -intersection_length/2, 
        #     lane_width / 2,
        # ])
        
        # bridge_center_to_west = (
        #     "west" + "_" + "o3",
        #     "center" + "_" + "o1",
        #     StraightLane(
        #         start, 
        #         end, 
        #         line_types=[LineType.NONE, LineType.CONTINUOUS], 
        #         priority=3, 
        #         speed_limit=10
        #     ),
        # )
        # road_net.add_lane(*bridge_center_to_west)


        # self._make_intersection(
        #     intersection_name="west",
        #     road_net=road_net,
        #     traffic_net=traffic_net,
        #     right_turn_radius=right_turn_radius,
        #     left_turn_radius=left_turn_radius,
        #     outer_distance=outer_distance,
        #     offset=(-(intersection_length+bridge_length), 0),
        #     access_length=access_length
        # )
        
        road = Road(
            network=road_net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
            traffic_light_network=traffic_net
        )
        self.road = road 


    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3
        
        # # Enable lane changes for NPCs
        # if hasattr(vehicle_type, 'LANE_CHANGE_MIN_ACC_GAIN'):
        #     vehicle_type.LANE_CHANGE_MIN_ACC_GAIN = 0.1  # Lower threshold for more aggressive lane changes
        #     vehicle_type.LANE_CHANGE_DELAY = 0.5  # More frequent lane change attempts
        #     vehicle_type.POLITENESS = 0.1  # Slightly more considerate

        # Random vehicles
        simulation_steps = 3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [
                (
                    self.road.act(),
                    self.road.step(1 / self.config["simulation_frequency"]),
                )
                for _ in range(self.config["simulation_frequency"])
            ]

        # Challenger vehicle
        self._spawn_vehicle(
            20,
            spawn_probability=1,
            go_straight=True,
            position_deviation=0.1,
            speed_deviation=0,
        )

        # Controlled vehicles
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(
                # (f"center_oi{ego_id % 4}_0", f"center_ii{ego_id % 4}_0", 0)
                (f"center_oi{ego_id % 4}", f"center_ii{ego_id % 4}", 1)

            )
            destination = self.config["destination"]
            # while destination is None or destination == f"center_oo{ego_id % 4}":
            while destination is None or  f"center_oo{ego_id % 4}" in destination:
                destination = "center_oo" + str(
                    self.np_random.integers(0, 4)
                )
            print("plan from " + f"center_oi{ego_id % 4} to {destination}")
            ego_vehicle = self.action_type.vehicle_class(
                self.road,
                ego_lane.position(20 + 5 * self.np_random.normal(1), 0),
                speed=ego_lane.speed_limit,
                heading=ego_lane.heading_at(60),
            )
            try:
                ego_vehicle.plan_route_to(destination)
                print("Planned route:", ego_vehicle.route)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(
                    ego_lane.speed_limit
                )
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(
                    ego_vehicle.speed_index
                )
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            for v in self.road.vehicles:  # Prevent early collisions
                if (
                    v is not ego_vehicle
                    and np.linalg.norm(v.position - ego_vehicle.position) < 20
                ):
                    self.road.vehicles.remove(v)

    def _spawn_vehicle(
        self,
        longitudinal: float = 0,
        position_deviation: float = 1.0,
        speed_deviation: float = 1.0,
        spawn_probability: float = 0.6,
        go_straight: bool = False,
    ) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        lane_id = self.np_random.choice(range(2))
        route = self.np_random.choice(range(4), size=2, replace=False)
        # route = np.array([0, 2])
        # route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        route[1] = (route[0] - 1) % 4
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(
            self.road,
            # ("center_oi" + str(route[0]) + "_1", "center_ii" + str(route[0]) + "_1", 0),
            ("center_oi" + str(route[0]), "center_ii" + str(route[0]), lane_id),

            longitudinal=(
                longitudinal + 5 + self.np_random.normal() * position_deviation
            ),
            speed=3 + self.np_random.normal() * speed_deviation,
        )
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        # vehicle.plan_route_to("center_oo" + str(route[1]) + "_1")
        vehicle.plan_route_to("center_oo" + str(route[1]))

        # Enable lane changes for NPC vehicles
        if hasattr(vehicle, 'enable_lane_change'):
            vehicle.enable_lane_change = True

        # if route[0] == 1:
        #     print(str(vehicle.route))

        # print(str(vehicle.route))
        
        # # Set route with no specific lane preference to allow more lane changes
        # if hasattr(vehicle, 'route') and vehicle.route:
        #     # Remove specific lane requirements from route to allow more flexibility
        #     for i, (from_node, to_node, lane_id) in enumerate(vehicle.route):
        #         vehicle.route[i] = (from_node, to_node, None)  # None allows any lane
        
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        # print("spawned vehicle on " + str(vehicle.route))
        return vehicle

    def _clear_vehicles(self) -> None:
        is_leaving = (
            lambda vehicle: "io" in vehicle.lane_index[0]
            and "oo" in vehicle.lane_index[1]
            and vehicle.lane.local_coordinates(vehicle.position)[0]
            >= vehicle.lane.length - 4 * vehicle.LENGTH
        )
        self.road.vehicles = [
            vehicle
            for vehicle in self.road.vehicles
            if vehicle in self.controlled_vehicles
            or not (is_leaving(vehicle) or vehicle.route is None)
        ]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        # print("has_arrived check:", vehicle.lane_index)
        if not vehicle.route:
            return True

        if (
        not vehicle.lane_index
        or vehicle.lane_index[0] is None
        or vehicle.lane_index[1] is None
        ):
            # print("has_arrived: missing lane index")
            return False
        # print("vehicle last path:", vehicle.route[-1])
        # print("vehicle destination:", vehicle.lane_index)
        result = (
            "io" in vehicle.lane_index[0]
            and vehicle.route[-1][1] in vehicle.lane_index[1]
            and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance
        )
        # print("has_arrived:", vehicle.lane_index, vehicle.lane.local_coordinates(vehicle.position), "->", result)
        
        return result

    