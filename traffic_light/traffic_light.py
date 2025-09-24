from highway_env.road.lane import AbstractLane
from enum import Enum

import numpy as np


class TrafficLightState(Enum):
    RED = 1
    YELLOW = 2
    GREEN = 3
    NONE = 4


class TrafficLightRule:
    config: list[tuple[TrafficLightState, int]]    # (状態, 継続時間)
    targets: list[tuple[str, str, AbstractLane]]  # 対象レーン

    def __init__(self, config: dict, targets: list[tuple[str, str, AbstractLane]]) -> None:
        self.config = config
        self.targets = targets


class TrafficLightNetwork:
    # 交差点名: {ルール名: ルール, ...}
    graph: dict[str, dict[str, TrafficLightRule]] 

    DIST_PASSED  = -1.0
    DIST_UNKNOWN = -1000

    def __init__(self, env) -> None:
        self.graph = {}
        self.env = env

    def add_intersection(self, intersection_name: str) -> None:
        self.graph[intersection_name] = {}

    def add_rule(self, intersection_name: str, rule_name: str, rule: TrafficLightRule) -> None:
        self.graph[intersection_name][rule_name] = rule

    def get_current_state(
        self,
        intersection_name: str,
        rule_name: str,
        time: float
    ) -> TrafficLightState:
        
        # print(f"get_current_state: intersection_name={intersection_name}, rule_name={rule_name}")

        if intersection_name not in self.graph:
            return TrafficLightState.NONE

        if rule_name not in self.graph[intersection_name]:
            return TrafficLightState.NONE

        rule = self.graph[intersection_name][rule_name]

        total_cycle_time = sum(duration for _, duration in rule.config)
        if total_cycle_time == 0:
            return TrafficLightState.NONE
            
        cycle_time = time % total_cycle_time
        time_in_cycle = 0
        
        for state, duration in rule.config:
            time_in_cycle += duration
            if cycle_time < time_in_cycle:
                return state
                
        return TrafficLightState.NONE

    def observe_traffic_light(self, vehicle) -> tuple[TrafficLightState, float | None]:
        route = vehicle.route
        intersection_name = None
        rule_name = None
        distance = np.float64(self.DIST_UNKNOWN)
        state = TrafficLightState.NONE
        if not route:
            return state, distance
        
        

        intersection_name, rule_name, dist = self.get_traffic_names_and_distance_by_lane(vehicle, route[0])

        # 信号通過済みなら距離を-1にする
        if intersection_name is not None:
            lane_length = vehicle.road.network.get_lane(route[0]).length

            distance = - np.float64(lane_length - dist)
            state = self.get_current_state(
                intersection_name, 
                rule_name, 
                self.env.time
            )
            # print("state:", state, "distance:", distance)
            return state, distance


        if len(route) == 1:
            return state, distance

        intersection_name, rule_name, dist = self.get_traffic_names_and_distance_by_lane(vehicle, route[1])

        state = self.get_current_state(intersection_name, rule_name, self.env.time)
        # print("state:", state, "distance:", dist)
        return state, dist

    def get_traffic_names_and_distance_by_lane(self, vehicle, path) -> tuple[str | None, str | None, float | None]:
        _from, _to, _ = path
        # print(f"_from: {_from}, _to: {_to}")
        for i_name, i_rules in self.graph.items():
            for r_name, rule in i_rules.items():
                for f, t, l in rule.targets:
                    # print(f"f: {f}, t: {t}")
                    if (f, t) != (_from, _to):
                        continue
                    dist = self.get_distance_to_endpoint(vehicle, l)
                    return i_name, r_name, dist
        return None, None, None

    def get_distance_to_endpoint(self, vehicle, lane) -> float | None:
        if lane is None:
            return None
        longitudinal, _ = lane.local_coordinates(vehicle.position)
        vehicle_length = vehicle.LENGTH
        return vehicle_length - longitudinal - vehicle_length / 2

        

    