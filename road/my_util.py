from highway_env.road.road import RoadNetwork
from highway_env.road.lane import AbstractLane, StraightLane, CircularLane, LineType
from highway_env.traffic_light.traffic_light import TrafficLightNetwork, TrafficLightRule, TrafficLightState
import numpy as np


class MyUtil:

    @staticmethod
    def make_intersection(
            intersection_name: str,
            road_net: RoadNetwork,
            traffic_net: TrafficLightNetwork,
            offset: tuple[float, float] = (0,0),
            access_length: float = 100,
            dir1_rule_config: list[tuple[TrafficLightState, int]] = 
                [   
                    (TrafficLightState.GREEN, 15), 
                    (TrafficLightState.YELLOW, 10), 
                    (TrafficLightState.RED, 25)
                ],
            dir2_rule_config: list[tuple[TrafficLightState, int]] = 
                [
                    (TrafficLightState.RED, 25), 
                    (TrafficLightState.GREEN, 15), 
                    (TrafficLightState.YELLOW, 10)
                ],
        ) -> None:
        lane_width = AbstractLane.DEFAULT_WIDTH

        left_turn_radius = lane_width + 5  # [m}
        right_turn_radius = left_turn_radius + lane_width  # [m}
        outer_distance = left_turn_radius + lane_width * 1/2

        traffic_net.add_intersection(intersection_name)

        dir1_targets = []
        dir2_targets = []

        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):

            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            # Incoming
            start = np.array(offset) + rotation @ np.array(
                [-lane_width / 2, access_length + outer_distance]
            )
            end = np.array(offset) + rotation @ np.array(
                [-lane_width / 2, outer_distance]
            )
            incoming_lane = (
                intersection_name + "_" + "oi" + str(corner),
                intersection_name + "_" + "ii" + str(corner),
                StraightLane(
                    start, end, line_types=[c, s], priority=priority, speed_limit=10
                ),
            )
            road_net.add_lane(*incoming_lane)

            # Right turn
            r_center = np.array(offset) + rotation @ (
                np.array([
                    right_turn_radius - lane_width / 2, 
                    right_turn_radius - lane_width / 2,
                ])
            )
            right_turn_lane = (
                intersection_name + "_" + "ii" + str(corner),
                intersection_name + "_" + "io" + str((corner - 1) % 4),
                CircularLane(
                    r_center,
                    right_turn_radius,
                    angle + np.radians(180),
                    angle + np.radians(270),
                    line_types=[n, n],
                    priority=priority,
                    speed_limit=10,
                ),
            )
            road_net.add_lane(*right_turn_lane)
            # Left turn
            l_center = np.array(offset) + rotation @ (
                np.array(
                    [
                        -outer_distance,
                        outer_distance,
                    ]
                )
            )
            left_turn_lane = (
                intersection_name + "_" + "ii" + str(corner),
                intersection_name + "_" + "io" + str((corner + 1) % 4),
                CircularLane(
                    l_center,
                    left_turn_radius,
                    angle + np.radians(0),
                    angle + np.radians(-90),
                    clockwise=False,
                    line_types=[c, n],
                    priority=priority - 1,
                    speed_limit=10,
                ),
            )
            road_net.add_lane(*left_turn_lane)
            # Straight
            start = np.array(offset) + rotation @ np.array([-lane_width / 2, outer_distance])
            end = np.array(offset) + rotation @ np.array([-lane_width / 2, -outer_distance])
            straight_lane = (
                intersection_name + "_" + "ii" + str(corner),
                intersection_name + "_" + "io" + str((corner + 2) % 4),
                StraightLane(
                    start, end, line_types=[n, n], priority=priority, speed_limit=10
                ),
            )
            road_net.add_lane(*straight_lane)
            # Exit
            
            start = np.array(offset) + rotation @ np.flip(
                [-lane_width / 2, outer_distance], axis=0
            )
            end = np.array(offset) + rotation @ np.flip(
                [-lane_width / 2, access_length + outer_distance], axis=0
            )
            exiting_lane = (
                intersection_name + "_" + "io" + str((corner - 1) % 4),
                intersection_name + "_" + "oo" + str((corner - 1) % 4),
                StraightLane(
                    start, end, line_types=[c, n], priority=priority, speed_limit=10
                ),
            )
            road_net.add_lane(*exiting_lane)

            if corner % 2 == 0:
                dir1_targets += [straight_lane, right_turn_lane, left_turn_lane]
            else:
                dir2_targets += [straight_lane, right_turn_lane, left_turn_lane]
    
        dir1_rule = TrafficLightRule(
            config=dir1_rule_config,
            targets=dir1_targets,
        )
        dir2_rule = TrafficLightRule(
            config=dir2_rule_config,
            targets=dir2_targets,
        )

        traffic_net.add_rule(intersection_name, "dir1_rule", dir1_rule)
        traffic_net.add_rule(intersection_name, "dir2_rule", dir2_rule)


    @staticmethod
    def make_intersection_v2(
            intersection_name: str,
            road_net: RoadNetwork,
            traffic_net: TrafficLightNetwork,
            offset: tuple[float, float] = (0,0),
            access_length: float = 100,
            dir1_rule_config: list[tuple[TrafficLightState, int]] = 
                [   
                    (TrafficLightState.GREEN, 15), 
                    (TrafficLightState.YELLOW, 10), 
                    (TrafficLightState.RED, 25)
                ],
            dir2_rule_config: list[tuple[TrafficLightState, int]] = 
                [
                    (TrafficLightState.RED, 25), 
                    (TrafficLightState.GREEN, 15), 
                    (TrafficLightState.YELLOW, 10)
                ],
            num_lanes: int = 1,
        ) -> None:
        lane_width = AbstractLane.DEFAULT_WIDTH
        left_turn_radius = lane_width + 5  # [m}
        right_turn_radius = left_turn_radius + num_lanes * lane_width  # [m}
        outer_distance = left_turn_radius + lane_width * (num_lanes - 1/2)

        traffic_net.add_intersection(intersection_name)

        dir1_targets = []
        dir2_targets = []

        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):

            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )

            # Incoming
            for lane in range(num_lanes):
                start = np.array(offset) + rotation @ np.array(
                    [-lane_width / 2 - lane * lane_width, access_length + outer_distance]
                )
                end = np.array(offset) + rotation @ np.array(
                    [-lane_width / 2 - lane * lane_width, outer_distance]
                )
                if num_lanes == 1:
                    line_types = [c, c]
                elif lane == 0:
                    line_types = [s, c]
                elif lane == num_lanes - 1:
                    line_types = [c, s]
                else:
                    line_types = [s, s]
                incoming_lane = (
                    intersection_name + "_" + "oi" + str(corner) + "_" + str(lane),
                    intersection_name + "_" + "ii" + str(corner) + "_" + str(lane),
                    StraightLane(
                        start, end, line_types=line_types, priority=priority, speed_limit=10
                    ),
                )
                road_net.add_lane(*incoming_lane)


            # Right turn
            r_center = np.array(offset) + rotation @ (
                np.array([
                    right_turn_radius - lane_width / 2, 
                    right_turn_radius - lane_width / 2,
                ])
            )
            right_turn_lane = (
                intersection_name + "_" + "ii" + str(corner) + "_0",
                intersection_name + "_" + "io" + str((corner - 1) % 4 ) + "_0",
                CircularLane(
                    r_center,
                    right_turn_radius,
                    angle + np.radians(180),
                    angle + np.radians(270),
                    line_types=[n, n],
                    priority=priority,
                    speed_limit=10,
                ),
            )
            road_net.add_lane(*right_turn_lane)

            # Left turn
            l_center = np.array(offset) + rotation @ (
                np.array(
                    [
                        -outer_distance,
                        outer_distance,
                    ]
                )
            )
            left_turn_lane = (
                intersection_name + "_" + "ii" + str(corner) + "_" + str(num_lanes - 1),
                intersection_name + "_" + "io" + str((corner + 1) % 4) + "_" + str(num_lanes - 1),
                CircularLane(
                    l_center,
                    left_turn_radius,
                    angle + np.radians(0),
                    angle + np.radians(-90),
                    clockwise=False,
                    line_types=[c, n],
                    priority=priority - 1,
                    speed_limit=10,
                ),
            )
            road_net.add_lane(*left_turn_lane)

            # Straight
            for lane in range(num_lanes):
                straight_lane = (
                    intersection_name + "_" + "ii" + str(corner) + "_" + str(lane),
                    intersection_name + "_" + "io" + str((corner + 2) % 4) + "_" + str(lane),
                    StraightLane(
                        np.array(offset) + rotation @ np.array(
                            [-lane_width / 2 - lane * lane_width, outer_distance]
                        ),
                        np.array(offset) + rotation @ np.array(
                            [-lane_width / 2 - lane * lane_width, -outer_distance]
                        ),
                        line_types=[n,n],
                        priority=priority,
                        speed_limit=10,
                    ),
                )
                road_net.add_lane(*straight_lane)
                if corner % 2 == 0:
                    dir1_targets.append(straight_lane)
                else:
                    dir2_targets.append(straight_lane)

            # Exit
            for lane in range(num_lanes):
                if num_lanes == 1:
                    line_types = [c, c]
                elif lane == 0:
                    line_types = [s, c]
                elif lane == num_lanes - 1:
                    line_types = [c, s]
                else:
                    line_types = [s, s]
                start = np.array(offset) + rotation @ np.flip(
                    [-lane_width / 2 - lane * lane_width, outer_distance], axis=0
                )
                end = np.array(offset) + rotation @ np.flip(
                    [-lane_width / 2 - lane * lane_width, access_length + outer_distance], axis=0
                )
                exiting_lane = (
                    intersection_name + "_" + "io" + str((corner - 1) % 4) + "_" + str(lane),
                    intersection_name + "_" + "oo" + str((corner - 1) % 4) + "_" + str(lane),
                    StraightLane(
                        start, end, line_types=line_types, priority=priority, speed_limit=10
                    ),
                )
                road_net.add_lane(*exiting_lane)

            if corner % 2 == 0:
                dir1_targets += [right_turn_lane, left_turn_lane]
            else:
                dir2_targets += [right_turn_lane, left_turn_lane]
    
        dir1_rule = TrafficLightRule(
            config=dir1_rule_config,
            targets=dir1_targets,
        )
        dir2_rule = TrafficLightRule(
            config=dir2_rule_config,
            targets=dir2_targets,
        )

        traffic_net.add_rule(intersection_name, "dir1_rule", dir1_rule)
        traffic_net.add_rule(intersection_name, "dir2_rule", dir2_rule)