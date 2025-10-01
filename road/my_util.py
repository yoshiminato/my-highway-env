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

    @staticmethod
    def make_intersection_v3(
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
            start = np.array([-lane_width / 2, access_length + outer_distance])
            end = np.array([-lane_width / 2, outer_distance])

            _from = intersection_name + "_" + "oi" + str(corner)
            _to   = intersection_name + "_" + "ii" + str(corner)

            # print(f"=======incoming lanes: from {_from} to {_to}")

            incoming_lanes = MyUtil.make_multi_straight_lane(
                num_lanes,
                start,
                end,
                offset=np.array(offset),
                angle=angle,
            )
            for lane in incoming_lanes:
                road_net.add_lane(_from, _to, lane)

            # Right turn
            r_center = np.array(offset) + rotation @ (
                np.array([
                    right_turn_radius - lane_width / 2, 
                    right_turn_radius - lane_width / 2,
                ])
            )
            _from = intersection_name + "_" + "ii" + str(corner)
            _to   = intersection_name + "_" + "io" + str((corner - 1) % 4)

            # print(f"=======right turn lane: from {_from} to {_to}")
            # print(f"center: {r_center}")
            

            lane = CircularLane(
                    r_center,
                    right_turn_radius,
                    angle + np.radians(180),
                    angle + np.radians(270),
                    line_types=[n, n],
                    priority=priority,
                    speed_limit=10,
                )
            
            # print(f"start: {lane.position(0,0)}, end: {lane.position(lane.length,0)}")

            right_turn_lane = (_from, _to, lane)
                
            road_net.add_lane(*right_turn_lane)
            right_lane_index = (_from, _to, None)
            assignTrafficRuleInIntersection(dir1_targets, dir2_targets, right_lane_index, corner)
            

            # Left turn
            l_center = np.array(offset) + rotation @ (
                np.array(
                    [
                        -outer_distance,
                        outer_distance,
                    ]
                )
            )
            _from = intersection_name + "_" + "ii" + str(corner)
            _to   = intersection_name + "_" + "io" + str((corner + 1) % 4)

            # print(f"=======left turn lane: from {_from} to {_to}")

            lane = CircularLane(
                    l_center,
                    left_turn_radius,
                    angle + np.radians(0),
                    angle + np.radians(-90),
                    clockwise=False,
                    line_types=[c, n],
                    priority=priority - 1,
                    speed_limit=10,
                )

            # print(f"start: {lane.position(0,0)}, end: {lane.position(lane.length,0)}")

            left_turn_lane = (_from, _to, lane)
            # if (corner + 1) % 4 == 1:
            #     start = left_turn_lane[2].position(0, 0)
            #     print("io1 end of left turn lane: " + str(start))
            road_net.add_lane(*left_turn_lane)
            left_lane_index = (_from, _to, None)
            assignTrafficRuleInIntersection(dir1_targets, dir2_targets, left_lane_index, corner)
            
            # Straight
            start = np.array([-lane_width / 2, outer_distance])
            end = np.array([-lane_width / 2, -outer_distance])

            _from = intersection_name + "_" + "ii" + str(corner)
            _to   = intersection_name + "_" + "io" + str((corner + 2) % 4)

            # print(f"=======straight lanes: from {_from} to {_to}")

            straight_lanes = MyUtil.make_multi_straight_lane(
                num_lanes,
                start,
                end,
                offset=np.array(offset),
                angle=angle,
                no_line=True,
            )

            for id, lane in enumerate(straight_lanes):    
                road_net.add_lane(_from, _to, lane)
                straight_lane_index = (_from, _to, id)
                assignTrafficRuleInIntersection(dir1_targets, dir2_targets, straight_lane_index, corner)


            # Exit
            start = np.array([-lane_width / 2, -outer_distance])
            end   = np.array([-lane_width / 2, -access_length - outer_distance]) 

            # if (corner + 2) % 4 == 1:
            #     print("======= target exit lane =======")

            _from = intersection_name + "_" + "io" + str((corner + 2) % 4)
            _to   = intersection_name + "_" + "oo" + str((corner + 2) % 4)

            # print(f"=======exiting lanes: from {_from} to {_to}")

            exiting_lanes = MyUtil.make_multi_straight_lane(
                num_lanes,
                start,
                end,
                offset=np.array(offset),
                angle=angle,
            )

            # if (corner + 2) % 4 == 1:
            #     print("======= fin =======")

            # if (corner + 2) % 4 == 1:
            #     print("io1 start of exit lane: " + str(start))


            for lane in exiting_lanes:
                road_net.add_lane(
                    _from,
                    _to,
                    lane
                )

    
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
    def make_multi_straight_lane(
        lane_count: int,
        start: np.ndarray,
        end: np.ndarray,
        offset: np.ndarray = np.array([0,0]),
        angle: float = 0,
        no_line: bool = False,
    ) -> list[StraightLane]:
        """
        複数直線車線を作成
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        lanes = []

        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

        for lane in range(lane_count):
            lane_offset = np.array([- lane * lane_width, 0])
            lane_start = offset + rotation @ (start + lane_offset)
            lane_end = offset + rotation @ (end + lane_offset)
            line_types = [
                LineType.CONTINUOUS_LINE if lane == lane_count - 1 else LineType.NONE,
                LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
            ] if not no_line else [LineType.NONE, LineType.NONE]
            # print(f"start: {lane_start}, end: {lane_end}")
            lanes.append(StraightLane(lane_start, lane_end, line_types=line_types))
        return lanes


def assignTrafficRuleInIntersection(arr1, arr2, target, corner) -> None:
    """
    交差点において通行方向ごと(水平方向or垂直方向)に配列に振り分ける
    arr1: 水平方向のレーンを格納する配列
    arr2: 垂直方向のレーンを格納する配列
    target: 振り分け対象のレーンインデックス
    corner: 交差点の角(0~3)
    """
    if corner % 2 == 0:
        arr1 += [target]
    else:
        arr2 += [target]
    return
