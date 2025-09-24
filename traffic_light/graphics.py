from __future__ import annotations

import pygame
import numpy as np
from typing import TYPE_CHECKING

from highway_env.traffic_light.traffic_light import TrafficLightNetwork, TrafficLightState


if TYPE_CHECKING:
    from highway_env.road.graphics import WorldSurface


class TrafficLightGraphics:
    """Graphics for rendering traffic lights."""
    
    # Colors for traffic light states
    RED_COLOR = (255, 0, 0)
    YELLOW_COLOR = (255, 255, 0)
    GREEN_COLOR = (0, 255, 0)
    OFF_COLOR = (100, 100, 100)  # Dark gray for unlit lights
    FRAME_COLOR = (50, 50, 50)   # Dark frame color
    
    LIGHT_RADIUS = 0.8  # Radius of each light in meters
    LIGHT_SPACING = 2.5  # Spacing between lights in meters
    FRAME_WIDTH = 6.0   # Width of the frame in meters
    FRAME_HEIGHT = 2.0  # Height of the frame in meters

    @classmethod 
    def display(cls):
        pass

    @classmethod
    def display_agent_obs(
        cls,
        current_state: TrafficLightState,
        surface: WorldSurface,
        position: tuple[float, float] = (0, 0),
    ) -> None:

        if current_state == TrafficLightState.NONE:
            return  # No traffic light to display
        
        # if not surface.is_visible(position):
        #     return

        # print("Rendering traffic light:", current_state)

        screen_width, screen_height = surface.get_size()
        light_pos = [screen_width // 2, screen_height // 10]

        frame_rect = (
            light_pos[0] - surface.pix(cls.FRAME_WIDTH / 2),
            light_pos[1] - surface.pix(cls.FRAME_HEIGHT / 2),
            surface.pix(cls.FRAME_WIDTH),
            surface.pix(cls.FRAME_HEIGHT)
        )
        pygame.draw.rect(surface, cls.FRAME_COLOR, frame_rect)
        pygame.draw.rect(surface, (0, 0, 0), frame_rect, 2)

        pos_delta = [
            (surface.pix(p[0]), surface.pix(p[1])) # pygame画面の座標系に変換
            for p in [
                (-cls.LIGHT_SPACING, 0), (0, 0), (cls.LIGHT_SPACING, 0) # 各信号の位置差分
            ]
        ]

        light_states = [TrafficLightState.RED, TrafficLightState.YELLOW, TrafficLightState.GREEN]
        light_colors = [cls.RED_COLOR, cls.YELLOW_COLOR, cls.GREEN_COLOR]

        for i, (delta, state, color) in enumerate(zip(pos_delta, light_states, light_colors)):
            radius = surface.pix(cls.LIGHT_RADIUS)
            pos = (light_pos[0]+delta[0], light_pos[1]+delta[1])

            # Determine if this light should be lit
            if current_state == state:
                # Light is on - use full color
                pygame.draw.circle(surface, color, pos, radius)
            else:
                # Light is off - use dim color
                pygame.draw.circle(surface, cls.OFF_COLOR, pos, radius)

            # Draw border around each light
            pygame.draw.circle(surface, (0, 0, 0), pos, radius, 2)
