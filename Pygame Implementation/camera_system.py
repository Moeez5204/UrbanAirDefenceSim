# camera_system.py - Camera and control system (Updated for full 360° rotation)

import pygame
import math


class CameraSystem:
    """Handles camera movement and controls for larger city"""

    def __init__(self):
        # Camera position and orientation
        self.x = 0
        self.y = 0
        self.z = 0
        self.yaw = 0
        self.pitch = 0

        # Camera settings
        self.height = 20
        self.speed = 8.0  # Increased for larger city
        self.mouse_sensitivity = 0.002

        # Flight mode
        self.flight_mode = True

        # Screen center for mouse control
        self.screen_center = (0, 0)

        # Fixed camera positions (base positions - can't move from these)
        self.fixed_cameras = [
            # Camera 1: City Center View
            {
                'name': 'Center View',
                'x': 1500, 'y': 1500, 'z': 300,
                'base_yaw': 0, 'base_pitch': -0.5
            },
            # Camera 2: Walled City View
            {
                'name': 'Walled City',
                'x': 900, 'y': 1200, 'z': 200,
                'base_yaw': 0.8, 'base_pitch': -0.4
            },
            # Camera 3: Gulberg View
            {
                'name': 'Gulberg',
                'x': 2100, 'y': 1650, 'z': 250,
                'base_yaw': 2.5, 'base_pitch': -0.3
            },
            # Camera 4: Cantonment View
            {
                'name': 'Cantonment',
                'x': 1200, 'y': 2100, 'z': 180,
                'base_yaw': 1.2, 'base_pitch': -0.35
            },
            # Camera 5: High Altitude View
            {
                'name': 'High Altitude',
                'x': 1500, 'y': 1500, 'z': 600,
                'base_yaw': 0, 'base_pitch': -1.0
            },
            # Camera 6: Edge View
            {
                'name': 'Edge View',
                'x': 2400, 'y': 2400, 'z': 350,
                'base_yaw': 3.9, 'base_pitch': -0.25
            }
        ]

        self.current_fixed_camera = None
        self.fixed_camera_rotation = {'yaw_offset': 0, 'pitch_offset': 0}
        self.was_fixed_camera = False

    def setup(self, city_center, screen_width=1600, screen_height=900):
        """Setup camera initial position for larger city"""
        self.x = city_center[0]
        self.y = city_center[1]
        self.z = 80  # Start higher for better view
        self.yaw = 0
        self.pitch = -0.3

        self.screen_center = (screen_width // 2, screen_height // 2)

        # Setup mouse control
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        pygame.mouse.set_pos(self.screen_center)

    def update(self):
        """Update camera based on input"""
        self.handle_mouse_look()

        # Only handle keyboard movement if not in fixed camera mode
        if self.current_fixed_camera is None:
            self.handle_keyboard_movement()
        else:
            # Update to fixed camera position (CAN'T MOVE - only position is fixed)
            cam = self.fixed_cameras[self.current_fixed_camera]
            self.x = cam['x']
            self.y = cam['y']
            self.z = cam['z']

            # Apply rotation offsets to base orientation (FULL 360° ROTATION)
            self.yaw = cam['base_yaw'] + self.fixed_camera_rotation['yaw_offset']
            self.pitch = cam['base_pitch'] + self.fixed_camera_rotation['pitch_offset']

            # Keep yaw in 0-2π range for cleaner math
            self.yaw = self.yaw % (2 * math.pi)

            # FULL 360° PITCH CONTROL - allow looking straight up and down
            # Pitch range: -π to π (-180° to 180°)
            self.pitch = self.pitch % (2 * math.pi)
            if self.pitch > math.pi:
                self.pitch = self.pitch - (2 * math.pi)

            self.was_fixed_camera = True

    def handle_mouse_look(self):
        """Handle mouse look input - works in both modes!"""
        mouse_pos = pygame.mouse.get_pos()
        mouse_dx = mouse_pos[0] - self.screen_center[0]
        mouse_dy = mouse_pos[1] - self.screen_center[1]

        if abs(mouse_dx) > 0 or abs(mouse_dy) > 0:
            if self.current_fixed_camera is not None:
                # In fixed camera mode, update rotation offsets
                self.fixed_camera_rotation['yaw_offset'] += mouse_dx * self.mouse_sensitivity
                self.fixed_camera_rotation['pitch_offset'] -= mouse_dy * self.mouse_sensitivity

                # NO LIMITS ON ROTATION - FULL 360° FREEDOM
                # Let yaw_offset wrap around naturally
                self.fixed_camera_rotation['yaw_offset'] = self.fixed_camera_rotation['yaw_offset'] % (2 * math.pi)

                # Keep pitch_offset in reasonable bounds but allow full rotation
                # If you want true 360° vertical rotation, remove this constraint:
                # For more natural viewing, keep pitch within ±π/2 (±90°)
                # For FULL 360° vertical: remove this constraint
                max_pitch = math.pi  # Full 180° up and down (π radians total)
                if abs(self.fixed_camera_rotation['pitch_offset']) > max_pitch:
                    self.fixed_camera_rotation['pitch_offset'] = max_pitch * math.copysign(1,
                                                                                           self.fixed_camera_rotation[
                                                                                               'pitch_offset'])

            else:
                # In free camera mode, update direct orientation
                self.yaw += mouse_dx * self.mouse_sensitivity
                self.yaw = self.yaw % (2 * math.pi)  # Keep in 0-2π range

                self.pitch -= mouse_dy * self.mouse_sensitivity
                # FULL 360° PITCH CONTROL for free camera too
                self.pitch = self.pitch % (2 * math.pi)
                if self.pitch > math.pi:
                    self.pitch = self.pitch - (2 * math.pi)

            pygame.mouse.set_pos(self.screen_center)

    def handle_keyboard_movement(self):
        """Handle keyboard movement input - only in free camera mode"""
        # Don't handle movement if in fixed camera mode
        if self.current_fixed_camera is not None:
            return

        keys = pygame.key.get_pressed()

        forward_x = math.cos(self.yaw)
        forward_y = math.sin(self.yaw)
        right_x = math.cos(self.yaw + math.pi / 2)
        right_y = math.sin(self.yaw + math.pi / 2)

        current_speed = self.speed
        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
            current_speed *= 3

        move_x, move_y, move_z = 0, 0, 0

        # W - Move forward
        if keys[pygame.K_w]:
            move_x += forward_x * current_speed
            move_y += forward_y * current_speed

        # S - Move backward
        if keys[pygame.K_s]:
            move_x -= forward_x * current_speed
            move_y -= forward_y * current_speed

        # A - Strafe left
        if keys[pygame.K_a]:
            move_x += right_x * current_speed
            move_y += right_y * current_speed

        # D - Strafe right
        if keys[pygame.K_d]:
            move_x -= right_x * current_speed
            move_y -= right_y * current_speed

        # Space - Move up
        if keys[pygame.K_SPACE]:
            move_z += current_speed

        # Shift - Move down
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            move_z -= current_speed

        # Apply movement
        self.x += move_x
        self.y += move_y

        # Apply vertical movement based on mode
        if self.flight_mode:
            self.z += move_z
            self.z = max(5, self.z)
        else:
            self.z += move_z
            self.z = max(20, min(100, self.z))

    def switch_to_fixed_camera(self, camera_index):
        """Switch to a fixed camera position"""
        if 0 <= camera_index < len(self.fixed_cameras):
            self.current_fixed_camera = camera_index
            self.flight_mode = True  # Force flight mode for fixed cameras
            self.fixed_camera_rotation = {'yaw_offset': 0, 'pitch_offset': 0}  # Reset rotation
            return True
        return False

    def switch_to_free_camera(self):
        """Switch back to free camera movement"""
        # If we were in fixed camera mode and just switched to free,
        # keep the current orientation
        if self.was_fixed_camera and self.current_fixed_camera is not None:
            # Use the last orientation from fixed camera
            cam = self.fixed_cameras[self.current_fixed_camera]
            self.x = cam['x']
            self.y = cam['y']
            self.z = cam['z']
            self.yaw = cam['base_yaw'] + self.fixed_camera_rotation['yaw_offset']
            self.pitch = cam['base_pitch'] + self.fixed_camera_rotation['pitch_offset']
            self.was_fixed_camera = False

        self.current_fixed_camera = None
        self.fixed_camera_rotation = {'yaw_offset': 0, 'pitch_offset': 0}

    def reset_fixed_camera_rotation(self):
        """Reset rotation of current fixed camera to its base orientation"""
        if self.current_fixed_camera is not None:
            self.fixed_camera_rotation = {'yaw_offset': 0, 'pitch_offset': 0}

    def is_fixed_camera_mode(self):
        """Check if in fixed camera mode"""
        return self.current_fixed_camera is not None

    def get_current_camera_name(self):
        """Get name of current fixed camera"""
        if self.current_fixed_camera is not None:
            return self.fixed_cameras[self.current_fixed_camera]['name']
        return "Free Camera"

    def get_rotation_info(self):
        """Get current rotation information"""
        yaw_degrees = math.degrees(self.yaw) % 360
        pitch_degrees = math.degrees(self.pitch)

        # Convert to compass direction
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        index = int((yaw_degrees + 22.5) / 45) % 8

        return {
            'yaw_degrees': yaw_degrees,
            'pitch_degrees': pitch_degrees,
            'direction': directions[index],
            'is_fixed': self.is_fixed_camera_mode()
        }

    def project_3d_to_2d(self, point_x, point_y, point_z, screen_width, screen_height):
        """Project 3D world point to 2D screen coordinates"""
        dx = point_x - self.x
        dy = point_y - self.y
        dz = point_z - (self.z + self.height)

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        cos_pitch = math.cos(self.pitch)
        sin_pitch = math.sin(self.pitch)

        x_rot = dx * cos_yaw + dz * sin_yaw
        z_rot = -dx * sin_yaw + dz * cos_yaw

        y_rot = dy * cos_pitch - z_rot * sin_pitch
        z_final = dy * sin_pitch + z_rot * cos_pitch

        if z_final > 0.1:
            fov = math.pi / 3

            screen_x = (x_rot / z_final) * (screen_height / math.tan(fov / 2)) + screen_width / 2
            screen_y = (-y_rot / z_final) * (screen_height / math.tan(fov / 2)) + screen_height / 2

            scale_factor = 1000 / (z_final + 1000)

            return int(screen_x), int(screen_y), scale_factor

        return None, None, 0

    def toggle_flight_mode(self):
        """Toggle between flight and walk mode"""
        # Only toggle in free camera mode
        if self.current_fixed_camera is None:
            self.flight_mode = not self.flight_mode
            if not self.flight_mode and self.z > 100:
                self.z = 50

    def reset_position(self):
        """Reset camera to city center"""
        self.x = 1500
        self.y = 1500
        self.z = 80
        self.yaw = 0
        self.pitch = -0.3
        self.current_fixed_camera = None
        self.fixed_camera_rotation = {'yaw_offset': 0, 'pitch_offset': 0}

    def adjust_speed(self, delta):
        """Adjust movement speed"""
        # Only adjust in free camera mode
        if self.current_fixed_camera is None:
            self.speed = max(2, min(20, self.speed + delta))

    def cleanup(self):
        """Cleanup camera system"""
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)