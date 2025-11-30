from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import json
from scipy.linalg import block_diag
import matplotlib.pyplot as plt


@dataclass
class TopologicalContext:
    """What type of urban area the target is currently in"""
    in_canyon: bool = False
    near_obstacle: bool = False
    in_void: bool = False
    in_radar_shadow: bool = False
    canyon_persistence: float = 0.0  # How strong/important this canyon is
    obstacle_threat: float = 0.0  # How dangerous this obstacle area is
    shadow_strength: float = 0.0  # How much radar is blocked (0-1)


@dataclass
class FilterSettings:
    """All the adjustable settings for the tracking system"""
    time_step: float = 0.1  # How often we update (seconds)
    state_size: int = 7  # [x,y,z,vx,vy,vz,canyon_affinity]
    measurement_size: int = 3  # [x,y,z] from radar
    initial_canyon_affinity: float = 0.5  # Starting guess for target behavior
    position_uncertainty: float = 1.0  # How uncertain we are about position
    velocity_uncertainty: float = 2.0  # How uncertain we are about speed
    affinity_uncertainty: float = 0.01  # How uncertain we are about behavior
    measurement_uncertainty: float = 10.0  # How much we trust radar measurements


@dataclass
class TrackingResult:
    """One complete snapshot of everything we know at a moment in time"""
    timestamp: float
    position: List[float]
    velocity: List[float]
    canyon_affinity: float  # How much target likes canyons (0-1)
    topological_context: str  # "canyon", "shadow", "open", etc.
    filter_confidence: List[float]  # How sure we are about position [x_conf, y_conf, z_conf]
    measurement_used: List[float]  # What the radar actually reported


class UrbanAwareTracker:
    """
    Main tracking system that follows targets in cities while learning their behavior
    Tracks: [x_position, y_position, z_position, x_velocity, y_velocity, z_velocity, canyon_preference]
    """

    def __init__(self, settings: FilterSettings = None, city_map_data=None):
        self.settings = settings or FilterSettings()
        self.city_map_data = city_map_data or {}  # Map of canyons and buildings
        self.current_urban_context: Optional[TopologicalContext] = None
        self.tracking_history: List[TrackingResult] = []  # Record of everything
        self.total_time: float = 0.0

        # Set up all the math for tracking
        self._setup_tracking_system()

    def _setup_tracking_system(self):
        """Create all the mathematical parts needed for tracking"""
        # What we're tracking: [x, y, z, speed_x, speed_y, speed_z, canyon_love]
        self.current_state = np.zeros(self.settings.state_size)
        self.current_state[6] = self.settings.initial_canyon_affinity  # Start with medium canyon preference

        # Movement prediction matrix - how things change over time
        self.movement_predictor = np.eye(self.settings.state_size)
        self.movement_predictor[0, 3] = self.settings.time_step  # x + vx*time
        self.movement_predictor[1, 4] = self.settings.time_step  # y + vy*time
        self.movement_predictor[2, 5] = self.settings.time_step  # z + vz*time

        # Measurement matrix - what we can actually see from sensors
        self.measurement_extractor = np.zeros((self.settings.measurement_size, self.settings.state_size))
        self.measurement_extractor[0, 0] = 1  # We see x position
        self.measurement_extractor[1, 1] = 1  # We see y position
        self.measurement_extractor[2, 2] = 1  # We see z position

        # Confidence matrix - how sure we are about each part of current_state
        self.confidence_matrix = np.eye(self.settings.state_size) * 100  # Start not very confident
        self.confidence_matrix[6, 6] = 0.1  # But somewhat confident about canyon preference

        # Uncertainty matrices
        self.base_prediction_uncertainty = self._create_base_uncertainty()
        self.base_measurement_uncertainty = np.eye(
            self.settings.measurement_size) * self.settings.measurement_uncertainty

        # Store all radar measurements we've received
        self.radar_history: List[np.ndarray] = []

    def _create_base_uncertainty(self) -> np.ndarray:
        """Create the basic uncertainty levels for predictions"""
        uncertainty_matrix = np.eye(self.settings.state_size)
        uncertainty_matrix[0:3, 0:3] *= self.settings.position_uncertainty  # Position uncertainty
        uncertainty_matrix[3:6, 3:6] *= self.settings.velocity_uncertainty  # Velocity uncertainty
        uncertainty_matrix[6, 6] *= self.settings.affinity_uncertainty  # Behavior uncertainty
        return uncertainty_matrix

    def predict_next_position(self, urban_context: TopologicalContext = None) -> np.ndarray:
        """Guess where the target will be next, considering urban terrain"""
        self.current_urban_context = urban_context
        self._update_canyon_behavior_dynamics()

        # Standard prediction: new_state = movement_predictor * current_state
        self.current_state = self.movement_predictor @ self.current_state

        # Update confidence: new_confidence = movement_predictor * confidence * movement_predictor^T + urban_aware_uncertainty
        self.confidence_matrix = self.movement_predictor @ self.confidence_matrix @ self.movement_predictor.T + self._get_urban_aware_prediction_uncertainty()

        return self.current_state.copy()

    def update_with_measurement(self, radar_measurement: np.ndarray) -> np.ndarray:
        """Correct our prediction with actual radar data"""
        self.radar_history.append(radar_measurement)

        # Adjust how much we trust radar based on urban area
        adjusted_radar_trust = self._get_urban_aware_measurement_uncertainty()

        # Kalman filter math to blend prediction with measurement
        measurement_error = radar_measurement - self.measurement_extractor @ self.current_state
        innovation_covariance = self.measurement_extractor @ self.confidence_matrix @ self.measurement_extractor.T + adjusted_radar_trust
        kalman_gain = self.confidence_matrix @ self.measurement_extractor.T @ np.linalg.inv(innovation_covariance)

        # Update state and confidence
        self.current_state += kalman_gain @ measurement_error
        self.confidence_matrix = (np.eye(
            self.settings.state_size) - kalman_gain @ self.measurement_extractor) @ self.confidence_matrix

        # Learn from target's behavior
        self._learn_canyon_preference_from_behavior(radar_measurement)

        # Save everything we know at this moment
        self._save_tracking_snapshot(radar_measurement)

        return self.current_state.copy()

    def _save_tracking_snapshot(self, radar_measurement: np.ndarray):
        """Save complete picture of what we know right now"""
        context_description = self._get_context_description()
        position_confidence = np.diag(self.confidence_matrix)[:3].tolist()  # How sure about x,y,z

        snapshot = TrackingResult(
            timestamp=self.total_time,
            position=self.current_state[:3].tolist(),
            velocity=self.current_state[3:6].tolist(),
            canyon_affinity=float(self.current_state[6]),
            topological_context=context_description,
            filter_confidence=position_confidence,
            measurement_used=radar_measurement.tolist()
        )

        self.tracking_history.append(snapshot)
        self.total_time += self.settings.time_step

    def _get_context_description(self) -> str:
        """Convert urban context to readable description"""
        if not self.current_urban_context:
            return "unknown_area"

        context = self.current_urban_context
        if context.in_canyon:
            return f"canyon_strength_{context.canyon_persistence:.2f}"
        elif context.near_obstacle:
            return f"near_obstacle_danger_{context.obstacle_threat:.2f}"
        elif context.in_void:
            return "open_void_area"
        elif context.in_radar_shadow:
            return f"radar_shadow_{context.shadow_strength:.2f}"
        else:
            return "open_clear_area"

    def _update_canyon_behavior_dynamics(self):
        """How canyon preference changes when we're not learning from data"""
        self.movement_predictor[6, 6] = 0.95  # Slowly forget canyon preference (decay toward neutral)

    def _get_urban_aware_prediction_uncertainty(self) -> np.ndarray:
        """Adjust prediction uncertainty based on urban terrain"""
        prediction_uncertainty = self.base_prediction_uncertainty.copy()

        if not self.current_urban_context:
            return prediction_uncertainty

        context = self.current_urban_context

        if context.in_canyon:
            canyon_strength = min(1.0, context.canyon_persistence / 1000)
            # In canyons: more certain about position, less certain about speed
            prediction_uncertainty[0:3, 0:3] *= (0.3 + 0.5 * (1 - canyon_strength))  # Lower position uncertainty
            prediction_uncertainty[3:6, 3:6] *= (1.0 + 0.3 * canyon_strength)  # Higher velocity uncertainty

        if context.near_obstacle:
            prediction_uncertainty *= (1.0 + 0.4 * context.obstacle_threat)  # More uncertainty near obstacles

        if context.in_void:
            prediction_uncertainty *= 1.2  # Slightly more uncertainty in open areas

        return prediction_uncertainty

    def _get_urban_aware_measurement_uncertainty(self) -> np.ndarray:
        """Adjust radar trust based on urban terrain"""
        radar_uncertainty = self.base_measurement_uncertainty.copy()

        if not self.current_urban_context:
            return radar_uncertainty

        context = self.current_urban_context

        if context.in_radar_shadow:
            radar_uncertainty *= (1.0 + 2.0 * context.shadow_strength)  # Trust radar much less in shadows

        if context.in_canyon:
            radar_uncertainty *= 1.5  # Trust radar less in canyons due to reflections

        return radar_uncertainty

    def _learn_canyon_preference_from_behavior(self, radar_measurement: np.ndarray):
        """Update canyon preference based on how target actually moves"""
        if len(self.radar_history) < 3:  # Need enough history to learn
            return

        recent_positions = np.array(self.radar_history[-5:])  # Last 5 positions
        if len(recent_positions) < 3:
            return

        straightness_score = self._calculate_movement_straightness(recent_positions)

        # Learn: if target moves straight in canyons, it likes canyons
        if self.current_urban_context and self.current_urban_context.in_canyon:
            if straightness_score < 0.7:  # Not perfectly straight - following canyon
                self.current_state[6] = min(1.0, self.current_state[6] + 0.1)  # Increase canyon preference
            else:  # Very straight - probably not using canyon
                self.current_state[6] = max(0.0, self.current_state[6] - 0.05)  # Decrease canyon preference

    def _calculate_movement_straightness(self, positions: np.ndarray) -> float:
        """Calculate how straight the target is moving (0=turning, 1=perfectly straight)"""
        if len(positions) < 3:
            return 1.0

        # Calculate movement directions between points
        movement_vectors = np.diff(positions, axis=0)
        movement_lengths = np.linalg.norm(movement_vectors, axis=1)
        normalized_directions = movement_vectors / movement_lengths[:, np.newaxis]

        if len(normalized_directions) < 2:
            return 1.0

        # Check how similar consecutive directions are
        direction_similarities = [np.dot(normalized_directions[i], normalized_directions[i + 1])
                                  for i in range(len(normalized_directions) - 1)]

        return np.mean(direction_similarities) if direction_similarities else 1.0

    def analyze_urban_terrain(self, position: np.ndarray) -> TopologicalContext:
        """Figure out what type of urban area this position is in"""
        terrain_info = TopologicalContext()

        if not self.city_map_data:
            return terrain_info

        position_2d = position[:2]  # Only need x,y for terrain analysis

        # Check if in any urban canyons
        for canyon in self.city_map_data.get('canyons', []):
            if self._is_position_near_canyon_center(position_2d, canyon.get('centerline', [])):
                terrain_info.in_canyon = True
                terrain_info.canyon_persistence = canyon.get('persistence', 0) / 1000
                break

        # Check if near any major obstacles/buildings
        for obstacle in self.city_map_data.get('obstacles', []):
            if self._is_position_near_obstacle(position_2d, obstacle):
                terrain_info.near_obstacle = True
                terrain_info.obstacle_threat = obstacle.get('threat_score', 0)
                terrain_info.in_radar_shadow = True
                terrain_info.shadow_strength = obstacle.get('concealment_value', 0)
                break

        # If not in canyon and not near obstacle, must be in open void
        if not terrain_info.in_canyon and not terrain_info.near_obstacle:
            terrain_info.in_void = True

        return terrain_info

    def _is_position_near_canyon_center(self, position_2d: np.ndarray, canyon_centerline: List) -> bool:
        """Check if position is close to a canyon's center path"""
        if len(canyon_centerline) < 2:
            return False

        centerline_points = np.array(canyon_centerline)[:, :2]
        distances_to_centerline = np.linalg.norm(centerline_points - position_2d, axis=1)
        return np.min(distances_to_centerline) < 50.0  # Within 50 meters of canyon center

    def _is_position_near_obstacle(self, position_2d: np.ndarray, obstacle: Dict) -> bool:
        """Check if position is near a significant obstacle"""
        birth_radius = np.sqrt(obstacle.get('birth', 100))
        death_radius = np.sqrt(obstacle.get('death', 400)) if obstacle.get('death') != float(
            'inf') else birth_radius * 2
        average_obstacle_radius = (birth_radius + death_radius) / 2
        obstacle_center = np.array([obstacle.get('birth', 0), obstacle.get('death', 0)])

        distance_to_obstacle = np.linalg.norm(position_2d - obstacle_center)
        return distance_to_obstacle < average_obstacle_radius * 1.5

    def export_complete_tracking_data(self, filename: str = 'urban_tracking_data.json'):
        """Save all tracking results for the next phase of processing"""
        print(f"Saving urban tracking data to {filename}...")

        results_package = {
            'summary_info': {
                'total_tracking_steps': len(self.tracking_history),
                'final_canyon_preference': float(self.current_state[6]),
                'total_tracking_time': self.total_time,
                'tracker_settings': {
                    'time_step_size': self.settings.time_step,
                    'starting_canyon_guess': self.settings.initial_canyon_affinity
                }
            },
            'complete_tracking_history': [{
                'time': result.timestamp,
                'estimated_position': result.position,
                'estimated_velocity': result.velocity,
                'canyon_preference': result.canyon_affinity,
                'urban_terrain_type': result.topological_context,
                'position_confidence': result.filter_confidence,
                'radar_measurement': result.measurement_used
            } for result in self.tracking_history],
            'performance_stats': {
                'average_canyon_preference': np.mean([r.canyon_affinity for r in self.tracking_history]),
                'different_terrains_encountered': len(set([r.topological_context for r in self.tracking_history])),
                'final_position_confidence': self.tracking_history[
                    -1].filter_confidence if self.tracking_history else []
            }
        }

        with open(filename, 'w') as f:
            json.dump(results_package, f, indent=2)

        print(f"Saved {len(self.tracking_history)} tracking steps to {filename}")
        return results_package


def plot_canyon_preference_learning(preference_history, positions, terrain_types):
    """Create an advanced 3D visualization of urban tracking behavior"""
    print("\nCreating advanced urban tracking visualization...")

    # Create a professional-looking figure with only 2 subplots
    fig = plt.figure(figsize=(16, 8))

    # 3D trajectory plot - Top Left
    ax1 = fig.add_subplot(121, projection='3d')

    # Convert positions to numpy array for easier manipulation
    pos_array = np.array(positions)

    # Color trajectory by canyon preference
    colors = plt.cm.viridis(preference_history)

    # Plot 3D trajectory with color coding
    for i in range(len(positions) - 1):
        ax1.plot([pos_array[i, 0], pos_array[i + 1, 0]],
                 [pos_array[i, 1], pos_array[i + 1, 1]],
                 [pos_array[i, 2], pos_array[i + 1, 2]],
                 color=colors[i], linewidth=3, alpha=0.8)

    # Add markers at each position
    scatter = ax1.scatter(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2],
                          c=preference_history, cmap='viridis', s=100, alpha=0.8)

    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.set_title('3D Target Trajectory\n(Color = Canyon Preference)')
    plt.colorbar(scatter, ax=ax1, label='Canyon Preference')

    # Behavioral radar chart - Top Right
    ax2 = fig.add_subplot(122, polar=True)

    # Calculate behavioral metrics
    metrics = ['Canyon\nUsage', 'Altitude\nStability', 'Speed\nConsistency',
               'Urban\nAdaptation', 'Path\nStraightness']

    # Simulate some behavioral scores based on the data
    canyon_usage = np.mean(preference_history)
    alt_stability = max(0, 1.0 - (np.std(pos_array[:, 2]) / 50))  # Normalize and ensure non-negative
    speed_consistency = 0.7  # Placeholder
    urban_adaptation = min(1.0, len(set(terrain_types)) / 3)  # Adaptation to different terrains
    path_straightness = 0.6  # Placeholder

    values = [canyon_usage, alt_stability, speed_consistency, urban_adaptation, path_straightness]
    values += values[:1]  # Close the radar chart

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    ax2.plot(angles, values, 'o-', linewidth=2, label='Target Behavior')
    ax2.fill(angles, values, alpha=0.25)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(0, 1)
    ax2.set_title('Behavioral Profile Radar', size=14, pad=20)
    ax2.grid(True)

    # Add overall title and adjust layout
    plt.suptitle('Topology-Enhanced Kalman Filter: Urban Behavior Analysis',
                 fontsize=16, fontweight='bold', y=0.95)

    plt.tight_layout()
    plt.show()


def demonstrate_urban_tracker():
    """Test the urban-aware tracker with example data"""
    print("=== Testing Urban-Aware Tracking System ===")

    # Create more realistic city map data
    urban_map = {
        'canyons': [
            {
                'centerline': [[0, 0, 0], [50, 0, 0], [100, 0, 0], [150, 0, 0], [200, 0, 0]],
                'persistence': 600
            }
        ],
        'obstacles': [
            {'birth': 120, 'death': 180, 'threat_score': 0.8, 'concealment_value': 0.7}
        ]
    }

    # Set up tracker with custom settings
    settings = FilterSettings(time_step=0.1, initial_canyon_affinity=0.3)
    tracker = UrbanAwareTracker(settings=settings, city_map_data=urban_map)

    # More detailed target path through different urban areas
    target_positions = [
        [10, 5, 50],  # Start in canyon
        [30, 3, 50],  # Moving through canyon
        [60, 8, 50],  # Still in canyon
        [90, 6, 50],  # Canyon continues
        [120, 15, 50],  # Near obstacle
        [150, 20, 50],  # In obstacle shadow
        [180, 25, 50],  # Leaving obstacle
        [220, 30, 50],  # Open area
        [250, 35, 50]  # Open area
    ]

    print("\nTracking target through urban environment:")

    # Store data for visualization
    preference_history = []
    terrain_history = []

    for step, position in enumerate(target_positions):
        # Analyze what type of urban area we're in
        terrain = tracker.analyze_urban_terrain(np.array(position))

        # Predict and update
        predicted_state = tracker.predict_next_position(urban_context=terrain)
        updated_state = tracker.update_with_measurement(np.array(position))

        # Store for plotting
        preference_history.append(updated_state[6])

        # Determine terrain type for visualization
        if terrain.in_canyon:
            terrain_type = "Canyon"
        elif terrain.near_obstacle:
            terrain_type = "Obstacle"
        elif terrain.in_void:
            terrain_type = "Void"
        else:
            terrain_type = "Open"
        terrain_history.append(terrain_type)

        print(f"Step {step + 1}: Position={position}, Canyon Preference={updated_state[6]:.3f}")
        active_terrain = [feature for feature, value in terrain.__dict__.items()
                          if value and feature not in ['canyon_persistence', 'obstacle_threat', 'shadow_strength']]
        print(f"         Urban Area: {active_terrain}")

    print(f"\nFinal canyon preference: {tracker.current_state[6]:.3f}")

    # Save results for next processing phase
    export_results = tracker.export_complete_tracking_data('urban_tracking_data.json')

    # Show learning progress - only top 2 graphs
    plot_canyon_preference_learning(preference_history, target_positions, terrain_history)

    print("Urban tracking test complete!")
    print(f"✓ Tracking data saved for next phase")
    print(f"✓x {export_results['summary_info']['total_tracking_steps']} tracking steps recorded")

    return tracker


if __name__ == "__main__":
    demonstrate_urban_tracker()