import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import json
import matplotlib.pyplot as plt
import random


@dataclass
class SectorPriority:
    """Simple sector prioritization system"""
    sector_name: str
    base_priority: float  # Fixed importance (e.g., airport = 0.9, park = 0.3)
    threat_activity: float = 0.0  # Recent threat scores in this sector
    last_update_time: float = 0.0
    current_priority: float = 0.0


class SimpleSectorAllocator:
    """
    Simplified sector prioritization without complex bandit algorithms
    Updates priorities based on recent threat activity
    """

    def __init__(self, sectors: List[Dict]):
        """
        Initialize with sectors and their base priorities
        Example sectors:
        [
            {"name": "Airport", "base_priority": 0.9, "assets": ["airport"]},
            {"name": "Government", "base_priority": 0.8, "assets": ["gov_house"]},
            {"name": "Heritage", "base_priority": 0.7, "assets": ["lahore_fort", "badshahi_mosque"]},
        ]
        """
        self.sectors = {}
        for sector_data in sectors:
            self.sectors[sector_data["name"]] = SectorPriority(
                sector_name=sector_data["name"],
                base_priority=sector_data["base_priority"],
                current_priority=sector_data["base_priority"]
            )

        # Track recent threats per sector
        self.threat_history = {name: [] for name in self.sectors.keys()}
        self.time_decay = 0.9  # How quickly old threats are forgotten

    def update_from_threat_assessment(self, threat_data: List[Dict]):
        """Update sector priorities based on new threat assessments"""

        # Reset activity scores
        for sector in self.sectors.values():
            sector.threat_activity = 0.0

        # Sum threat scores by sector
        for threat in threat_data:
            sector = threat.get('asset_sector', 'Other_Sector')
            if sector in self.sectors:
                self.sectors[sector].threat_activity += threat['overall_score']
                # Keep history for trend analysis
                self.threat_history[sector].append(threat['overall_score'])
                if len(self.threat_history[sector]) > 10:  # Keep last 10
                    self.threat_history[sector].pop(0)

        # Update priorities
        for sector_name, sector in self.sectors.items():
            # Calculate dynamic component based on recent threats
            if self.threat_history[sector_name]:
                avg_threat = np.mean(self.threat_history[sector_name])
                # Normalize threat to 0-0.5 range (so base priority still matters)
                threat_component = min(0.5, avg_threat / 20)
            else:
                threat_component = 0

            # Combine base priority with threat activity
            sector.current_priority = (
                    sector.base_priority * 0.6 +  # Base importance (60%)
                    threat_component * 0.4  # Recent threats (40%)
            )

            # Add small bonus for sudden threat spikes
            if len(self.threat_history[sector_name]) >= 2:
                recent_change = self.threat_history[sector_name][-1] - self.threat_history[sector_name][-2]
                if recent_change > 5:  # Sudden spike
                    sector.current_priority = min(1.0, sector.current_priority + 0.1)

    def get_priority_order(self) -> List[str]:
        """Get sectors in priority order"""
        sorted_sectors = sorted(
            self.sectors.items(),
            key=lambda x: x[1].current_priority,
            reverse=True
        )
        return [sector[0] for sector in sorted_sectors]

    def get_resources_allocation(self, total_resources: int, epsilon: float = 0.1) -> Dict[str, int]:
        """Allocate resources (sensors, interceptors) based on priority with ε-greedy exploration"""

        # ======= ε-GREEDY ENHANCEMENT =======
        # ε% of the time: Explore (allocate 1 unit to random low-priority sector)
        # (1-ε)% of the time: Exploit (use normal priority-based allocation)

        if random.random() < epsilon:  # Exploration phase (ε% chance)
            print(f"[ε-GREEDY: EXPLORATION MODE] Allocating 1 unit to lowest-priority sector")

            # Get lowest priority sector
            sorted_sectors = sorted(self.sectors.items(), key=lambda x: x[1].current_priority)
            low_priority_sector = sorted_sectors[0][0]  # Worst sector

            # Normal allocation for (total-1) resources
            allocation = self._get_basic_allocation(total_resources - 1)

            # Add 1 exploration unit to lowest priority sector
            allocation[low_priority_sector] = allocation.get(low_priority_sector, 0) + 1
            return allocation
        else:  # Exploitation phase (normal)
            return self._get_basic_allocation(total_resources)

    def _get_basic_allocation(self, total_resources: int) -> Dict[str, int]:
        """Basic proportional allocation (helper method)"""
        priorities = {name: sector.current_priority for name, sector in self.sectors.items()}
        total_priority = sum(priorities.values())

        allocation = {}
        allocated_so_far = 0

        # First pass: proportional allocation
        for sector_name, priority in priorities.items():
            resources = int((priority / total_priority) * total_resources)
            allocation[sector_name] = resources
            allocated_so_far += resources

        # Second pass: distribute remaining units to highest priorities
        remaining = total_resources - allocated_so_far
        if remaining > 0:
            priority_order = self.get_priority_order()
            for sector_name in priority_order:
                if remaining <= 0:
                    break
                allocation[sector_name] += 1
                remaining -= 1

        return allocation

    def visualize_priorities(self):
        """Simple visualization of sector priorities"""
        sectors = list(self.sectors.keys())
        priorities = [self.sectors[s].current_priority for s in sectors]
        base_priorities = [self.sectors[s].base_priority for s in sectors]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Priority bars
        x = np.arange(len(sectors))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, base_priorities, width, label='Base Priority', alpha=0.7, color='blue')
        bars2 = ax1.bar(x + width / 2, priorities, width, label='Current Priority', alpha=0.7, color='red')

        ax1.set_xlabel('Sector')
        ax1.set_ylabel('Priority Score')
        ax1.set_title('Sector Prioritization')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sectors, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.0)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        # Resource allocation
        allocation = self.get_resources_allocation(total_resources=20)
        resources = [allocation.get(s, 0) for s in sectors]

        bars = ax2.bar(sectors, resources, color='green', alpha=0.7)
        ax2.set_xlabel('Sector')
        ax2.set_ylabel('Resources Allocated')
        ax2.set_title('Resource Allocation (Total: 20 units)')
        ax2.set_xticks(range(len(sectors)))
        ax2.set_xticklabels(sectors, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bar, res in zip(bars, resources):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{res}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def export_for_pygame_visualization(self, filename='sector_allocation_3d.json'):
        """
        Export sector allocation data for 3D PyGame visualization
        Format specifically designed for your urban defense simulation
        """
        print(f"\nExporting data for PyGame 3D visualizer to {filename}...")

        # Get current allocation
        allocation = self.get_resources_allocation(total_resources=15)
        priority_order = self.get_priority_order()

        # Define 3D positions for Lahore sectors (approximate coordinates)
        # These would be replaced with actual GPS coordinates in your visualizer
        sector_positions_3d = {
            'Walled_City': {'x': -200, 'y': 100, 'z': 50, 'color': 'red'},
            'Central_Lahore': {'x': 0, 'y': 0, 'z': 100, 'color': 'blue'},
            'Gulberg': {'x': 150, 'y': -50, 'z': 80, 'color': 'green'},
            'Cantonment': {'x': -100, 'y': 150, 'z': 60, 'color': 'orange'},
            'Other_Sector': {'x': 100, 'y': 200, 'z': 40, 'color': 'purple'}
        }

        # Create comprehensive export data
        export_data = {
            'metadata': {
                'export_time': '2024-01-15T12:00:00',
                'total_resources': 15,
                'epsilon_value': 0.1,
                'simulation_phase': '3.3.2'
            },
            'sector_data': [],
            'allocation_summary': {
                'total_units_allocated': sum(allocation.values()),
                'highest_priority_sector': priority_order[0] if priority_order else 'none',
                'lowest_priority_sector': priority_order[-1] if priority_order else 'none'
            }
        }

        # Populate sector data
        for sector_name, sector_obj in self.sectors.items():
            sector_info = {
                'name': sector_name,
                'base_priority': sector_obj.base_priority,
                'current_priority': sector_obj.current_priority,
                'resources_allocated': allocation.get(sector_name, 0),
                'threat_history': self.threat_history.get(sector_name, []),
                'threat_activity': sector_obj.threat_activity,
                'visualization': {
                    'position': sector_positions_3d.get(sector_name, {'x': 0, 'y': 0, 'z': 0}),
                    'size': 10 + (allocation.get(sector_name, 0) * 5),
                    'color': sector_positions_3d.get(sector_name, {}).get('color', 'white'),
                    'label': f"{sector_name}: {allocation.get(sector_name, 0)} units"
                },
                'assets': []
            }

            # Add strategic recommendations
            if sector_obj.current_priority > 0.7:
                sector_info['recommendation'] = 'HIGH ALERT: Deploy maximum defenses'
            elif sector_obj.current_priority > 0.5:
                sector_info['recommendation'] = 'MODERATE: Maintain current coverage'
            else:
                sector_info['recommendation'] = 'LOW: Patrol only'

            export_data['sector_data'].append(sector_info)

        # Add threat assessment data if available
        export_data['threat_assessment'] = {
            'total_threats_detected': sum(len(threats) for threats in self.threat_history.values()),
            'highest_threat_sector': max(self.threat_history.items(),
                                         key=lambda x: np.mean(x[1]) if x[1] else 0,
                                         default=('none', []))[0]
        }

        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"✓ Exported {len(export_data['sector_data'])} sectors to {filename}")
        print(f"✓ Total resources allocated: {export_data['allocation_summary']['total_units_allocated']}")
        print(f"✓ Highest priority: {export_data['allocation_summary']['highest_priority_sector']}")

        return export_data


# ====== INTEGRATION WITH EXISTING SYSTEM ======

def integrate_with_threat_calculator():
    """Example of how to integrate with your existing ThreatScoreCalculator"""

    # Define Lahore sectors with base priorities
    lahore_sectors = [
        {"name": "Walled_City", "base_priority": 0.8, "assets": ["lahore_fort", "badshahi_mosque"]},
        {"name": "Central_Lahore", "base_priority": 0.9, "assets": ["gov_house", "mayo_hospital"]},
        {"name": "Gulberg", "base_priority": 0.7, "assets": ["liberty_market", "broadcast_tower"]},
        {"name": "Cantonment", "base_priority": 0.85, "assets": ["Cantt"]},
        {"name": "Other_Sector", "base_priority": 0.5,
         "assets": ["airport", "power_plant", "university", "water_plant"]},
    ]

    # Initialize allocator
    allocator = SimpleSectorAllocator(lahore_sectors)

    # Simulate threat updates (would come from your ThreatScoreCalculator)
    sample_threats = [
        {'asset_sector': 'Walled_City', 'overall_score': 8.5},
        {'asset_sector': 'Central_Lahore', 'overall_score': 6.2},
        {'asset_sector': 'Central_Lahore', 'overall_score': 7.8},
        {'asset_sector': 'Gulberg', 'overall_score': 4.1},
        {'asset_sector': 'Other_Sector', 'overall_score': 9.0},  # Airport threat
    ]

    # Update priorities
    allocator.update_from_threat_assessment(sample_threats)

    # Get results
    priority_order = allocator.get_priority_order()

    print("=" * 60)
    print("LAHORE URBAN DEFENSE - SECTOR PRIORITIZATION")
    print("=" * 60)
    print("\nSector Priority Order (After Threat Assessment):")
    print("-" * 50)
    for i, sector in enumerate(priority_order, 1):
        current = allocator.sectors[sector].current_priority
        base = allocator.sectors[sector].base_priority
        change = current - base
        change_symbol = "↑" if change > 0 else "↓" if change < 0 else "→"
        print(f"{i}. {sector:15} Priority: {current:.3f} (Base: {base:.2f}) {change_symbol}{abs(change):.3f}")

    print("\n" + "-" * 50)
    print("Resource Allocation (15 Total Units):")
    print("-" * 50)

    # Run allocation 5 times to show ε-greedy in action
    for run in range(1, 6):
        resource_allocation = allocator.get_resources_allocation(total_resources=15, epsilon=0.1)
        print(f"\nRun {run}:")
        total_allocated = 0
        for sector in priority_order:
            resources = resource_allocation.get(sector, 0)
            print(f"  {sector:15}: {resources:2d} units")
            total_allocated += resources

    print("\n" + "=" * 60)
    print("ε-GREEDY EXPLANATION:")
    print("- 90% of time: Normal priority-based allocation")
    print("- 10% of time: +1 unit to lowest-priority sector")
    print("- Prevents complete neglect of any area")
    print("- Allows discovery of emerging threats")
    print("=" * 60)

    # Visualize
    print("\nGenerating visualization...")
    allocator.visualize_priorities()

    # Export for 3D PyGame visualizer
    pygame_data = allocator.export_for_pygame_visualization('lahore_defense_3d.json')



    return allocator, pygame_data


if __name__ == "__main__":
    integrate_with_threat_calculator()