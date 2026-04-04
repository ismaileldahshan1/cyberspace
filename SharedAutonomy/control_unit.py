"""
ControlUnit
===========
Execution-stage Shared Autonomy for the CyberScape framework.

Implements the "Control Unit" described in Ghaly et al. [4] – the
arbitration layer that lets operators seamlessly switch robots between
AUTONOMOUS and MANUAL modes without interrupting overall mission
coordination.

Architecture (mirrors Fig. 2 from the shared-autonomy paper [4]):

  ┌─────────────────────────────────────────────────────┐
  │                    ControlUnit                       │
  │                                                      │
  │  [Swarm/LLM Autonomous Path]  [Human Control Path]  │
  │        ↓                              ↓              │
  │  Process autonomous            Process manual        │
  │  commands                      commands              │
  │        ↓                              ↓              │
  │            ──── Mode Selector ────                   │
  │                      ↓                               │
  │            Drive robot commands                      │
  └─────────────────────────────────────────────────────┘

Each robot has its own namespace (/<robot_name>/...) so that switching
one robot to manual mode does not affect the others.

References:
  Ghaly et al. [4] – shared autonomy in swarm robotics for S&R
  Goodrich & Schultz [16] – adjustable autonomy / cognitive load
"""

import logging
import time
from typing import Any, Dict, List, Optional

from SharedAutonomy.shared_autonomy_manager import RobotMode, get_manager

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Supported manual command types (robot-agnostic)
# ─────────────────────────────────────────────────────────

MANUAL_COMMAND_SCHEMA = {
    "move":     ["linear_x", "angular_z"],       # velocity setpoint
    "go_to":    ["x", "y"],                      # absolute waypoint
    "stop":     [],                               # zero velocity
    "hover":    [],                               # UAV-specific hover
    "land":     [],                               # UAV land
    "takeoff":  ["altitude"],                     # UAV takeoff
    "inspect":  ["target_label"],                 # stop & inspect object
}


class ControlUnit:
    """
    Per-process singleton that acts as the arbitration layer for all
    robots registered in the current mission.

    The Flask server creates one ControlUnit and calls its methods from
    the HITL endpoints.  Robot nodes (ROS2 or simulated) poll
    /hitl/get_manual_command to receive buffered operator commands.
    """

    def __init__(self):
        self._manager = get_manager()
        self._registered_robots: Dict[str, Dict[str, Any]] = {}

    # ──────────────────────────────────────────────────────
    # Registration
    # ──────────────────────────────────────────────────────

    def register_robot(
        self,
        robot_name: str,
        robot_type: str = "ground",          # "ground" | "aerial"
        capabilities: Optional[List[str]] = None,
    ) -> None:
        """
        Register a robot so the ControlUnit is aware of its type.
        Called at server start-up from the robot configuration.
        """
        self._registered_robots[robot_name] = {
            "type": robot_type,
            "capabilities": capabilities or [],
            "registered_at": time.time(),
            "last_heartbeat": None,
        }
        logger.info(f"[ControlUnit] Registered robot: {robot_name} ({robot_type})")

    def heartbeat(self, robot_name: str) -> None:
        """
        Robot nodes call POST /hitl/heartbeat to signal they are alive.
        Used by the dashboard to show online/offline status.
        """
        if robot_name in self._registered_robots:
            self._registered_robots[robot_name]["last_heartbeat"] = time.time()

    # ──────────────────────────────────────────────────────
    # Mode arbitration
    # ──────────────────────────────────────────────────────

    def switch_to_manual(self, robot_name: str) -> Dict[str, Any]:
        """
        Operator requests manual control.
        Returns a status dict for the API response.
        """
        ok = self._manager.take_manual_control(robot_name)
        if ok:
            return {
                "status": "ok",
                "robot": robot_name,
                "mode": RobotMode.MANUAL.name,
                "message": (
                    f"{robot_name} is now under manual control. "
                    "Autonomous execution has been suspended for this robot."
                ),
            }
        return {
            "status": "error",
            "robot": robot_name,
            "message": (
                f"Cannot take manual control of {robot_name} in its current "
                f"mode: {self._manager.get_robot_mode(robot_name)}"
            ),
        }

    def switch_to_autonomous(self, robot_name: str) -> Dict[str, Any]:
        """
        Operator releases manual control; robot resumes its LLM plan.
        """
        ok = self._manager.release_manual_control(robot_name)
        if ok:
            return {
                "status": "ok",
                "robot": robot_name,
                "mode": RobotMode.AUTONOMOUS.name,
                "message": (
                    f"{robot_name} has returned to autonomous execution "
                    "and will continue from the next pending phase."
                ),
            }
        return {
            "status": "error",
            "robot": robot_name,
            "message": (
                f"Cannot release {robot_name} – it is not under manual control."
            ),
        }

    # ──────────────────────────────────────────────────────
    # Manual command processing
    # ──────────────────────────────────────────────────────

    def dispatch_manual_command(
        self, robot_name: str, cmd_type: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and buffer a manual command for a robot.

        The robot's node polls GET /hitl/get_manual_command to retrieve it.
        """
        if self._manager.get_robot_mode(robot_name) != RobotMode.MANUAL.name:
            return {
                "status": "error",
                "message": (
                    f"{robot_name} is not under manual control. "
                    "Take manual control first via POST /hitl/take_manual_control."
                ),
            }

        if cmd_type not in MANUAL_COMMAND_SCHEMA:
            return {
                "status": "error",
                "message": (
                    f"Unknown command type '{cmd_type}'. "
                    f"Valid types: {list(MANUAL_COMMAND_SCHEMA.keys())}"
                ),
            }

        # Check robot type constraints
        robot_type = self._registered_robots.get(robot_name, {}).get("type", "ground")
        aerial_only = {"hover", "land", "takeoff"}
        if cmd_type in aerial_only and robot_type != "aerial":
            return {
                "status": "error",
                "message": f"Command '{cmd_type}' is only valid for aerial robots.",
            }

        # Validate required params
        required = MANUAL_COMMAND_SCHEMA[cmd_type]
        missing = [p for p in required if p not in params]
        if missing:
            return {
                "status": "error",
                "message": f"Missing required parameters for '{cmd_type}': {missing}",
            }

        command = {"type": cmd_type, "params": params}
        ok = self._manager.send_manual_command(robot_name, command)
        if ok:
            return {
                "status": "ok",
                "robot": robot_name,
                "command": command,
                "message": f"Command '{cmd_type}' dispatched to {robot_name}.",
            }
        return {"status": "error", "message": "Failed to buffer command."}

    def poll_manual_command(self, robot_name: str) -> Optional[Dict[str, Any]]:
        """
        Called by robot nodes to consume their next manual command.
        Returns None if no command is pending.
        """
        return self._manager.get_manual_command(robot_name)

    # ──────────────────────────────────────────────────────
    # Whole-mission controls
    # ──────────────────────────────────────────────────────

    def pause_all(self) -> Dict[str, Any]:
        ok = self._manager.pause_mission()
        return {
            "status": "ok" if ok else "error",
            "message": "Mission paused." if ok else "Mission is not currently executing.",
        }

    def resume_all(self) -> Dict[str, Any]:
        ok = self._manager.resume_mission()
        return {
            "status": "ok" if ok else "error",
            "message": "Mission resumed." if ok else "Mission is not paused.",
        }

    # ──────────────────────────────────────────────────────
    # Dashboard helpers
    # ──────────────────────────────────────────────────────

    def get_fleet_status(self) -> Dict[str, Any]:
        """
        Combined status of all registered robots for the HITL dashboard.
        Merges registration metadata with live mode from the state machine.
        """
        now = time.time()
        fleet = {}
        for robot, meta in self._registered_robots.items():
            last_hb = meta["last_heartbeat"]
            online = last_hb is not None and (now - last_hb) < 10.0  # 10-s window
            fleet[robot] = {
                "type": meta["type"],
                "mode": self._manager.get_robot_mode(robot),
                "online": online,
                "last_heartbeat_secs_ago": (
                    round(now - last_hb, 1) if last_hb else None
                ),
            }
        return fleet

    def get_valid_command_types(self, robot_name: str) -> List[str]:
        robot_type = self._registered_robots.get(robot_name, {}).get("type", "ground")
        aerial_only = {"hover", "land", "takeoff"}
        return [
            ct for ct in MANUAL_COMMAND_SCHEMA
            if robot_type == "aerial" or ct not in aerial_only
        ]
