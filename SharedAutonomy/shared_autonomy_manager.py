"""
SharedAutonomyManager
=====================
Core state machine for Shared Autonomy integration in the CyberScape framework.

Implements Human-in-the-Loop (HITL) control at both the:
  - Planning stage  : operator reviews, modifies, approves/rejects LLM-generated plans
  - Execution stage : operator can pause, resume, or take manual control of any robot

Based on: Ghaly et al. [4] – shared autonomy architecture for swarm robotics,
          and thesis Objective 6 (Spring 2026 Capstone).
"""

import threading
import time
from enum import Enum, auto
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# State Enumerations
# ─────────────────────────────────────────────────────────

class PlanReviewState(Enum):
    """Lifecycle of an LLM-generated plan before execution."""
    NOT_SUBMITTED   = auto()   # No plan yet
    PENDING_REVIEW  = auto()   # Waiting for operator decision
    APPROVED        = auto()   # Operator approved – ready to run
    REJECTED        = auto()   # Operator rejected – triggers replanning
    MODIFICATION_REQUESTED = auto()  # Operator requested NL edits (in progress)
    MODIFIED        = auto()   # Edits applied, back to PENDING_REVIEW


class RobotMode(Enum):
    """Per-robot execution mode."""
    IDLE            = auto()   # Not executing any plan phase
    AUTONOMOUS      = auto()   # Executing LLM plan autonomously
    PAUSED          = auto()   # Autonomous execution paused by operator
    MANUAL          = auto()   # Operator has direct control
    COMPLETED       = auto()   # Mission finished
    ERROR           = auto()   # Unrecoverable error


class MissionState(Enum):
    """Overall mission lifecycle."""
    IDLE            = auto()
    PLANNING        = auto()   # Manager.py / LLM is generating plan
    AWAITING_APPROVAL = auto() # Plan exists but not yet approved
    EXECUTING       = auto()   # At least one robot is running
    PAUSED          = auto()   # All robots paused by operator
    COMPLETED       = auto()
    ABORTED         = auto()


# ─────────────────────────────────────────────────────────
# Core Manager
# ─────────────────────────────────────────────────────────

class SharedAutonomyManager:
    """
    Thread-safe singleton that tracks plan-review state and per-robot
    execution modes.  All Flask endpoint handlers delegate to this class.
    """

    _instance: Optional["SharedAutonomyManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialised = False
            return cls._instance

    def __init__(self):
        if self._initialised:
            return
        self._initialised = True

        # ── Plan review state (one pending plan at a time) ──
        self._review_state: PlanReviewState = PlanReviewState.NOT_SUBMITTED
        self._pending_plans: Dict[str, Any] = {}          # robot_name → plan dict
        self._modification_request: Optional[str] = None  # NL text from operator
        self._review_comments: Optional[str] = None       # operator notes
        self._review_timestamp: Optional[float] = None

        # ── Per-robot execution modes ──
        self._robot_modes: Dict[str, RobotMode] = {}
        self._manual_commands: Dict[str, Any] = {}        # buffered manual commands

        # ── Mission state ──
        self._mission_state: MissionState = MissionState.IDLE
        self._mission_title: Optional[str] = None

        # ── Event log for audit / dashboard ──
        self._event_log: list = []

        # ── Thread lock for state mutations ──
        self._state_lock = threading.RLock()

        logger.info("[SharedAutonomy] Manager initialised.")

    # ──────────────────────────────────────────────────────
    # Plan Review API
    # ──────────────────────────────────────────────────────

    def submit_plans_for_review(self, plans: Dict[str, Any], mission_title: str) -> None:
        """
        Called by generate_plan() after Manager.py finishes.
        Puts all robot plans into PENDING_REVIEW state.
        """
        with self._state_lock:
            self._pending_plans = plans
            self._mission_title = mission_title
            self._review_state = PlanReviewState.PENDING_REVIEW
            self._review_timestamp = time.time()
            self._mission_state = MissionState.AWAITING_APPROVAL
            self._log_event("PLAN_SUBMITTED", {
                "mission": mission_title,
                "robots": list(plans.keys())
            })
            logger.info(f"[SharedAutonomy] Plans submitted for review: {list(plans.keys())}")

    def approve_plans(self, comments: Optional[str] = None) -> bool:
        """
        Operator approves all pending plans.
        Returns True if transition was valid.
        """
        with self._state_lock:
            if self._review_state not in (
                PlanReviewState.PENDING_REVIEW,
                PlanReviewState.MODIFIED,
            ):
                return False
            self._review_state = PlanReviewState.APPROVED
            self._review_comments = comments
            self._mission_state = MissionState.EXECUTING
            # Initialise robots to AUTONOMOUS mode
            for robot in self._pending_plans:
                self._robot_modes[robot] = RobotMode.AUTONOMOUS
            self._log_event("PLAN_APPROVED", {"comments": comments})
            logger.info("[SharedAutonomy] Plans approved by operator.")
            return True

    def reject_plans(self, reason: Optional[str] = None) -> bool:
        """
        Operator rejects plans – caller should trigger replanning.
        """
        with self._state_lock:
            if self._review_state not in (
                PlanReviewState.PENDING_REVIEW,
                PlanReviewState.MODIFIED,
                PlanReviewState.APPROVED,
            ):
                return False
            self._review_state = PlanReviewState.REJECTED
            self._mission_state = MissionState.PLANNING
            self._log_event("PLAN_REJECTED", {"reason": reason})
            logger.info(f"[SharedAutonomy] Plans rejected: {reason}")
            return True

    def request_plan_modification(self, nl_request: str) -> bool:
        """
        Operator requests an NL-based modification to the pending plan.
        The PlanReviewModule will process nl_request and call update_modified_plans().
        """
        with self._state_lock:
            if self._review_state != PlanReviewState.PENDING_REVIEW:
                return False
            self._review_state = PlanReviewState.MODIFICATION_REQUESTED
            self._modification_request = nl_request
            self._log_event("MODIFICATION_REQUESTED", {"request": nl_request})
            logger.info(f"[SharedAutonomy] Modification requested: {nl_request}")
            return True

    def update_modified_plans(self, updated_plans: Dict[str, Any]) -> None:
        """
        Called by PlanReviewModule after LLM edits are applied.
        Puts plans back to PENDING_REVIEW for final operator sign-off.
        """
        with self._state_lock:
            self._pending_plans = updated_plans
            self._review_state = PlanReviewState.MODIFIED
            self._modification_request = None
            self._log_event("PLAN_MODIFIED", {"robots": list(updated_plans.keys())})
            logger.info("[SharedAutonomy] Modified plans ready for re-review.")

    # ──────────────────────────────────────────────────────
    # Execution Control API
    # ──────────────────────────────────────────────────────

    def take_manual_control(self, robot: str) -> bool:
        """
        Operator seizes control of a specific robot.
        Robot pauses its LLM-plan execution until released.
        """
        with self._state_lock:
            if robot not in self._robot_modes:
                self._robot_modes[robot] = RobotMode.MANUAL
                return True
            if self._robot_modes[robot] in (RobotMode.AUTONOMOUS, RobotMode.PAUSED):
                self._robot_modes[robot] = RobotMode.MANUAL
                self._log_event("MANUAL_CONTROL_TAKEN", {"robot": robot})
                logger.info(f"[SharedAutonomy] Manual control taken: {robot}")
                return True
            return False

    def release_manual_control(self, robot: str) -> bool:
        """
        Operator releases manual control; robot resumes autonomous execution.
        """
        with self._state_lock:
            if self._robot_modes.get(robot) != RobotMode.MANUAL:
                return False
            self._robot_modes[robot] = RobotMode.AUTONOMOUS
            # Clear any buffered manual commands for this robot
            self._manual_commands.pop(robot, None)
            self._log_event("MANUAL_CONTROL_RELEASED", {"robot": robot})
            logger.info(f"[SharedAutonomy] Manual control released: {robot}")
            return True

    def send_manual_command(self, robot: str, command: Dict[str, Any]) -> bool:
        """
        Buffer a manual command for a robot that is under manual control.
        The robot's ROS node polls this via /hitl/get_manual_command.
        """
        with self._state_lock:
            if self._robot_modes.get(robot) != RobotMode.MANUAL:
                return False
            self._manual_commands[robot] = {
                "command": command,
                "timestamp": time.time()
            }
            self._log_event("MANUAL_COMMAND_SENT", {"robot": robot, "command": command})
            return True

    def get_manual_command(self, robot: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve and clear the buffered manual command for a robot.
        """
        with self._state_lock:
            return self._manual_commands.pop(robot, None)

    def pause_mission(self) -> bool:
        """Pause ALL autonomous robots simultaneously."""
        with self._state_lock:
            if self._mission_state != MissionState.EXECUTING:
                return False
            for robot, mode in self._robot_modes.items():
                if mode == RobotMode.AUTONOMOUS:
                    self._robot_modes[robot] = RobotMode.PAUSED
            self._mission_state = MissionState.PAUSED
            self._log_event("MISSION_PAUSED", {})
            logger.info("[SharedAutonomy] Mission paused.")
            return True

    def resume_mission(self) -> bool:
        """Resume all paused robots."""
        with self._state_lock:
            if self._mission_state != MissionState.PAUSED:
                return False
            for robot, mode in self._robot_modes.items():
                if mode == RobotMode.PAUSED:
                    self._robot_modes[robot] = RobotMode.AUTONOMOUS
            self._mission_state = MissionState.EXECUTING
            self._log_event("MISSION_RESUMED", {})
            logger.info("[SharedAutonomy] Mission resumed.")
            return True

    def mark_robot_completed(self, robot: str) -> None:
        with self._state_lock:
            self._robot_modes[robot] = RobotMode.COMPLETED
            if all(m == RobotMode.COMPLETED for m in self._robot_modes.values()):
                self._mission_state = MissionState.COMPLETED
                self._log_event("MISSION_COMPLETED", {})

    def mark_robot_error(self, robot: str, description: str) -> None:
        with self._state_lock:
            self._robot_modes[robot] = RobotMode.ERROR
            self._log_event("ROBOT_ERROR", {"robot": robot, "error": description})

    # ──────────────────────────────────────────────────────
    # Query helpers
    # ──────────────────────────────────────────────────────

    def get_robot_mode(self, robot: str) -> Optional[str]:
        mode = self._robot_modes.get(robot)
        return mode.name if mode else None

    def is_robot_autonomous(self, robot: str) -> bool:
        return self._robot_modes.get(robot) == RobotMode.AUTONOMOUS

    def is_robot_paused_or_manual(self, robot: str) -> bool:
        return self._robot_modes.get(robot) in (RobotMode.MANUAL, RobotMode.PAUSED)

    def get_review_state(self) -> str:
        return self._review_state.name

    def get_pending_plans(self) -> Dict[str, Any]:
        return dict(self._pending_plans)

    def get_modification_request(self) -> Optional[str]:
        return self._modification_request

    def get_full_status(self) -> Dict[str, Any]:
        """Snapshot of the entire shared-autonomy state for the dashboard."""
        with self._state_lock:
            return {
                "mission_state":  self._mission_state.name,
                "mission_title":  self._mission_title,
                "review_state":   self._review_state.name,
                "modification_request": self._modification_request,
                "review_comments": self._review_comments,
                "review_timestamp": self._review_timestamp,
                "robots": {
                    robot: {
                        "mode": mode.name,
                        "has_pending_command": robot in self._manual_commands
                    }
                    for robot, mode in self._robot_modes.items()
                },
                "recent_events":  self._event_log[-20:]
            }

    # ──────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────

    def _log_event(self, event_type: str, data: Dict) -> None:
        self._event_log.append({
            "type": event_type,
            "timestamp": time.time(),
            "data": data
        })
        # Keep only the last 200 events
        if len(self._event_log) > 200:
            self._event_log = self._event_log[-200:]


# ─────────────────────────────────────────────────────────
# Module-level singleton accessor
# ─────────────────────────────────────────────────────────

def get_manager() -> SharedAutonomyManager:
    """Return the process-wide SharedAutonomyManager singleton."""
    return SharedAutonomyManager()
