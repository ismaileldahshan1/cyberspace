"""
HITL Flask Blueprint
====================
All Human-in-the-Loop API endpoints, registered under the /hitl prefix.

Endpoint map
────────────
Planning stage:
  GET  /hitl/status               → full shared-autonomy snapshot
  GET  /hitl/review_status        → plan review state + pending plans
  POST /hitl/approve_plan         → operator approves pending plans
  POST /hitl/reject_plan          → operator rejects (triggers replanning)
  POST /hitl/modify_plan          → operator NL modification request
  GET  /hitl/get_approved_plans   → retrieve approved plan JSON (for executor)

Execution stage:
  GET  /hitl/fleet_status                    → all robots' mode & online status
  GET  /hitl/robot_mode/<robot>              → single robot's current mode
  POST /hitl/take_manual_control             → seize control of a robot
  POST /hitl/release_manual_control          → return robot to autonomous
  POST /hitl/send_manual_command             → dispatch a manual command
  GET  /hitl/get_manual_command/<robot>      → robot node polls this
  POST /hitl/heartbeat                       → robot node keep-alive signal
  POST /hitl/pause_mission                   → pause all autonomous robots
  POST /hitl/resume_mission                  → resume paused robots
  POST /hitl/mark_completed                  → robot reports mission phase done
  POST /hitl/mark_error                      → robot reports an error
"""

import logging
from flask import Blueprint, jsonify, request

from SharedAutonomy.shared_autonomy_manager import get_manager
from SharedAutonomy.plan_review_module import PlanReviewModule
from SharedAutonomy.control_unit import ControlUnit

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Blueprint factory
# ─────────────────────────────────────────────────────────

def create_hitl_blueprint(llm, control_unit: ControlUnit) -> Blueprint:
    """
    Factory function so the main server can inject its LLM instance and
    the shared ControlUnit before registering the blueprint.
    """
    bp = Blueprint("hitl", __name__, url_prefix="/hitl")
    plan_review = PlanReviewModule(llm)
    manager = get_manager()

    # ── attach objects to blueprint for reuse ──
    bp.plan_review = plan_review
    bp.control_unit = control_unit

    # ─────────────────────────────────────────────────────
    # General status
    # ─────────────────────────────────────────────────────

    @bp.route("/status", methods=["GET"])
    def status():
        """Full shared-autonomy snapshot – used by the HITL dashboard."""
        return jsonify(manager.get_full_status()), 200

    # ─────────────────────────────────────────────────────
    # Planning-stage endpoints
    # ─────────────────────────────────────────────────────

    @bp.route("/review_status", methods=["GET"])
    def review_status():
        """Current plan-review state and metadata."""
        return jsonify(plan_review.get_status_summary()), 200

    @bp.route("/approve_plan", methods=["POST"])
    def approve_plan():
        """
        Operator approves the currently pending plan.
        Optional JSON body: {"comments": "Looks good"}
        """
        data = request.get_json(silent=True) or {}
        comments = data.get("comments")
        ok = plan_review.approve(comments)
        if ok:
            return jsonify({
                "status": "approved",
                "message": "Plans approved. Execution can now begin.",
                "comments": comments,
            }), 200
        return jsonify({
            "status": "error",
            "message": (
                f"Cannot approve plans in current state: "
                f"{plan_review.get_review_state()}"
            ),
        }), 409

    @bp.route("/reject_plan", methods=["POST"])
    def reject_plan():
        """
        Operator rejects the current plan.
        JSON body: {"reason": "The scan pattern is too slow"}
        """
        data = request.get_json(silent=True) or {}
        reason = data.get("reason")
        ok = plan_review.reject(reason)
        if ok:
            return jsonify({
                "status": "rejected",
                "message": "Plans rejected. Please regenerate the mission plan.",
                "reason": reason,
            }), 200
        return jsonify({
            "status": "error",
            "message": (
                f"Cannot reject plans in current state: "
                f"{plan_review.get_review_state()}"
            ),
        }), 409

    @bp.route("/modify_plan", methods=["POST"])
    def modify_plan():
        """
        Operator requests an NL-based modification to the pending plan.
        JSON body: {"request": "Make the drone scan in a spiral pattern instead"}
        The modification runs asynchronously; poll GET /hitl/review_status
        for state == 'MODIFIED' before calling approve_plan.
        """
        data = request.get_json(silent=True) or {}
        nl_request = data.get("request", "").strip()
        if not nl_request:
            return jsonify({
                "status": "error",
                "message": "JSON body must include a 'request' field.",
            }), 400

        ok = plan_review.request_modification(nl_request)
        if ok:
            return jsonify({
                "status": "processing",
                "message": (
                    "Modification request received. The LLM is editing the plan. "
                    "Poll GET /hitl/review_status until state == 'MODIFIED'."
                ),
            }), 202
        return jsonify({
            "status": "error",
            "message": (
                f"Cannot request modification in current state: "
                f"{plan_review.get_review_state()}"
            ),
        }), 409

    @bp.route("/pending_plans", methods=["GET"])
    def pending_plans():
        """
        Retrieve the pending plans for operator review.
        Available in any review state (PENDING_REVIEW, MODIFIED, APPROVED, etc.).
        Use this to inspect plans before calling approve_plan.
        """
        plans = manager.get_pending_plans()
        if not plans:
            return jsonify({
                "status": "error",
                "message": "No plans are currently pending review.",
            }), 404
        return jsonify({
            "status": "ok",
            "review_state": manager.get_review_state(),
            "plans": plans,
        }), 200

    @bp.route("/get_approved_plans", methods=["GET"])
    def get_approved_plans():
        """
        Retrieve approved plan JSON.
        Returns 403 if plans have not yet been approved.
        Used by the comm-server executor to unlock plan loading.
        """
        approved = plan_review.get_approved_plans()
        if approved is None:
            return jsonify({
                "status": "error",
                "message": (
                    "Plans are not yet approved. "
                    f"Current review state: {plan_review.get_review_state()}"
                ),
            }), 403
        return jsonify({"status": "approved", "plans": approved}), 200

    # ─────────────────────────────────────────────────────
    # Execution-stage endpoints
    # ─────────────────────────────────────────────────────

    @bp.route("/fleet_status", methods=["GET"])
    def fleet_status():
        """Mode and online status for all registered robots."""
        return jsonify(control_unit.get_fleet_status()), 200

    @bp.route("/robot_mode/<robot>", methods=["GET"])
    def robot_mode(robot):
        """Current execution mode for a single robot."""
        mode = manager.get_robot_mode(robot.upper())
        if mode is None:
            return jsonify({"status": "error", "message": f"Robot '{robot}' not found."}), 404
        return jsonify({"robot": robot.upper(), "mode": mode}), 200

    @bp.route("/take_manual_control", methods=["POST"])
    def take_manual_control():
        """
        Operator seizes control of a robot.
        JSON body: {"robot": "DRONE"}
        """
        data = request.get_json(silent=True) or {}
        robot = data.get("robot", "").upper()
        if not robot:
            return jsonify({"status": "error", "message": "Missing 'robot' field."}), 400
        result = control_unit.switch_to_manual(robot)
        code = 200 if result["status"] == "ok" else 409
        return jsonify(result), code

    @bp.route("/release_manual_control", methods=["POST"])
    def release_manual_control():
        """
        Operator returns control of a robot to autonomous mode.
        JSON body: {"robot": "DRONE"}
        """
        data = request.get_json(silent=True) or {}
        robot = data.get("robot", "").upper()
        if not robot:
            return jsonify({"status": "error", "message": "Missing 'robot' field."}), 400
        result = control_unit.switch_to_autonomous(robot)
        code = 200 if result["status"] == "ok" else 409
        return jsonify(result), code

    @bp.route("/send_manual_command", methods=["POST"])
    def send_manual_command():
        """
        Dispatch a manual command to a robot under manual control.

        JSON body:
        {
            "robot":   "ROBOT_DOG",
            "command": "move",
            "params":  {"linear_x": 0.3, "angular_z": 0.0}
        }
        """
        data = request.get_json(silent=True) or {}
        robot   = data.get("robot", "").upper()
        cmd     = data.get("command", "")
        params  = data.get("params", {})

        if not robot or not cmd:
            return jsonify({
                "status": "error",
                "message": "JSON body must include 'robot' and 'command'.",
            }), 400

        result = control_unit.dispatch_manual_command(robot, cmd, params)
        code = 200 if result["status"] == "ok" else 400
        return jsonify(result), code

    @bp.route("/get_manual_command/<robot>", methods=["GET"])
    def get_manual_command(robot):
        """
        Robot node polls this to consume its next manual command.
        Returns 204 No Content when no command is pending.
        """
        cmd = control_unit.poll_manual_command(robot.upper())
        if cmd is None:
            return "", 204
        return jsonify(cmd), 200

    @bp.route("/valid_commands/<robot>", methods=["GET"])
    def valid_commands(robot):
        """List of valid manual command types for a given robot type."""
        cmds = control_unit.get_valid_command_types(robot.upper())
        return jsonify({"robot": robot.upper(), "commands": cmds}), 200

    @bp.route("/heartbeat", methods=["POST"])
    def heartbeat():
        """
        Robot nodes POST here every few seconds to indicate they are online.
        JSON body: {"robot": "DRONE"}
        """
        data = request.get_json(silent=True) or {}
        robot = data.get("robot", "").upper()
        if not robot:
            return jsonify({"status": "error", "message": "Missing 'robot' field."}), 400
        control_unit.heartbeat(robot)
        return jsonify({"status": "ok", "robot": robot}), 200

    @bp.route("/pause_mission", methods=["POST"])
    def pause_mission():
        """Pause all autonomous robots."""
        result = control_unit.pause_all()
        code = 200 if result["status"] == "ok" else 409
        return jsonify(result), code

    @bp.route("/resume_mission", methods=["POST"])
    def resume_mission():
        """Resume all paused robots."""
        result = control_unit.resume_all()
        code = 200 if result["status"] == "ok" else 409
        return jsonify(result), code

    @bp.route("/mark_completed", methods=["POST"])
    def mark_completed():
        """
        Robot node reports that its mission is complete.
        JSON body: {"robot": "DRONE"}
        """
        data = request.get_json(silent=True) or {}
        robot = data.get("robot", "").upper()
        if not robot:
            return jsonify({"status": "error", "message": "Missing 'robot'."}), 400
        manager.mark_robot_completed(robot)
        return jsonify({"status": "ok", "robot": robot, "mode": "COMPLETED"}), 200

    @bp.route("/mark_error", methods=["POST"])
    def mark_error():
        """
        Robot node reports an unrecoverable error.
        JSON body: {"robot": "ROBOT_DOG", "description": "Motor failure"}
        """
        data = request.get_json(silent=True) or {}
        robot = data.get("robot", "").upper()
        desc  = data.get("description", "Unknown error")
        if not robot:
            return jsonify({"status": "error", "message": "Missing 'robot'."}), 400
        manager.mark_robot_error(robot, desc)
        return jsonify({"status": "ok", "robot": robot, "mode": "ERROR"}), 200

    return bp
