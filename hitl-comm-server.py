"""
HITL Communication Server
=========================
Extended version of CyberScape's comm-server.py with full
Shared Autonomy / Human-in-the-Loop support.

New behaviour vs. the original comm-server.py
──────────────────────────────────────────────
1. After /generate_plan finishes running Manager.py, plans are placed in
   PENDING_REVIEW state instead of being immediately available for execution.
2. Robots CANNOT call /get_instruction until the operator approves the plan
   via POST /hitl/approve_plan.
3. During execution, any robot can be switched to MANUAL mode via
   POST /hitl/take_manual_control — causing /get_instruction to block until
   the robot is released back to AUTONOMOUS.
4. All /hitl/* endpoints are served by the Blueprint in hitl_endpoints.py.

Usage:
    python hitl-comm-server.py

Runs on port 5001 to avoid collision with the original server during testing.
Change PORT below to 5000 when deploying as the primary server.

Author: CyberScape Thesis Team – AUC × Siemens Digital Industries, Spring 2026
"""

import json
import logging
import os
import subprocess
import sys

from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# ── SharedAutonomy package ──────────────────────────────
# Ensure the project root is on the Python path so the
# SharedAutonomy sub-package can be found.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from SharedAutonomy.shared_autonomy_manager import get_manager, RobotMode
from SharedAutonomy.control_unit import ControlUnit
from SharedAutonomy.hitl_endpoints import create_hitl_blueprint

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────

CONFIG_FILE = "config.json"
PORT = 5001

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

with open(CONFIG_FILE, "r") as _f:
    config = json.load(_f)

# ---- COMMENTED OUT: OpenAI API (original) ----
# OPENAI_API_KEY = config.get("openai_api_key", "")
# ---- END COMMENTED OUT: OpenAI API ----

# ---- ACTIVE: LLaMA via SambaNova (free) ----
LLAMA_API_KEY = config.get("llama_api_key", "")

# ─────────────────────────────────────────────────────────
# Flask app + LLM
# ─────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="hitl_dashboard", static_url_path="/dashboard")
CORS(app)

# ---- COMMENTED OUT: OpenAI API (original) ----
# llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)
# ---- END COMMENTED OUT: OpenAI API ----

# ---- ACTIVE: LLaMA via SambaNova (free) ----
llm = ChatOpenAI(
    api_key=LLAMA_API_KEY,
    base_url="https://api.sambanova.ai/v1",
    model_name="Meta-Llama-3.3-70B-Instruct",
    temperature=0.1
)

# ─────────────────────────────────────────────────────────
# Shared Autonomy layer
# ─────────────────────────────────────────────────────────

control_unit = ControlUnit()

# Register robots from config with their types
ROBOT_TYPES = {
    "DRONE":     "aerial",
    "ROBOT_DOG": "ground",
    "TURTLEBOT": "ground",
    "UAV":       "aerial",
}

for robot_name in config.get("robots_in_curr_mission", []):
    rtype = ROBOT_TYPES.get(robot_name.upper(), "ground")
    control_unit.register_robot(robot_name.upper(), robot_type=rtype)

# Register the HITL Blueprint
hitl_bp = create_hitl_blueprint(llm, control_unit)
app.register_blueprint(hitl_bp)

# ─────────────────────────────────────────────────────────
# Plan store (mirrors original comm-server)
# ─────────────────────────────────────────────────────────

plans = {}
progress = {}


def load_plan(robot_name: str, filename: str) -> None:
    with open(filename) as f:
        plans[robot_name] = json.load(f)
        progress[robot_name] = {
            "completed_phases": set(),
            "outputs": {}
        }


# Load existing plans (if any) at startup – they still need approval
_initial_load_ok = True
for _robot in config.get("robots_in_curr_mission", []):
    _filename = config["robots_config"][_robot]["final_low"]
    if os.path.exists(_filename):
        load_plan(_robot, _filename)
    else:
        _initial_load_ok = False

if _initial_load_ok and plans:
    # Pre-existing plans need operator review before execution
    sa_manager = get_manager()
    sa_manager.submit_plans_for_review(
        plans, config.get("mission_text_file", "existing_mission")
    )
    logger.info(
        "Pre-loaded plans submitted for review. "
        "Approve via POST /hitl/approve_plan to enable execution."
    )

# ─────────────────────────────────────────────────────────
# Helper: fill-in variables (unchanged from original)
# ─────────────────────────────────────────────────────────

def fill_in_variables(plan_text: str, robot: str) -> str:
    if robot not in progress:
        return plan_text
    # Collect outputs from all robots across all phases
    all_outputs = {}
    for r, prog in progress.items():
        for phase_outputs in prog["outputs"].values():
            all_outputs.update(phase_outputs)
    if not all_outputs:
        return plan_text
    # Direct string replacement — replace each variable name with its value
    result = plan_text
    for var_name, var_value in all_outputs.items():
        result = result.replace(str(var_name), str(var_value))
    logger.info(f"[fill_in_variables] Substituted variables: {all_outputs}")
    return result


# ─────────────────────────────────────────────────────────
# Original CyberScape endpoints (HITL-aware)
# ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return jsonify({
        "message": "CyberScape HITL Multi-Robot Server is running",
        "hitl_dashboard": "/dashboard/index.html",
        "api_docs": {
            "original_endpoints": [
                "/generate_plan (POST)",
                "/get_instruction (GET)",
                "/complete_phase (POST)",
                "/report_error (POST)",
            ],
            "hitl_endpoints": "/hitl/status (GET) — see /hitl/* for full list"
        }
    }), 200


@app.route("/generate_plan", methods=["POST"])
def generate_plan():
    """
    Generate a new mission plan via Manager.py, then submit it for
    operator review (HITL gate) instead of immediately loading it.
    """
    data = request.get_json()
    if not data or "mission_title" not in data or "mission_text" not in data:
        return jsonify({
            "error": "Request must include 'mission_title' and 'mission_text'."
        }), 400

    mission_title = data["mission_title"]
    mission_text  = data["mission_text"]

    # Write mission file and update config (identical to original)
    mission_files_dir = "mission_files"
    os.makedirs(mission_files_dir, exist_ok=True)
    new_mission_file = os.path.join(mission_files_dir, f"{mission_title}.txt")

    try:
        with open(new_mission_file, "w") as f:
            f.write(mission_text)

        config_path = "config.json"
        with open(config_path, "r") as f:
            cfg = json.load(f)
        cfg["mission_text_file"] = new_mission_file
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=4)

        # Run the CyberScape planning pipeline
        subprocess.run(["python3", "Manager.py"], check=True)

        # Load generated plans into memory
        for robot in cfg["robots_in_curr_mission"]:
            filename = cfg["robots_config"][robot]["final_low"]
            load_plan(robot, filename)

        # ── HITL GATE: submit for operator review ──────────
        hitl_bp.plan_review.submit(plans, mission_title)
        # ────────────────────────────────────────────────────

        return jsonify({
            "message": (
                "Mission plan generated successfully. "
                "Waiting for operator approval before execution can begin. "
                "Review at GET /hitl/review_status and approve via POST /hitl/approve_plan."
            ),
            "review_url": "/hitl/review_status",
            "approve_url": "/hitl/approve_plan",
        }), 200

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Manager.py failed: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_instruction", methods=["GET"])
def get_instruction():
    """
    Retrieve the low-level plan for a phase.

    HITL changes vs. original:
      - Returns 403 if plans have not been approved yet.
      - Returns 423 (Locked) if the robot is under manual control or paused.
    """
    robot = request.args.get("robot")
    phase = request.args.get("phase")

    if not robot or not phase:
        return jsonify({"error": "Missing 'robot' or 'phase' parameters."}), 400

    robot = robot.upper()

    try:
        phase = int(phase)
    except ValueError:
        return jsonify({"error": "Phase must be an integer."}), 400

    # ── HITL Gate 1: plan must be approved ──
    sa_manager = get_manager()
    if not hitl_bp.plan_review.is_approved():
        return jsonify({
            "error": "Plans have not been approved by the operator yet.",
            "review_state": sa_manager.get_review_state(),
            "approve_url": "/hitl/approve_plan",
        }), 403

    # ── HITL Gate 2: robot must be in autonomous mode ──
    if sa_manager.is_robot_paused_or_manual(robot):
        return jsonify({
            "error": (
                f"Robot '{robot}' is currently under manual or paused mode. "
                "Release control via POST /hitl/release_manual_control."
            ),
            "robot_mode": sa_manager.get_robot_mode(robot),
        }), 423   # 423 Locked

    # ── Original logic ──
    if robot not in plans:
        return jsonify({"error": f"No plan found for robot '{robot}'."}), 404

    phases = plans[robot]["phases"]
    matching_phase = next((p for p in phases if p["phase_number"] == phase), None)
    if not matching_phase:
        return jsonify({"error": f"Phase {phase} not found for robot {robot}."}), 404

    for p in range(1, phase):
        if p not in progress[robot]["completed_phases"]:
            return jsonify({
                "error": f"Phase {p} must be completed before requesting phase {phase}."
            }), 400

    return jsonify({
        "phase_number":      phase,
        "low_level_plan":    matching_phase["low_level_plan"],
        "expected_outputs":  matching_phase.get("outputs", {}),
        "robot_mode":        sa_manager.get_robot_mode(robot),
    }), 200


@app.route("/complete_phase", methods=["POST"])
def complete_phase():
    """Mark a phase as completed (unchanged from original, with mode check)."""
    data = request.get_json()
    if not data or "robot" not in data or "phase" not in data or "outputs" not in data:
        return jsonify({
            "error": "Request must include 'robot', 'phase', and 'outputs'."
        }), 400

    robot   = data["robot"].upper()
    phase   = data["phase"]
    outputs = data["outputs"]

    try:
        phase = int(phase)
    except ValueError:
        return jsonify({"error": "Phase must be an integer."}), 400

    # Mode check
    sa_manager = get_manager()
    if sa_manager.is_robot_paused_or_manual(robot):
        return jsonify({
            "error": f"Robot '{robot}' is in {sa_manager.get_robot_mode(robot)} mode.",
        }), 423

    if robot not in plans:
        return jsonify({"error": f"No plan found for robot '{robot}'."}), 404

    phases = plans[robot]["phases"]
    matching_phase = next((p for p in phases if p["phase_number"] == phase), None)
    if not matching_phase:
        return jsonify({"error": f"Phase {phase} not found."}), 404

    for p in range(1, phase):
        if p not in progress[robot]["completed_phases"]:
            return jsonify({
                "error": f"Phase {p} must be completed before marking phase {phase}."
            }), 400

    required_outputs = matching_phase.get("outputs", {})
    missing = [k for k in required_outputs if k not in outputs]
    if missing:
        return jsonify({"error": f"Missing required outputs: {missing}"}), 400

    progress[robot]["completed_phases"].add(phase)
    progress[robot]["outputs"][phase] = outputs

    next_phase = next((p for p in phases if p["phase_number"] == phase + 1), None)
    if next_phase:
        input_variables = next_phase.get("inputs", [])
        if input_variables:
            updated_plan = fill_in_variables(next_phase["low_level_plan"], robot)
            next_phase["low_level_plan"] = updated_plan

    # Auto-mark robot as completed when all phases done
    if not next_phase:
        sa_manager.mark_robot_completed(robot)

    return jsonify({
        "message": f"Phase {phase} for {robot} completed.",
        "stored_outputs": outputs,
    }), 200


@app.route("/report_error", methods=["POST"])
def report_error():
    """Report an execution error and trigger adaptive replanning."""
    data = request.get_json()
    if not data or not all(
        k in data for k in ("robot", "phase", "instruction_number", "description")
    ):
        return jsonify({
            "error": "Request must include 'robot', 'phase', 'instruction_number', 'description'."
        }), 400

    robot              = data["robot"].upper()
    phase              = data["phase"]
    instruction_number = data["instruction_number"]
    description        = data["description"]

    try:
        phase = int(phase)
    except ValueError:
        return jsonify({"error": "Phase must be an integer."}), 400

    if robot not in plans:
        return jsonify({"error": f"No plan found for robot '{robot}'."}), 404

    phases = plans[robot]["phases"]
    if not any(p["phase_number"] == phase for p in phases):
        return jsonify({"error": f"Phase {phase} not found."}), 404

    for p in range(1, phase):
        if p not in progress[robot]["completed_phases"]:
            return jsonify({
                "error": f"Phase {p} must be completed before reporting error in phase {phase}."
            }), 400

    # Log error in shared autonomy layer
    get_manager().mark_robot_error(robot, description)

    error_file_path = os.path.join(
        "Execution_Errors_Files", f"error_{robot}_phase_{phase}.json"
    )
    os.makedirs("Execution_Errors_Files", exist_ok=True)

    with open(error_file_path, "w") as f:
        json.dump({
            "description": description,
            "failed_instruction_number": instruction_number,
        }, f, indent=4)

    try:
        subprocess.run(
            ["python3", "APM.py", robot, str(phase), instruction_number, error_file_path],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"APM execution failed: {e}")
        return jsonify({"error": "APM execution failed."}), 500

    # Reload corrected plan and re-submit for review
    with open("config.json", "r") as f:
        cfg = json.load(f)
    load_plan(robot, cfg["robots_config"][robot]["final_low"])

    # After adaptive replanning, operator must re-approve the updated plan
    get_manager().submit_plans_for_review(
        {robot: plans[robot]}, f"APM replan – {robot} phase {phase}"
    )

    return jsonify({
        "message": f"Error in phase {phase} for {robot} processed. "
                   "Updated plan submitted for operator review.",
        "updated_plan": plans[robot],
        "review_url": "/hitl/review_status",
    }), 200


# ─────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  CyberScape HITL Communication Server")
    logger.info(f"  Listening on http://0.0.0.0:{PORT}")
    logger.info(f"  Dashboard  : http://localhost:{PORT}/dashboard/index.html")
    logger.info(f"  HITL Status: http://localhost:{PORT}/hitl/status")
    logger.info("=" * 60)
    app.run(host="0.0.0.0", port=PORT, debug=False)
