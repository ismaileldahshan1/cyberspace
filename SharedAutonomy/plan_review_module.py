"""
PlanReviewModule
================
Handles the planning-stage portion of Shared Autonomy:

  1. Receives LLM-generated plans from CyberScape's Manager.py pipeline
  2. Exposes them for operator inspection (via the HITL dashboard / API)
  3. Applies NL-requested modifications using the LLM (GPT-4o / LLaMA)
  4. Returns the (possibly modified) plans back to the execution pipeline
     only after the operator explicitly approves them

This module works closely with SharedAutonomyManager (state machine) and
is called by the Flask HITL endpoints.

References:
  - Sayed et al. [2] : CyberScape rubric-based plan verification pattern
  - Goodrich & Schultz [16] : HITL planning frameworks
  - Desai et al. [17] : Trust calibration in adjustable autonomy
"""

import json
import logging
import threading
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from SharedAutonomy.shared_autonomy_manager import get_manager, PlanReviewState

logger = logging.getLogger(__name__)


class PlanReviewModule:
    """
    Manages the full plan-review lifecycle for a single mission.

    Usage (inside Flask endpoint):
        prm = PlanReviewModule(llm)
        prm.submit(plans, mission_title)    # after Manager.py runs
        # … operator reviews via dashboard …
        prm.apply_modification(nl_text)     # if operator requests edits
        approved_plans = prm.get_approved_plans()
    """

    def __init__(self, llm: ChatOpenAI):
        self._llm = llm
        self._manager = get_manager()
        self._lock = threading.Lock()

    # ──────────────────────────────────────────────────────
    # Step 1 – Submit plans for review
    # ──────────────────────────────────────────────────────

    def submit(self, plans: Dict[str, Any], mission_title: str) -> None:
        """
        Entry point called right after Manager.py finishes generating plans.
        Puts all robot plans into PENDING_REVIEW state so the operator can
        inspect them before execution begins.
        """
        self._manager.submit_plans_for_review(plans, mission_title)
        logger.info(f"[PlanReview] Submitted plans for '{mission_title}'.")

    # ──────────────────────────────────────────────────────
    # Step 2a – Operator approves
    # ──────────────────────────────────────────────────────

    def approve(self, comments: Optional[str] = None) -> bool:
        return self._manager.approve_plans(comments)

    # ──────────────────────────────────────────────────────
    # Step 2b – Operator rejects
    # ──────────────────────────────────────────────────────

    def reject(self, reason: Optional[str] = None) -> bool:
        return self._manager.reject_plans(reason)

    # ──────────────────────────────────────────────────────
    # Step 2c – Operator requests NL-based modification
    # ──────────────────────────────────────────────────────

    def request_modification(self, nl_request: str) -> bool:
        """
        Records the operator's NL request and kicks off async LLM editing.
        The dashboard will poll GET /hitl/review_status to see when it's done.
        """
        ok = self._manager.request_plan_modification(nl_request)
        if ok:
            # Run modification in background so the HTTP response is immediate
            t = threading.Thread(
                target=self._apply_modification_async,
                args=(nl_request,),
                daemon=True,
            )
            t.start()
        return ok

    def _apply_modification_async(self, nl_request: str) -> None:
        """Background worker: calls the LLM to edit each robot's plan."""
        with self._lock:
            current_plans = self._manager.get_pending_plans()
            updated_plans: Dict[str, Any] = {}

            for robot, plan in current_plans.items():
                try:
                    updated_plan = self._modify_plan_with_llm(plan, robot, nl_request)
                    updated_plans[robot] = updated_plan
                    logger.info(f"[PlanReview] Plan modified for {robot}.")
                except Exception as exc:
                    logger.error(f"[PlanReview] LLM modification failed for {robot}: {exc}")
                    updated_plans[robot] = plan  # fall back to original

            self._manager.update_modified_plans(updated_plans)

    def _modify_plan_with_llm(
        self,
        plan: Dict[str, Any],
        robot_name: str,
        nl_request: str,
    ) -> Dict[str, Any]:
        """
        Sends the current plan + operator's NL request to the LLM and
        returns a modified plan in the same JSON schema.

        Follows the same rubric-based iterative refinement pattern used in
        CyberScape's Verification_Module.py (Sayed et al. [2]).
        """
        plan_json = json.dumps(plan, indent=2)

        prompt = f"""You are a mission plan editor for a multi-robot system.
The robot is: {robot_name}

CURRENT MISSION PLAN (JSON):
{plan_json}

OPERATOR'S MODIFICATION REQUEST:
{nl_request}

Your task:
1. Understand the operator's request carefully.
2. Modify the mission plan JSON to satisfy the request while preserving the
   overall mission objective and the existing JSON schema.
3. Only modify phases or instructions that are directly relevant to the request.
4. Keep the same JSON structure: "target", "phases" array with "phase_number",
   "state", "phase_target", "inputs", "outputs", and "low_level_plan" fields.
5. If the modification is impossible or unsafe, return the ORIGINAL plan unchanged
   and add a top-level "modification_note" field explaining why.

Return ONLY the modified JSON, with no extra commentary."""

        response = self._llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )

        return json.loads(raw)

    # ──────────────────────────────────────────────────────
    # Getters
    # ──────────────────────────────────────────────────────

    def get_review_state(self) -> str:
        return self._manager.get_review_state()

    def is_approved(self) -> bool:
        return self._manager.get_review_state() == PlanReviewState.APPROVED.name

    def get_approved_plans(self) -> Optional[Dict[str, Any]]:
        """Returns plans only if they have been approved; otherwise None."""
        if self.is_approved():
            return self._manager.get_pending_plans()
        return None

    def get_status_summary(self) -> Dict[str, Any]:
        """Light summary for dashboard polling."""
        state = self._manager.get_review_state()
        return {
            "review_state": state,
            "modification_in_progress": (
                state == PlanReviewState.MODIFICATION_REQUESTED.name
            ),
            "modification_request": self._manager.get_modification_request(),
            "robots": list(self._manager.get_pending_plans().keys()),
        }
