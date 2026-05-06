import os
import argparse
import re
import json
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from Utils import read_file
from langchain_google_genai import GoogleGenerativeAI

import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

def llm_invoke_with_retry(llm, messages, retries=5, base_delay=30):
    for attempt in range(1, retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = base_delay * attempt
                print(f"[LLM] Rate limit hit (attempt {attempt}/{retries}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"LLM call failed after {retries} retries.")


# Load Gemini Pro Model

class PlanPhase:
    """Represents a single phase in a mission plan."""
    def __init__(self, target, phase_number, state, phase_target, inputs=None, outputs=None):
        self.target = target  # Identifier for the entity (e.g., "DRONE", "ROBOT_DOG")
        self.phase_number = phase_number
        self.state = state  # Represents the mission state before executing this phase
        self.phase_target = phase_target  # Independent mission objective for this phase
        self.inputs = inputs if inputs else []  # Required inputs for this phase
        self.outputs = outputs if outputs else []  # Outputs passed to the next phase

    def to_dict(self):
        """Converts the phase details to a dictionary."""
        return {
            "target": self.target,
            "phase_number": self.phase_number,
            "state": self.state,
            "phase_target": self.phase_target,
            "inputs": self.inputs,
            "outputs": self.outputs
        }


class Plan:
    """Represents a structured plan for a specific entity (e.g., drone, robot dog)."""
    def __init__(self, target):
        self.target = target  # Identifier for the entity (e.g., "DRONE", "ROBOT_DOG")
        self.phases = []

    def add_phase(self, phase):
        """Adds a phase to the plan."""
        self.phases.append(phase)

    def to_dict(self):
        """Converts the plan into a dictionary format."""
        return {
            "target": self.target,
            "phases": [phase.to_dict() for phase in self.phases]
        }


import re

def parse_generated_plan(llm, generated_text, target):
    """Converts LLM-generated structured text into structured Plan and PlanPhase objects."""

    parsing_prompt = f"""
    The following text describes a mission plan for a {target} in an unstructured format:

    {generated_text}

    **Instructions:**
    - Reformat the plan in a structured way using the following template:
    
    ```
    Target: {target}

    Phase 1:
    - State: <state before execution>
    - Target: <goal of this phase>
    - Inputs: [input1, input2, ...]
    - Outputs: [output1, output2, ...]

    Phase 2:
    - State: <state before execution>
    - Target: <goal of this phase>
    - Inputs: [input1, input2, ...]
    - Outputs: [output1, output2, ...]
    ```

    **Rules:**
    - **Strictly follow the format above.** No explanations.
    - **Only return the plan text** (do NOT use markdown formatting like ```).
    - Ensure **inputs and outputs are written as comma-separated lists inside square brackets.**
    """

    response = llm_invoke_with_retry(llm, [SystemMessage(content="You are a helpful assistant"), HumanMessage(content=parsing_prompt)])

    if not response:
        raise ValueError("Error: LLM returned an empty response.")

    structured_text = response.content.strip()

    # Extract the target
    target_match = re.search(r'Target:\s*(\w+)', structured_text)
    extracted_target = target_match.group(1).strip() if target_match else target

    # Extract phases
    phase_pattern = re.findall(
        r'Phase (\d+):\s*'
        r'- State:\s*(.*?)\s*'
        r'- Target:\s*(.*?)\s*'
        r'- Inputs:\s*\[(.*?)\]\s*'
        r'- Outputs:\s*\[(.*?)\]',
        structured_text, re.DOTALL
    )

    plan = Plan(extracted_target)
    def parse_variables(variable_text):
        """Extracts variable names and types from text like 'X <float>, Y <int>' into a dictionary."""
        variable_dict = {}
        variable_matches = re.findall(r'(\w+)\s*<(\w+)>', variable_text)
        for var_name, var_type in variable_matches:
            variable_dict[var_name.strip()] = {"type": var_type.strip()}
        return variable_dict

    for phase_num, state, phase_target, inputs, outputs in phase_pattern:
        inputs_dict = parse_variables(inputs)
        outputs_dict = parse_variables(outputs)
        plan.add_phase(PlanPhase(
            extracted_target, 
            int(phase_num.strip()), 
            state.strip(), 
            phase_target.strip(), 
            inputs_dict, 
            outputs_dict
        ))

    return plan

def generate_plan(llm, mission_text, target):
    """Generates an initial plan using the LLM with enhanced clarity and constraints."""
    prompt = f"""
    **Mission Overview:**
    {mission_text}

    **Your Task:**  
    Generate a structured, step-by-step mission plan specifically for the **{target}**.

    **Plan Structure:**
    - Each phase must describe an independent **mission objective**.
    - The **drone** is responsible for aerial scanning and providing location data.
    - The **robot dog** handles ground operations, jumping, and object interaction.
    - The **TurtleBot3** is a ground robot that navigates, scans with LiDAR, detects objects with camera, and retrieves objects. It cannot fly or jump.
    - The **output of each phase** must serve as the **input for the next relevant phase**.
    - Clearly specify **inputs and outputs** for each phase.

    **Key Rules:**

    1. **Thorough State Descriptions**
       - Clearly describe the **robot's current state** before each phase.
       - Provide relevant **sensor data, positional information, or active tasks** in the state.
       - Avoid vague descriptions like "waiting" or "processing"; be specific.
       - Summarize the main mission and progress so far in the state.
       - Make state description around 70 words for each phase.

    2. **Phase Independence**
       - Each phase must be fully independent and self-contained.
       - If a phase requires an input (e.g., a coordinate from a previous phase), explicitly state it:
         - `"Given a coordinate (X, Y), perform the following action..."`

    3. **MANDATORY Coordinate Variable Rules — YOU MUST FOLLOW THESE EXACTLY:**
       - Every variable MUST be either a float or a string. No arrays, no dicts.
       - **RULE A — DRONE DETECTION:** Any drone phase that detects or locates an object/position MUST have:
         `Outputs: [X <float>, Y <float>]`
         Never leave outputs empty if the drone found a location.
       - **RULE B — GROUND ROBOT NAVIGATION:** Any phase where ROBOT_DOG or TURTLEBOT3 navigates to a location received from another robot MUST have:
         `Inputs: [X <float>, Y <float>]`
         Never leave inputs empty if the robot is moving to a coordinate.
       - **RULE C — VARIABLE NAMES MUST MATCH:** The output variable name from one robot's phase MUST exactly match the input variable name in the receiving robot's phase. Example: if drone outputs `X <float>, Y <float>` then the ground robot inputs must also be `X <float>, Y <float>`.
       - **RULE D — NO EMPTY OUTPUTS:** A phase that discovers, detects, or computes any value used by another phase MUST declare it as an output. `Outputs: []` is FORBIDDEN for such phases.

    **Example — Drone locates, TurtleBot3 navigates:**
    ```
    Drone Phase 1:
    - State: Mission to locate target. Drone is at 10m altitude scanning with camera.
    - Target: Scan area and identify target coordinates.
    - Inputs: []
    - Outputs: [X <float>, Y <float>]

    TurtleBot3 Phase 1:
    - State: Waiting for drone to finish scanning. TurtleBot3 is at base station.
    - Target: Wait for drone signal then navigate to received coordinates (X, Y).
    - Inputs: [X <float>, Y <float>]
    - Outputs: [arrival_status <string>]

    TurtleBot3 Phase 2:
    - State: TurtleBot3 has arrived at target coordinates. Ready to inspect.
    - Target: Perform close-up inspection and report findings.
    - Inputs: []
    - Outputs: [inspection_result <string>]
    ```

    **Example — Drone locates two targets, Dog gets one, TurtleBot3 gets the other:**
    ```
    Drone Phase 1:
    - Outputs: [X1 <float>, Y1 <float>]

    Drone Phase 2:
    - Outputs: [X2 <float>, Y2 <float>]

    Robot Dog Phase 1:
    - Inputs: [X1 <float>, Y1 <float>]

    TurtleBot3 Phase 1:
    - Inputs: [X2 <float>, Y2 <float>]
    ```

    **Final Constraints:**
    - **Strictly follow the given format. Return only structured text, no explanations.**
    - **ALWAYS declare X <float> and Y <float> as outputs when a robot discovers a position.**
    - **ALWAYS declare X <float> and Y <float> as inputs when a robot navigates to a received position.**
    - **Variable names in outputs of one phase MUST match inputs of the dependent phase.**
    - **If the plan targets only a subset of robots, do not include plans for unused robots.**
    - **The TurtleBot3 is a ground robot — do NOT assign it aerial tasks. It uses LiDAR for area scanning and camera for object detection.**
    - **Outputs: [] is only allowed for phases that produce NO data needed by other phases.**
    """
    

    response = llm_invoke_with_retry(llm, [SystemMessage(content="You are a helpful assistant"), HumanMessage(content=prompt)])
    print(type(response))
    return parse_generated_plan(llm,response, target)


def refine_plan(llm, mission_text, plan):
    """Refines the plan iteratively until it meets rubric criteria with a rating of 10."""
    rating = 0
    count = 0
    final_plan = plan

    prompt_template = PromptTemplate(
        input_variables=["mission", "plan"],
        template="""
        We had the following mission:
        {mission}

        And we used an LLM to generate the following high-level plan:
        {plan}

        Evaluate this plan based on the following rubric:

        - **Logical Soundness (1-10):** Is the plan coherent and free of contradictions?
        - **Feasibility (1-10):** Can the robots execute the tasks given their capabilities?
        - **Completeness (1-10):** Does the plan cover all mission objectives?
        - **Data Flow Integrity (1-10):** Do the outputs from one phase correctly match the inputs of the next?

        Provide a structured response in the same format, ensuring proper input-output matching.

        - If the plan is already perfect, assign it a rating of **10** and return the same plan.
        - Do **not** provide explanations—just return the structured output.
        """
    )

    while count < 3:
        response = llm_invoke_with_retry(llm, [HumanMessage(content=prompt_template.format(mission=mission_text, plan=str(final_plan)))])
        new_rating, modified_plan_text = extract_rating_and_plan(response.content)

        if new_rating is None or new_rating <= rating or new_rating == 10:
            break

        rating = new_rating
        final_plan = parse_generated_plan(modified_plan_text, final_plan.target)
        count += 1

    return final_plan


def main():
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # Load the mission text file from the config, default to "mission_files/mission_scenario.txt" if not provided
    mission_text_file = config.get("mission_text_file", "mission_files/mission_scenario.txt")

    print("Generating structured mission plan...")

    mission_text = read_file(mission_text_file)
    if not mission_text:
        print("Error: Mission scenario file is empty or missing.")
        return

    # ---- COMMENTED OUT: OpenAI API (original) ----
    # OPENAI_API_KEY = config.get("openai_api_key", "")
    # llm = ChatOpenAI(
    #     model_name="gpt-4o",
    #     openai_api_key=OPENAI_API_KEY,
    #     temperature=0.1
    # )
    # ---- END COMMENTED OUT: OpenAI API ----

    # ---- ACTIVE: LLaMA via SambaNova (free) ----
    LLAMA_API_KEY = config.get("llama_api_key", "")
    llm = ChatOpenAI(
        api_key=LLAMA_API_KEY,
        base_url="https://api.sambanova.ai/v1",
        model_name="Meta-Llama-3.3-70B-Instruct",
        temperature=0.1
    )

    mission_output = {}
    for robot in config["robots_in_curr_mission"]:
        plan = generate_plan(llm, mission_text, robot)
        mission_output[ config["robots_config"][robot]["high_plan_key"] ] = plan.to_dict()

    mission_plan_file = config.get("mission_plan_file", "Plans/mission_plan.json")
    with open(mission_plan_file, 'w') as output_file:
        json.dump(mission_output, output_file, indent=4)

    print("Mission Plan written to", mission_plan_file)

if __name__ == "__main__":
    main()
