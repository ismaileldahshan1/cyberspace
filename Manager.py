import os
import json
import asyncio
import time
import warnings
warnings.filterwarnings("ignore")

async def run_python_script(command, retries=3, delay=15):
    for attempt in range(1, retries + 1):
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            output = stdout.decode() if stdout else ""
            errors = stderr.decode() if stderr else ""
            if stdout:
                print(output)
            if "RateLimitError" in errors or "rate_limit_exceeded" in errors:
                print(f"[Manager] Rate limit hit on attempt {attempt}/{retries}. Waiting {delay}s...")
                await asyncio.sleep(delay)
                continue
            if stderr:
                print(errors)
            return
        except Exception as e:
            print(f"Error running command '{command}': {e}")
            return
    print(f"[Manager] Command failed after {retries} attempts: {command}")

def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

def main():
    config = load_config()

    print("=== Starting High-Level Planning ===")
    # Run High-Level Planner (now uses config.json internally)
    high_level_command = "python3 High_Level_Plan_Generator.py"
    asyncio.run(run_python_script(high_level_command))

    for robot in config["robots_in_curr_mission"]:
        print(f"=== Starting Low-Level Planning for {robot} ===")
        robot_ll_command = f"python3 Low_Level_Planner.py {robot}"
        asyncio.run(run_python_script(robot_ll_command))
        print(f"[Manager] Cooling down 10s before next robot...")
        time.sleep(10)

    print("=== All Planning Stages Completed ===")

if __name__ == "__main__":
    main()
