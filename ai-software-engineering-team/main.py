import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Ensure the project folder is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.state import create_initial_state
from app.graph.workflow import create_graph
from app.config import WORKSPACE_ROOT, MODEL_NAME

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

def main():
    print("=" * 50)
    print("       AI SOFTWARE ENGINEERING TEAM - PHASE 1")
    print("=" * 50)
    
    # Prompt user for inputs or read from sys.argv
    if len(sys.argv) > 1:
        user_request = sys.argv[1]
    else:
        user_request = input("Enter your goal: ").strip()
        if not user_request:
            user_request = "Fix the multiply bug in calculator.py and verify tests pass"
            
    repo_path = str(WORKSPACE_ROOT)
    print(f"Target Repository: {repo_path}")
    print(f"Goal: {user_request}")
    print(f"Gemini Model: {MODEL_NAME}")
    print("-" * 50)
    
    # Build the compiled state graph
    graph = create_graph()
    
    # Create initial state
    initial_state = create_initial_state(user_request, repo_path)
    
    # Run the graph
    print("Initializing multi-agent graph execution...")
    try:
        events = graph.stream(initial_state)
        for event in events:
            for node_name, state_update in event.items():
                print(f"\n>>> Node [{node_name}] Executed.")
                # Show key transitions
                if "next_agent" in state_update:
                    print(f"    Next Routing Decision: {state_update['next_agent']}")
                if "current_task" in state_update:
                    print(f"    Current Active Task: {state_update['current_task']}")
                if "completed_tasks" in state_update:
                    print(f"    Completed Tasks: {state_update['completed_tasks']}")
                if "research_results" in state_update:
                    print(f"    Research Result: {state_update['research_results']}")
                if "code_changes" in state_update:
                    print(f"    Code Changes: {state_update['code_changes']}")
                if "test_results" in state_update:
                    print(f"    Test Results: {state_update['test_results']}")
                if "review_results" in state_update:
                    print(f"    Review Results: {state_update['review_results']}")
        print("\n" + "=" * 50)
        print("Graph run successfully completed.")
        print("=" * 50)
        
        from app.utils.logging import tracer
        print("\n" + tracer.generate_report())
    except Exception as e:
        logger.exception("An error occurred during graph execution:")
        sys.exit(1)

if __name__ == "__main__":
    main()
