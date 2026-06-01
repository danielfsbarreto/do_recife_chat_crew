#!/usr/bin/env python
import sys
import warnings

from do_recife_chat_crew.crew import DoRecifeChatCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Target MongoDB collection populated by ../do_recife_embedder.
# Use "do-recife-rag-enriched" (with metadata) or "do-recife-rag" (plain).
# Passed as a crew input; the before_kickoff hook points the tool at it.
COLLECTIONS = {"plain": "do-recife-rag", "enriched": "do-recife-rag-enriched"}


def run():
    """
    Run the crew.
    """
    inputs = {
        "question": "Quais decretos foram publicados na edição mais recente do Diário Oficial do Recife?",
        "collection_name": COLLECTIONS["enriched"],
    }

    try:
        DoRecifeChatCrew().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "question": "Quais decretos foram publicados na edição mais recente do Diário Oficial do Recife?",
        "collection_name": COLLECTIONS["enriched"],
    }
    try:
        DoRecifeChatCrew().crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        DoRecifeChatCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "question": "Quais decretos foram publicados na edição mais recente do Diário Oficial do Recife?",
        "collection_name": COLLECTIONS["enriched"],
    }

    try:
        DoRecifeChatCrew().crew().test(
            n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception(
            "No trigger payload provided. Please provide JSON payload as argument."
        )

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "question": "",
        "collection_name": COLLECTIONS["enriched"],
    }

    try:
        result = DoRecifeChatCrew().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
