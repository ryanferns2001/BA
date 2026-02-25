from blackboard.evaluation_methods.methods import candidate_test, post_processing_stuff, final_evalutions_signals
import logging

logger = logging.getLogger(__name__)

# Set this at the top of the file
EVAL_MODE = "candidate_only"   # change to "full" when needed

def run_evaluations(export_dir=""):
    if not export_dir:
        return

    # Always run candidate test (this is what your boss needs)
    try:
        candidate_test.run(export_dir)
    except Exception as e:
        logger.info(f"Candidate test failed: {e}")

    # Stop here if we only evaluate candidate generator
    if EVAL_MODE == "candidate_only":
        return

    # Full pipeline evaluation (only if needed)
    try:
        post_processing_stuff.run(export_dir)
    except Exception as e:
        logger.info(f"More global test failed (none specified): {e}")

    try:
        final_evalutions_signals.run(export_dir)
    except Exception as e:
        logger.info(f"Final evaluation test failed (none specified): {e}")
