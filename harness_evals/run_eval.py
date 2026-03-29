#!/usr/bin/env python3
"""
Harness eval runner for drone-rl-lab.

Runs eval cases against the harness logic to verify correct behavior.
Each case specifies a fixture (synthetic queue state) and expected outcomes.

Usage:
    python harness_evals/run_eval.py                     # run all cases
    python harness_evals/run_eval.py --case choose_next   # run matching cases
    python harness_evals/run_eval.py --tag queue-selection # run by tag
    python harness_evals/run_eval.py --verbose            # show details
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import shutil
from pathlib import Path

# Add scripts/ to path so we can import harness modules
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "scripts"))

from task_store import TaskStore
from job_store import JobStore

CASES_DIR = Path(__file__).resolve().parent / "cases"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def load_fixture(fixture_name: str) -> dict:
    path = FIXTURES_DIR / fixture_name
    return json.loads(path.read_text())


def load_cases(case_filter: str | None = None, tag_filter: str | None = None) -> list[dict]:
    cases = []
    for f in sorted(CASES_DIR.glob("*.json")):
        case = json.loads(f.read_text())
        if case_filter and case_filter not in case.get("case_id", ""):
            continue
        if tag_filter and tag_filter not in case.get("tags", []):
            continue
        cases.append(case)
    return cases


def setup_temp_store(fixture: dict) -> tuple[TaskStore, JobStore, Path]:
    """Create temporary task and job stores populated from fixture data."""
    tmpdir = Path(tempfile.mkdtemp(prefix="harness_eval_"))
    tasks_dir = tmpdir / "tasks"
    tasks_dir.mkdir()
    jobs_dir = tmpdir / "jobs"
    jobs_dir.mkdir()

    store = TaskStore(tasks_dir=tasks_dir)
    for task in fixture.get("tasks", []):
        store.save(task)

    job_store = JobStore(jobs_dir=jobs_dir)
    for job in fixture.get("job_history", []):
        job_path = jobs_dir / f"{job['job_id']}.json"
        job_path.write_text(json.dumps(job, indent=2) + "\n")

    return store, job_store, tmpdir


# ─── Evaluators ────────────────────────────────────────────────────────────

def eval_next_task_choice(case: dict, fixture: dict, verbose: bool) -> tuple[bool, str]:
    """Evaluate: does get_next() return the expected task?"""
    store, _job_store, tmpdir = setup_temp_store(fixture)
    try:
        expected_id = case["expected"]["chosen_task_id"]
        result = store.get_next(check_failures=False)
        if result is None:
            return False, f"get_next() returned None, expected {expected_id}"
        actual_id = result.get("task_id")
        if actual_id == expected_id:
            return True, f"correctly chose {expected_id}"
        return False, f"chose {actual_id}, expected {expected_id}"
    finally:
        shutil.rmtree(tmpdir)


def eval_stale_claim_recovery(case: dict, fixture: dict, verbose: bool) -> tuple[bool, str]:
    """Evaluate: can we reclaim a stale task and get it back to ready?"""
    store, _job_store, tmpdir = setup_temp_store(fixture)
    try:
        reclaim_id = case["expected"]["reclaimed_task_id"]
        task = store.load(reclaim_id)
        if task is None:
            return False, f"task {reclaim_id} not found in fixture"
        if task["status"] != "claimed":
            return False, f"task {reclaim_id} is {task['status']}, not claimed"

        # Simulate reclaim
        released = store.release(reclaim_id)
        if released is None:
            return False, f"release({reclaim_id}) returned None"

        task_after = store.load(reclaim_id)
        if task_after["status"] != case["expected"]["post_reclaim_status"]:
            return False, f"post-reclaim status is {task_after['status']}, expected {case['expected']['post_reclaim_status']}"
        if task_after.get("claimed_by") != case["expected"]["post_reclaim_claimed_by"]:
            return False, f"post-reclaim claimed_by is {task_after.get('claimed_by')}, expected {case['expected']['post_reclaim_claimed_by']}"

        nxt = store.get_next(check_failures=False)
        expected_next = case["expected"]["next_actionable_after_reclaim"]
        if nxt is None or nxt.get("task_id") != expected_next:
            actual = nxt.get("task_id") if nxt else None
            return False, f"next after reclaim is {actual}, expected {expected_next}"

        return True, f"reclaimed {reclaim_id}, next is {expected_next}"
    finally:
        shutil.rmtree(tmpdir)


def eval_repeated_failure_guard(case: dict, fixture: dict, verbose: bool) -> tuple[bool, str]:
    """Evaluate: does get_next(check_failures=True) skip repeatedly-failed tasks?"""
    store, job_store, tmpdir = setup_temp_store(fixture)
    try:
        skip_id = case["expected"]["should_skip_task_id"]
        job_history = fixture.get("job_history", [])
        failures = [j for j in job_history if j.get("task_id") == skip_id and j.get("status") == "failed"]

        if len(failures) < 2:
            return False, f"fixture should have 2+ failures for {skip_id}, found {len(failures)}"

        task = store.load(skip_id)
        if task is None:
            return False, f"task {skip_id} not in store"

        # Verify job_store detects the failures
        if not job_store.has_repeated_failures(skip_id):
            return False, f"job_store did not detect repeated failures for {skip_id}"

        # Monkey-patch task_store to use our temp job_store
        import task_store as ts_mod
        original_import = None
        import importlib
        import sys

        # Temporarily make job_store point to our temp dir
        class PatchedJobStore(JobStore):
            def __init__(self, jobs_dir=None):
                super().__init__(jobs_dir=job_store.jobs_dir)

        old_class = sys.modules.get("job_store")
        if old_class:
            old_jobstore_class = old_class.JobStore
            old_class.JobStore = PatchedJobStore

        try:
            nxt = store.get_next(check_failures=True)
        finally:
            if old_class:
                old_class.JobStore = old_jobstore_class

        if nxt is None:
            return True, f"correctly returned None (all ready tasks blocked by failures)"
        if nxt["task_id"] == skip_id:
            return False, f"get_next() still returned {skip_id} despite repeated failures"
        return True, f"correctly skipped {skip_id}, chose {nxt['task_id']}"
    finally:
        shutil.rmtree(tmpdir)


EVALUATORS = {
    "next_task_choice": eval_next_task_choice,
    "stale_claim_recovery": eval_stale_claim_recovery,
    "repeated_failure_guard": eval_repeated_failure_guard,
}


# ─── Runner ────────────────────────────────────────────────────────────────

def run_case(case: dict, verbose: bool) -> tuple[bool, str]:
    """Run a single eval case."""
    fixture = load_fixture(case["fixture"])
    evaluator_name = case.get("evaluator")
    evaluator = EVALUATORS.get(evaluator_name)
    if not evaluator:
        return False, f"unknown evaluator: {evaluator_name}"
    return evaluator(case, fixture, verbose)


def main():
    parser = argparse.ArgumentParser(description="Run harness eval cases")
    parser.add_argument("--case", default=None, help="Filter cases by ID substring")
    parser.add_argument("--tag", default=None, help="Filter cases by tag")
    parser.add_argument("--verbose", action="store_true", help="Show details")
    args = parser.parse_args()

    cases = load_cases(case_filter=args.case, tag_filter=args.tag)
    if not cases:
        print("No eval cases found matching filters.")
        sys.exit(1)

    passed = 0
    failed = 0
    results = []

    for case in cases:
        case_id = case.get("case_id", "unknown")
        ok, detail = run_case(case, args.verbose)
        status = "PASS" if ok else "FAIL"
        results.append((case_id, status, detail))
        if ok:
            passed += 1
        else:
            failed += 1

    # Print results
    print(f"\n{'='*60}")
    print(f"Harness Eval Results: {passed} passed, {failed} failed, {len(cases)} total")
    print(f"{'='*60}\n")

    for case_id, status, detail in results:
        icon = "+" if status == "PASS" else "X"
        print(f"  [{icon}] {case_id}: {status}")
        if args.verbose or status == "FAIL":
            print(f"      {detail}")

    print()
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
