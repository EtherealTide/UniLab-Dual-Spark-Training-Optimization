"""Exercise the real Bash launcher with local SSH and training substitutes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

FAKE_SSH = r'''#!/usr/bin/env python3
import json
import os
import subprocess
import sys

args = sys.argv[1:]
no_stdin = False
while args and args[0].startswith("-"):
    flag = args.pop(0)
    if flag == "-o":
        args.pop(0)
    elif flag == "-n":
        no_stdin = True
    else:
        raise SystemExit(f"Unexpected SSH option: {flag}")
host, command = args
with open(os.environ["SSH_RECORD"], "a") as record:
    record.write(json.dumps([host, command]) + "\n")
if host == os.environ.get("FAKE_SSH_FAIL_HOST"):
    print("Permission denied (publickey,password)", file=sys.stderr)
    raise SystemExit(255)
raise SystemExit(subprocess.call(["bash", "-c", command],
    stdin=subprocess.DEVNULL if no_stdin else None))
'''

FAKE_PYTHON = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
import time

if sys.argv[1] == "-":
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
settings = dict(arg.split("=", 1) for arg in sys.argv[1:] if "=" in arg)
rank = settings.get("--node_rank", os.environ.get("UNILAB_DP_RANK", "0"))
algo = "flashsac" if any("train_flashsac.py" in a for a in sys.argv) else (
    "sac" if any("train_sac.py" in a for a in sys.argv) else "ppo")
assert os.environ.get("PYTHONUNBUFFERED") == "1"
assert settings["++training.log_interval"] == "1"
print(f"mock start rank={rank}", flush=True)
print("Actor Model: mock model details", flush=True)
print(f"mock stderr rank={rank}", file=sys.stderr, flush=True)
print("Learning iteration 0/2", flush=True)
time.sleep(float(os.environ.get("FAKE_DELAY", "0.05")))
if rank == os.environ.get("FAKE_FAIL_RANK"):
    print("mock training failure", file=sys.stderr)
    raise SystemExit(7)
run_dir = Path(settings["training.log_dir"])
run_dir.mkdir(parents=True, exist_ok=True)
if rank == "0" and not os.environ.get("FAKE_MISSING_SUMMARY"):
    iterations = int(settings["algo.max_iterations"])
    summary = {
        "status": "completed", "algo": algo, "task": "MockTask",
        "completed_iterations": int(os.environ.get("FAKE_SUMMARY_ITERATIONS",
            str(iterations - 1 if algo == "ppo" else iterations))),
        "training_wall_time_sec": 1.25, "wall_time_sec": 2.0,
        "run_env_steps": 1000, "total_env_steps": 1000,
        "training_throughput_env_steps_per_sec": 800.0,
        "final_env_steps_per_sec": 900.0,
        "final_learner_replay_rows_per_sec": 12345.0,
        "final_cycle_wall_ms": 2.0,
        "final_mean_reward": None, "mean_episode_length": 42.0,
        "last_checkpoint": str(run_dir / "model.pt"),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary))
print("Learning iteration 1/2", flush=True)
print(f"mock finished rank={rank}", flush=True)
'''


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux/Bash required")
class LauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.bin = self.directory / "bin"
        self.bin.mkdir()
        self.runtime = self.directory / "runtime with spaces" / "UniLab"
        python = self.runtime / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text(FAKE_PYTHON)
        python.chmod(0o755)
        ssh = self.bin / "ssh"
        ssh.write_text(FAKE_SSH)
        ssh.chmod(0o755)
        self.config = self.directory / "cluster.env"
        self.config.write_text(
            "HOST0=master-alias\nHOST1=worker-alias\nMASTER_ADDR=10.77.0.1\n"
            "NCCL_SOCKET_IFNAME=test0\nNCCL_IB_HCA=test_hca\n"
            f"REMOTE_ROOT='{self.runtime.parent}'\nMAX_ITERATIONS=500\n"
        )
        self.record = self.directory / "ssh.jsonl"
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                        CLUSTER_FILE=str(self.config), MAX_ITERATIONS="2",
                        SSH_RECORD=str(self.record))
        self.run_name = f"launcher_test_{os.getpid()}_{self.directory.name}"
        self.addCleanup(self.cleanup_logs)

    def cleanup_logs(self) -> None:
        for suffix in ("single", "rank0", "rank1"):
            Path(f"/tmp/{self.run_name}_{suffix}.log").unlink(missing_ok=True)

    def command(self, mode: str = "single", algo: str = "ppo") -> list[str]:
        return ["bash", str(ROOT / "scripts/run_one.sh"), mode, algo,
                "go2_joystick_flat", "64", self.run_name, "29705"]

    def run_launcher(self, mode: str = "single", algo: str = "ppo", **env: str):
        return subprocess.run(self.command(mode, algo), env={**self.env, **env},
                              text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=15)

    def test_single_streams_before_exit_and_prints_statistics(self) -> None:
        process = subprocess.Popen(self.command(), env={**self.env, "FAKE_DELAY": "0.5", "CONSOLE_MODE": "full"},
                                   text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        try:
            lines = []
            saw_live_output = False
            for line in process.stdout:
                lines.append(line)
                if "[single] mock start" in line:
                    self.assertIsNone(process.poll(), "Output was buffered until exit")
                    saw_live_output = True
            self.assertTrue(saw_live_output, "No live training output")
            # Keep reading through the same TextIOWrapper: communicate() after
            # partial reads can skip bytes already prefetched by that wrapper.
            process.wait(timeout=15)
            output = "".join(lines)
            self.assertEqual(process.returncode, 0, output)
            self.assertIn("[single] mock stderr", output)
            self.assertIn("Completed iterations: 2 / 2", output)
            self.assertIn("Global throughput, full training (steps/s): 800.000", output)
            self.assertIn("Final mean reward: N/A", output)
            self.assertIn(f"DONE:{self.run_name}:0", output)
            saved = Path(f"/tmp/{self.run_name}_single.log").read_text()
            self.assertIn("mock start", saved)
            self.assertIn("mock stderr", saved)
            self.assertIn("[launcher] single exited (code=0)", saved)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()

    def test_dual_labels_and_summary_for_all_algorithms(self) -> None:
        for algo in ("ppo", "sac", "flashsac"):
            with self.subTest(algo=algo):
                self.run_name += f"_{algo}"
                result = self.run_launcher("dual", algo, CONSOLE_MODE="full")
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn("[rank0] mock start rank=0", result.stdout)
                self.assertIn("[rank1] mock start rank=1", result.stdout)
                self.assertIn("Completed iterations: 2 / 2", result.stdout)
                self.assertIn("Environments per rank / global: 64 / 128", result.stdout)
                if algo != "ppo":
                    self.assertIn("Global learner throughput, final iteration (rows/s): 12,345.000", result.stdout)
                self.cleanup_logs()

    def test_compact_mode_hides_details_but_preserves_log_and_stats(self) -> None:
        result = self.run_launcher(CONSOLE_MODE="compact")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("[single] Iteration 1/2", result.stdout)
        self.assertIn("[single] Iteration 2/2", result.stdout)
        self.assertIn("Full live log from another coordinator terminal: ssh master-alias", result.stdout)
        self.assertIn("tail", result.stdout)
        self.assertIn("=== Final training statistics (rank0) ===", result.stdout)
        self.assertIn("[launcher] single exited (code=0)", result.stdout)
        self.assertNotIn("mock model details", result.stdout)
        self.assertNotIn("mock stderr", result.stdout)
        saved = Path(f"/tmp/{self.run_name}_single.log").read_text()
        self.assertIn("mock model details", saved)
        self.assertIn("mock stderr", saved)

    def test_compact_failure_shows_log_tail(self) -> None:
        result = self.run_launcher(CONSOLE_MODE="compact", FAKE_FAIL_RANK="0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Last 20 lines", result.stdout)
        self.assertIn("mock training failure", result.stdout)
        self.assertIn("[launcher] single exited (code=7)", result.stdout)
        self.assertNotIn("DONE:", result.stdout)

    def test_compact_dual_writes_rank1_completion_marker(self) -> None:
        result = self.run_launcher("dual", CONSOLE_MODE="compact")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("mock model details", result.stdout)
        self.assertIn("[rank1] [launcher] rank1 exited (code=0)", result.stdout)
        self.assertIn("[launcher] rank1 exited (code=0)",
                      Path(f"/tmp/{self.run_name}_rank1.log").read_text())

    def test_compact_filter_throttles_progress_and_keeps_final_iteration(self) -> None:
        log = "\n".join(f"Learning iteration {i}/100" for i in range(100)) + "\n"
        result = subprocess.run(
            ["awk", "-v", "label=rank0", "-v", "mode=compact", "-v", "interval=3600",
             "-f", str(ROOT / "scripts/filter_console.awk")],
            input=log, text=True, capture_output=True, check=True,
        )
        self.assertEqual(result.stdout.splitlines(),
                         ["[rank0] Iteration 1/100", "[rank0] Iteration 100/100"])

    def test_compact_filter_handles_offpolicy_progress(self) -> None:
        log = "│ Iterations: 1/500 │\n│ Iterations: 499/500 │\n│ Iterations: 500/500 │\n"
        result = subprocess.run(
            ["awk", "-v", "label=rank0", "-v", "mode=compact", "-v", "interval=3600",
             "-f", str(ROOT / "scripts/filter_console.awk")],
            input=log, text=True, capture_output=True, check=True,
        )
        self.assertEqual(result.stdout.splitlines(),
                         ["[rank0] Iteration 1/500", "[rank0] Iteration 500/500"])

    def test_training_failure_is_not_hidden_by_tee(self) -> None:
        for mode in ("single", "dual"):
            with self.subTest(mode=mode):
                result = self.run_launcher(mode, FAKE_FAIL_RANK="0")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("rank0=7", result.stdout)
                self.assertNotIn("DONE:", result.stdout)
                shutil.rmtree(self.runtime / "logs", ignore_errors=True)

    def test_rank1_failure_is_not_hidden_by_wait(self) -> None:
        result = self.run_launcher("dual", FAKE_FAIL_RANK="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rank1=7", result.stdout)
        self.assertNotIn("DONE:", result.stdout)

    def test_ssh_preflight_fails_before_either_rank_starts(self) -> None:
        result = self.run_launcher("dual", FAKE_SSH_FAIL_HOST="worker-alias")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Preflight failed on worker-alias", result.stdout)
        self.assertNotIn("mock start", result.stdout)

    def test_existing_run_is_rejected(self) -> None:
        (self.runtime / "logs" / self.run_name).mkdir(parents=True)
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Run directory already exists", result.stdout)
        self.assertNotIn("mock start", result.stdout)

    def test_missing_summary_is_not_success(self) -> None:
        result = self.run_launcher(FAKE_MISSING_SUMMARY="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cannot read final summary", result.stdout)
        self.assertNotIn("DONE:", result.stdout)

    def test_incomplete_summary_is_not_success(self) -> None:
        result = self.run_launcher(FAKE_SUMMARY_ITERATIONS="0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not confirm all requested iterations", result.stdout)
        self.assertNotIn("DONE:", result.stdout)


if __name__ == "__main__":
    unittest.main()
