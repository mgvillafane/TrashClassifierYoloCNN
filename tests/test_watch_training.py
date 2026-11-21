import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import watch_taco_training as viewer


class TensorBoardViewerTests(unittest.TestCase):
    def test_startup_then_later_epochs_are_visible_to_the_same_reader(self):
        from shutil import copyfile
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from torch.utils.tensorboard import SummaryWriter

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            startup, first, second, logs = [root / name for name in ("startup", "first", "second", "logs")]
            logs.mkdir()
            with SummaryWriter(str(startup)) as writer:
                writer.add_text("configuration", "SortWaste", 0)
            copyfile(next(startup.glob("events.out.tfevents.*")), logs / "events.out.tfevents.startup")
            viewer.normalize_startup_log(logs)
            reader = EventAccumulator(str(logs)).Reload()
            for step, output in ((1, first), (2, second)):
                with SummaryWriter(str(output)) as writer:
                    writer.add_scalar("train/box_loss", 1 / step, step)
                copyfile(next(output.glob("events.out.tfevents.*")), logs / f"events.out.tfevents.epoch{step:06d}")
                reader.Reload()
                self.assertEqual([event.step for event in reader.Scalars("train/box_loss")], list(range(1, step + 1)))
            self.assertFalse((logs / "events.out.tfevents.startup").exists())
            self.assertLess(viewer.tensorboard_filename("events.out.tfevents.startup"), "events.out.tfevents.epoch000001")

    def test_logs_only_syncs_files_without_querying_job_status(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory)
            (local / "training_status.json").write_text(
                json.dumps({"training_finished": True}), encoding="utf-8")
            response = MagicMock()
            response.__enter__.return_value.status = 200
            process = SimpleNamespace(pid=123, poll=lambda: None)
            with patch.object(viewer, "find_cli", return_value="databricks"), \
                    patch.object(viewer.socket, "socket"), \
                    patch.object(viewer.subprocess, "Popen", return_value=process), \
                    patch.object(viewer.subprocess, "run", return_value=SimpleNamespace(
                        returncode=0, stdout="[]", stderr="")) as command, \
                    patch.object(viewer.urllib.request, "urlopen", return_value=response):
                viewer.main([
                    "--logs-only", "--remote-run", "/Volumes/workspace/default/data/run",
                    "--local-dir", str(local),
                ])
            self.assertEqual(len(command.call_args_list), 1)
            self.assertEqual(command.call_args.args[0][1:3], ["fs", "ls"])
            self.assertFalse((local / "databricks_run.json").exists())


if __name__ == "__main__":
    unittest.main()
