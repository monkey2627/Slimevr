import json
import io
import unittest
from pathlib import Path

import numpy as np

from mobileposer.realtime.recording import FORMAT, FORMAT_VERSION, RealtimeRecorder, load_recording


class RecordingTests(unittest.TestCase):
    def test_writer_serializes_arrays(self):
        recorder = RealtimeRecorder.__new__(RealtimeRecorder)
        recorder._file = io.StringIO()
        recorder.write("heading", {"matrices": np.eye(3, dtype=np.float32)[None]})
        recorder.write("frame", {
            "frame": 0,
            "feature": np.arange(60, dtype=np.float32),
            "smplLocalRotationMatrices": np.tile(np.eye(3, dtype=np.float32), (24, 1, 1)),
        })
        rows = [json.loads(line) for line in recorder._file.getvalue().splitlines()]
        self.assertEqual([row["type"] for row in rows], ["heading", "frame"])
        self.assertEqual(rows[1]["feature"], list(map(float, range(60))))

    def test_fixture_loads(self):
        path = Path(__file__).parent / "fixtures" / "realtime_recording.jsonl"
        metadata, rows = load_recording(path)
        self.assertEqual(metadata["format"], FORMAT)
        self.assertEqual(metadata["formatVersion"], FORMAT_VERSION)
        self.assertEqual(
            metadata["timestampSource"],
            "python_host_receive_and_process_time_not_sensor_time",
        )
        self.assertEqual([row["type"] for row in rows], ["heading", "frame"])
        self.assertEqual(rows[1]["feature"], list(map(float, range(60))))
