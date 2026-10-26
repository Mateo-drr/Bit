import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import program
from program import config


class ConfigTests(unittest.TestCase):
    def write_model(self, root: Path, name: str) -> Path:
        model_dir = root / name
        model_dir.mkdir()
        (model_dir / "bit_model.pt").write_bytes(b"")
        (model_dir / "training_meta.json").write_text("{}", encoding="utf-8")
        return model_dir

    def test_download_roots_live_under_program_directory(self):
        self.assertEqual(
            config.MODELS_ROOT,
            Path(program.__file__).resolve().parent / "models",
        )
        self.assertEqual(config.WHISPER_ROOT, config.MODELS_ROOT / "faster-whisper")
        self.assertEqual(config.PIPER_ROOT, config.MODELS_ROOT / "piper")
        self.assertEqual(config.HF_ROOT, config.MODELS_ROOT / "huggingface")

    def test_resolve_model_dir_picks_newest_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_model(root, "bit-20260101_000000")
            newer = self.write_model(root, "bit-20260102_000000")
            self.assertEqual(config.resolve_model_dir(models_root=root), newer)

    def test_resolve_model_dir_honours_name_and_path_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = self.write_model(root, "bit-20260101_000000")
            self.write_model(root, "bit-20260102_000000")
            self.assertEqual(
                config.resolve_model_dir("bit-20260101_000000", models_root=root),
                older,
            )
            self.assertEqual(
                config.resolve_model_dir(model_dir=str(older), models_root=root),
                older,
            )

    def test_resolve_model_dir_reports_missing_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "No trained Bit models"):
                config.resolve_model_dir(models_root=root)
            with self.assertRaisesRegex(FileNotFoundError, "Could not find Bit model"):
                config.resolve_model_dir("nope", models_root=root)

    def test_project_declares_voice_package_and_script(self):
        project_root = Path(__file__).resolve().parents[1]
        data = tomllib.loads(
            (project_root / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.assertIn("program", data["tool"]["setuptools"]["packages"])
        self.assertEqual(
            data["project"]["scripts"]["bit-voice"],
            "program.voice_chat:main",
        )
        self.assertEqual(
            set(data["project"]["optional-dependencies"]["voice"]),
            {
                "sounddevice==0.5.6",
                "faster-whisper==1.2.1",
                "piper-tts==1.8.0",
            },
        )

    def test_voice_program_forces_local_hf_cache(self):
        project_root = Path(__file__).resolve().parents[1]
        env = {key: value for key, value in os.environ.items() if key != "HF_HOME"}
        env["HF_HOME"] = str(project_root / "wrong-cache")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os, program.config; print(os.environ['HF_HOME'])",
            ],
            cwd=project_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), str(config.HF_ROOT))


if __name__ == "__main__":
    unittest.main()
