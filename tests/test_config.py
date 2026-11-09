import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from label_audit.config import load_env


class EnvTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / ".env"
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_loads_quotes_comments_bom_and_literal_values(self):
        self.path.write_text(
            '\ufeff# settings\nexport MODEL = "test model" # comment\n'
            "KEY='secret#part'\nURL=https://example.com/v1 # comment\n"
            "EMPTY=\nLITERAL=${MODEL}$(whoami)\n", encoding="utf-8")
        load_env(self.path)
        self.assertEqual(dict(os.environ), {
            "MODEL": "test model", "KEY": "secret#part",
            "URL": "https://example.com/v1", "EMPTY": "",
            "LITERAL": "${MODEL}$(whoami)",
        })

    def test_existing_environment_including_empty_values_wins(self):
        self.path.write_text("MODEL=file\nKEY=file\n", encoding="utf-8")
        os.environ.update(MODEL="shell", KEY="")
        load_env(self.path)
        self.assertEqual(os.environ["MODEL"], "shell")
        self.assertEqual(os.environ["KEY"], "")

    def test_missing_file_is_optional(self):
        load_env(self.path)
        self.assertEqual(dict(os.environ), {})

    def test_invalid_file_reports_line_without_secret_or_partial_load(self):
        for invalid in ["bad line secret", 'KEY="secret', "BAD-KEY=secret"]:
            with self.subTest(invalid=invalid):
                self.path.write_text("MODEL=test\n" + invalid, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "строке 2") as error:
                    load_env(self.path)
                self.assertNotIn("secret", str(error.exception))
                self.assertEqual(dict(os.environ), {})

    def test_cli_loads_project_env_before_running_command(self):
        from label_audit import cli

        self.path.write_text("LLM_MODEL=from-file\n", encoding="utf-8")

        def evaluate(*args):
            self.assertEqual(os.environ["LLM_MODEL"], "from-file")
            return {}

        with patch("sys.argv", ["label-audit", "evaluate"]), \
                patch.object(cli, "load_env", side_effect=lambda path: load_env(self.path)) as loader, \
                patch.object(cli, "evaluate_project", side_effect=evaluate), \
                patch("builtins.print"):
            cli.main()
        loader.assert_called_once_with(Path(cli.__file__).resolve().parents[1] / ".env")
