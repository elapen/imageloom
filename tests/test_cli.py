"""Start-up and command-line behaviour in real subprocesses: friendly one-line errors instead of tracebacks.
Every process gets a throwaway HOME and data folder; the engine is never present, so sd-cli is never started."""
from __future__ import annotations

import argparse
import http.client
import importlib.machinery
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
SERVER_PY = ROOT / "core" / "server.py"
CLI = ROOT / "core" / "imageloom"
POSIX = os.name == "posix"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Sandbox(unittest.TestCase):
    """A fake HOME with its own data folder; subprocesses see nothing of the real one."""

    def setUp(self):
        self.home = Path(tempfile.mkdtemp(prefix="imageloom-cli-")).resolve()
        self.addCleanup(shutil.rmtree, self.home, True)
        self.data = self.home / "data"
        self.data.mkdir()
        self.out = self.home / "out"
        self.port = free_port()
        self.procs: list[subprocess.Popen] = []
        self.addCleanup(self.stop_all)
        (self.data / "config.json").write_text(json.dumps({"output_dir": str(self.out)}), encoding="utf-8")

    def env(self, **extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith("IMAGELOOM_")}
        env.update(HOME=str(self.home), USERPROFILE=str(self.home), IMAGELOOM_HOME=str(self.data), IMAGELOOM_PORT=str(self.port),
                   PATH=os.pathsep.join(["/usr/bin", "/bin"]) if POSIX else env.get("PATH", ""), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
        env.update(extra)
        return env

    def cli(self, *args, timeout=30, **env):
        return subprocess.run([sys.executable, str(CLI), *args], capture_output=True, text=True, check=False, timeout=timeout, env=self.env(**env), stdin=subprocess.DEVNULL, encoding="utf-8")

    def start_server(self, **env) -> subprocess.Popen:
        log = open(self.home / f"server-{len(self.procs)}.log", "w+b")
        self.addCleanup(log.close)
        p = subprocess.Popen([sys.executable, str(SERVER_PY)], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=self.env(**env))
        p.logfile = log  # type: ignore[attr-defined]
        self.procs.append(p)
        return p

    def log_of(self, p) -> str:
        p.logfile.flush()
        return Path(p.logfile.name).read_text(encoding="utf-8", errors="replace")

    def wait_for(self, p, text: str, timeout=15.0) -> str:
        end = time.time() + timeout
        while time.time() < end:
            log = self.log_of(p)
            if text in log:
                return log
            if p.poll() is not None:
                break
            time.sleep(0.05)
        self.fail(f"{text!r} not in server output (exit={p.poll()}):\n{self.log_of(p)}")

    def stop_all(self):
        subprocess.run([sys.executable, str(CLI), "stop"], capture_output=True, check=False, timeout=20, env=self.env(), stdin=subprocess.DEVNULL)  # also stops servers the CLI started itself
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(5)

    def get(self, port: int, path: str) -> tuple[int, bytes]:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            c.request("GET", path, headers={"Host": f"127.0.0.1:{port}"})
            r = c.getresponse()
            return r.status, r.read()
        finally:
            c.close()


class Server(Sandbox):
    def test_banner_notes_and_clean_shutdown(self):
        p = self.start_server()
        log = self.wait_for(p, "engine:")
        self.assertIn(f"listening on http://127.0.0.1:{self.port}", log)
        self.assertIn("IMAGELOOM_SD_CLI", log, "a missing engine is explained, with the way out")
        self.assertIn("models:", log)
        self.assertNotIn("Traceback", log)
        info = json.loads((self.data / "server.json").read_text())
        self.assertEqual((info["port"], info["pid"]), (self.port, p.pid))
        self.assertEqual(self.get(self.port, "/api/status")[0], 200)
        p.send_signal(signal.SIGTERM if POSIX else signal.CTRL_BREAK_EVENT)
        p.wait(10)
        self.assertFalse((self.data / "server.json").exists(), "the port file goes away with the server")

    @unittest.skipUnless(POSIX, "signals")
    def test_a_second_instance_on_the_same_port_says_so_and_changes_nothing(self):
        first = self.start_server()
        self.wait_for(first, "listening")
        second = self.start_server()
        self.assertEqual(second.wait(15), 1)
        text = self.log_of(second)
        self.assertIn("already running", text)
        self.assertIn(f"127.0.0.1:{self.port}", text)
        self.assertNotIn("Traceback", text)
        self.assertEqual(json.loads((self.data / "server.json").read_text())["pid"], first.pid, "the first server keeps its port file")
        self.assertEqual(self.get(self.port, "/api/status")[0], 200)

    def test_a_port_held_by_something_else_falls_back_to_a_free_one(self):
        with socket.socket() as blocker:
            blocker.bind(("127.0.0.1", self.port))
            blocker.listen(1)
            p = self.start_server()
            log = self.wait_for(p, "listening on")
        self.assertIn(f"port {self.port} is busy", log)
        other = int(log.split("http://127.0.0.1:")[1].split()[0])
        self.assertNotEqual(other, self.port)
        self.assertEqual(self.get(other, "/api/status")[0], 200)
        self.assertEqual(json.loads((self.data / "server.json").read_text())["port"], other, "clients find the new port through server.json")

    def test_bad_port_variable(self):
        p = self.start_server(IMAGELOOM_PORT="http")
        self.assertEqual(p.wait(15), 1)
        self.assertIn("IMAGELOOM_PORT must be a port number", self.log_of(p))
        self.assertNotIn("Traceback", self.log_of(p))

    def test_damaged_state_files_do_not_stop_the_start(self):
        for name, junk in (("config.json", b"{ not json"), ("history.jsonl", b'{"file": "a.png"}\n\xff\xfe garbage\n{"file": "b.png", "pro'), ("chats.json", b"[[["),
                           ("models.json", b'{"models": 5}'), ("eta_stats.json", b"null")):
            (self.data / name).write_bytes(junk)
        (self.data / "config.json").write_bytes(b"{ not json")  # (no output_dir now: the fake HOME keeps the default library inside the sandbox)
        p = self.start_server()
        log = self.wait_for(p, "listening on")
        self.assertNotIn("Traceback", log)
        for name in ("config.json", "chats.json", "models.json", "eta_stats.json"):
            self.assertTrue(list(self.data.glob(name + ".bad-*")), f"{name} was set aside: {sorted(x.name for x in self.data.iterdir())}")
        self.assertTrue(list(self.data.glob("history.jsonl.bad-*")))
        self.assertEqual(self.get(self.port, "/api/models")[0], 200)
        status, body = self.get(self.port, "/api/history")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), [], "rows whose pictures are gone are hidden")
        self.assertTrue((self.home / "Pictures" / "ImageLoom").is_dir() or sys.platform == "win32")

    def test_unusable_data_folder(self):
        blocker = self.home / "a-file"
        blocker.write_text("x")
        p = self.start_server(IMAGELOOM_HOME=str(blocker / "sub"))
        self.assertEqual(p.wait(15), 1)
        text = self.log_of(p)
        self.assertIn("cannot create the data folder", text)
        self.assertNotIn("Traceback", text)

    def test_unusable_output_folder(self):
        blocker = self.home / "a-file"
        blocker.write_text("x")
        (self.data / "config.json").write_text(json.dumps({"output_dir": str(blocker / "out")}), encoding="utf-8")
        p = self.start_server()
        self.assertEqual(p.wait(15), 1)
        text = self.log_of(p)
        self.assertIn("cannot create the output folder", text)
        self.assertIn("output_dir", text)
        self.assertNotIn("Traceback", text)


class Cli(Sandbox):
    def no_traceback(self, r):
        self.assertNotIn("Traceback", r.stderr + r.stdout)

    def test_help(self):
        r = self.cli("--help")
        self.assertEqual(r.returncode, 0)
        self.assertIn("usage", r.stdout.lower())

    def test_status_and_stop_without_a_server(self):
        r = self.cli("status")
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "server: stopped"))
        r = self.cli("stop")
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "server: not running"))

    def test_creating_without_an_engine_explains_what_to_do(self):
        r = self.cli("a red fox")
        self.no_traceback(r)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("imageloom:", r.stderr)
        self.assertIn("engine", r.stderr)
        self.assertIn("imageloom ui", r.stderr)
        self.assertEqual(r.stdout, "", "stdout carries results only")
        self.assertEqual(self.cli("status").stdout.count("engine_found"), 1)  # the CLI started a server on demand; it answers
        self.assertIn('"engine_found": false', self.cli("status").stdout)

    def test_creating_with_an_engine_but_no_models_says_so(self):
        engine = self.home / "bin" / "sd-cli"
        engine.parent.mkdir()
        engine.write_text("not a real engine, and never executed")
        r = self.cli("a red fox", IMAGELOOM_SD_CLI=str(engine))
        self.no_traceback(r)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("no model is installed", r.stderr)
        self.assertIn("models add", r.stderr)

    def test_editing_without_an_engine(self):
        pic = self.home / "p.png"
        pic.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 40)
        r = self.cli("edit", str(pic), "make it blue")
        self.no_traceback(r)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("engine", r.stderr)

    def test_models_listing_works_on_a_clean_install(self):
        r = self.cli("models")
        self.no_traceback(r)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("qwen21-turbo-q8", r.stdout)

    def test_adding_a_model_with_a_bad_link(self):
        r = self.cli("models", "add", "https://evil.example.com/a/b")
        self.no_traceback(r)
        self.assertEqual(r.returncode, 1)
        self.assertIn("does not look like a Hugging Face link", r.stderr)

    def test_bad_arguments_are_one_line_errors(self):
        cases = [(("a fox", "-s", "banana"), "bad size"), (("a fox", "-s", "1024"), "bad size"), (("a fox", "-r", "wide"), "aspect ratio"), (("a fox", "-r", "1:0"), "aspect ratio"),
                 (("a fox", "-r", "0:1"), "aspect ratio"), (("edit", "missing.png", "x"), "input not found"), (("upscale", "missing.png"), "input not found"), (("export", "missing.png"), "input not found")]
        for args, text in cases:
            with self.subTest(args):
                r = self.cli(*args)
                self.no_traceback(r)
                self.assertEqual(r.returncode, 2, r.stderr)
                self.assertIn(text, r.stderr)
                self.assertEqual(len([ln for ln in r.stderr.splitlines() if ln.strip()]), 1, r.stderr)

    def test_bad_port_variable_and_damaged_config(self):
        r = self.cli("status", IMAGELOOM_PORT="abc")
        self.assertEqual(r.returncode, 1)
        self.assertIn("IMAGELOOM_PORT must be a port number", r.stderr)
        self.no_traceback(r)
        (self.data / "config.json").write_text("{nope")
        r = self.cli("status")
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "server: stopped"))
        self.assertIn("ignoring the unreadable", r.stderr)
        self.no_traceback(r)

    def test_doctor_names_the_missing_pieces(self):
        r = self.cli("doctor")
        self.no_traceback(r)
        self.assertEqual(r.returncode, 1)
        self.assertIn("MISSING", r.stdout)
        self.assertIn("sd-cli engine", r.stdout)

    def test_batch_reports_bad_files_without_starting_the_server(self):
        missing = self.cli("batch", str(self.home / "missing.txt"))
        self.assertEqual(missing.returncode, 2)
        self.assertIn("cannot read batch file", missing.stderr)
        self.assertEqual(missing.stdout, "")
        self.no_traceback(missing)

        empty = self.home / "empty.txt"
        empty.write_text("\n# only a comment\n", encoding="utf-8")
        result = self.cli("batch", str(empty))
        self.assertEqual(result.returncode, 2)
        self.assertIn("contains no prompts", result.stderr)
        self.assertEqual(result.stdout, "")
        self.no_traceback(result)


class CliInProcess(unittest.TestCase):
    """The module without running a command: the wrapper, argument checks and external-tool calls."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="imageloom-cli-mod-")).resolve()  # (its own folder: the module reads IMAGELOOM_HOME when it loads)
        loader = importlib.machinery.SourceFileLoader("imageloom_cli", str(CLI))
        spec = importlib.util.spec_from_loader("imageloom_cli", loader)
        cls.cli = importlib.util.module_from_spec(spec)
        with mock.patch.dict(os.environ, {"IMAGELOOM_HOME": str(cls.tmp)}):
            loader.exec_module(cls.cli)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_main(self, exc):
        err = []
        with mock.patch.object(self.cli, "run", side_effect=exc), mock.patch.object(self.cli, "eprint", lambda *a, **k: err.append(" ".join(map(str, a)))):
            return self.cli.main(), " ".join(err)

    def test_main_turns_failures_into_messages(self):
        code, text = self.run_main(urllib.error.URLError("Connection refused"))
        self.assertEqual(code, 1)
        self.assertIn("cannot reach the server", text)
        self.assertIn("imageloom ui", text)
        code, text = self.run_main(KeyboardInterrupt())
        self.assertEqual(code, 130)
        self.assertIn("keeps running", text)
        code, text = self.run_main(json.JSONDecodeError("x", "y", 0))
        self.assertEqual(code, 1)
        code, text = self.run_main(self.cli.CliError("nope", 7))
        self.assertEqual((code, text), (7, "imageloom: nope"))

    def test_external_tools_get_absolute_paths_and_explicit_arguments(self):
        calls = []
        src = self.tmp / "-weird name.jpg"
        src.write_bytes(b"x")

        def fake_run(cmd, **kw):
            calls.append(cmd)
            if "--out" in cmd:
                Path(cmd[cmd.index("--out") + 1]).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0\0\0\rIHDR" + (8).to_bytes(4, "big") * 2)
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        work = Path(tempfile.mkdtemp(dir=self.tmp))
        old = os.getcwd()
        os.chdir(self.tmp)
        try:
            with mock.patch.object(self.cli.shutil, "which", lambda n: "/usr/bin/sips"), mock.patch.object(self.cli.subprocess, "run", fake_run):
                self.cli.prepare_ref(Path("-weird name.jpg"), 16, 16, work, 0)  # relative and starting with a dash
        finally:
            os.chdir(old)
        first = calls[0]
        self.assertEqual(first[:4], ["sips", "-s", "format", "png"])
        self.assertTrue(os.path.isabs(first[4]), first)  # never "-weird name.jpg", which sips would read as an option
        self.assertTrue(all(isinstance(a, str) for a in first))

    def test_a_file_that_is_not_a_picture_is_reported_plainly(self):
        txt = self.tmp / "notes.txt"
        txt.write_text("hello")
        work = Path(tempfile.mkdtemp(dir=self.tmp))
        with self.assertRaises(self.cli.CliError) as cm:
            self.cli.prepare_ref(txt, None, None, work, 0)
        self.assertIn("notes.txt", str(cm.exception))

    def test_size_and_ratio_parsing(self):
        self.assertEqual(self.cli.parse_size("1024x768"), (1024, 768))
        self.assertEqual(self.cli.parse_size("512X512"), (512, 512))
        for bad in ("1024", "axb", "", "1x2x3", "1024*768"):
            with self.assertRaises(self.cli.CliError):
                self.cli.parse_size(bad)
        self.assertEqual(self.cli.dims("1:1", "fast"), (768, 768))
        for bad in ("x", "1:0", "0:1", "-1:2", "1:2:3", "1000:1"):
            with self.assertRaises(self.cli.CliError):
                self.cli.dims(bad, "fast")

    def test_batch_file_parsing(self):
        source = self.tmp / "prompts.txt"
        source.write_text(
            "# queued in order\n"
            '\"a red fox in snow\" size=1024x768 model=qwen21\n'
            "minimal blue logo model=z-image\n"
            "sunset over the sea\n",
            encoding="utf-8",
        )
        self.assertEqual(
            self.cli.parse_batch_file(source),
            [
                {"prompt": "a red fox in snow", "size": "1024x768", "model": "qwen21"},
                {"prompt": "minimal blue logo", "size": None, "model": "z-image"},
                {"prompt": "sunset over the sea", "size": None, "model": None},
            ],
        )

    def test_batch_file_rejects_ambiguous_rows_before_running(self):
        cases = {
            "size=512x512": "prompt is empty",
            "fox size=512x512 size=768x768": "duplicate size",
            "fox model=": "model cannot be empty",
            "'unterminated": "No closing quotation",
        }
        for text, message in cases.items():
            with self.subTest(text):
                source = self.tmp / "bad.txt"
                source.write_text(text + "\n", encoding="utf-8")
                with self.assertRaises(self.cli.CliError) as cm:
                    self.cli.parse_batch_file(source)
                self.assertIn(message, str(cm.exception))

    def test_batch_runs_sequentially_with_json_output_and_line_overrides(self):
        source = self.tmp / "prompts.txt"
        source.write_text("first size=1024x768 model=one\nsecond\n", encoding="utf-8")
        output = self.tmp / "results"
        args = argparse.Namespace(
            file=str(source), quality="fast", size=None, count=1, seed=None, output=str(output), json=False,
            open=True, upscale=None, transparent=False, model="default", chat=None, up_model="quality", ratio="1:1",
        )
        seen = []

        def generate(item):
            seen.append(item)
            return 0

        with mock.patch.object(self.cli, "cmd_generate", generate):
            self.assertEqual(self.cli.cmd_batch(args), 0)
        self.assertEqual([item.prompt for item in seen], ["first", "second"])
        self.assertEqual([(item.size, item.model) for item in seen], [("1024x768", "one"), (None, "default")])
        self.assertTrue(all(item.json for item in seen))
        self.assertTrue(all(item.output == str(output) and item.open for item in seen))

        args.output = str(self.tmp / "one.png")
        with self.assertRaises(self.cli.CliError) as cm:
            self.cli.cmd_batch(args)
        self.assertIn("must be a directory", str(cm.exception))

    def test_png_size_reads_only_the_header(self):
        p = self.tmp / "x.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR" + (30).to_bytes(4, "big") + (20).to_bytes(4, "big") + b"junk" * 100)
        self.assertEqual(self.cli.png_size(p), (30, 20))
        p.write_bytes(b"GIF89a" * 10)
        with self.assertRaises(self.cli.CliError):
            self.cli.png_size(p)
        p.write_bytes(b"\x89PNG")
        with self.assertRaises(self.cli.CliError):
            self.cli.png_size(p)

    def test_the_server_is_never_reached_through_a_proxy(self):
        class Hello(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

        srv = ThreadingHTTPServer(("127.0.0.1", 0), Hello)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (srv.shutdown(), srv.server_close()))
        dead_proxy = {"http_proxy": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9", "no_proxy": "", "NO_PROXY": ""}
        with mock.patch.dict(os.environ, dead_proxy):
            with self.cli.OPENER.open(f"http://127.0.0.1:{srv.server_address[1]}/", timeout=5) as r:
                self.assertEqual(r.read(), b"ok")


if __name__ == "__main__":
    unittest.main()
