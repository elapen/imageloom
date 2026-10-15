"""The MCP server (core/imageloom_mcp.py) driven over real stdio pipes, like an MCP client would: lifecycle, JSON-RPC errors, stdout purity,
tool schemas, every tool against the real HTTP server of server.py (random port, throwaway data folder), cancellation and EOF.
Offline: Hugging Face is faked, sd-cli is never started (the process launcher is a tripwire and the job worker thread does not exist here)."""
from __future__ import annotations

import base64
import contextlib
import http.client
import io
import itertools
import json
import os
import queue
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import uuid
from pathlib import Path
from typing import ClassVar
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import _support as S  # noqa: E402
import imageloom_mcp as mcp  # noqa: E402  (importing it has no side effects)
from _support import LiveServer, M, png_bytes, server  # noqa: E402

ROOT, CORE = S.ROOT, S.CORE
MCP_PY = CORE / "imageloom_mcp.py"
CLI = CORE / "imageloom"
POSIX = os.name == "posix"
HAS_PIL = server.Image is not None
HAS_CONVERTER = bool(shutil.which("sips")) or HAS_PIL  # what the CLI's prepare_ref needs to turn a picture into a PNG
VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
TOOL_NAMES = {"generate_image", "edit_image", "upscale_image", "export_image", "get_job", "wait_for_job", "cancel_job", "list_jobs", "list_models", "add_model", "set_default_model", "recent_images"}
SANDBOX = S.HOME / "mcp-home"  # HOME of every MCP process started here: nothing of the real home folder is ever touched
SANDBOX.mkdir(exist_ok=True)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def mcp_env(port: int, **extra) -> dict:
    assert port not in (7870, 7871), "the owner's servers are off limits"
    env = {k: v for k, v in os.environ.items() if not k.startswith("IMAGELOOM_")}
    env.update(HOME=str(SANDBOX), USERPROFILE=str(SANDBOX), IMAGELOOM_HOME=str(S.DATA), IMAGELOOM_PORT=str(port), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    env.update(extra)
    return env


def text_of(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if c["type"] == "text")


def images_of(result: dict) -> list:
    return [c for c in result["content"] if c["type"] == "image"]


def image_dims(data: bytes) -> tuple:
    """(width, height) of a PNG or JPEG, from its headers."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", data[16:24])
    assert data[:2] == b"\xff\xd8", "neither PNG nor JPEG"
    i = 2
    while i < len(data):
        assert data[i] == 0xFF
        marker, length = data[i + 1], struct.unpack(">H", data[i + 2:i + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        i += 2 + length
    raise AssertionError("no SOF marker")


class McpProc:
    """The MCP server as a child process; every line it writes to stdout is recorded."""

    def __init__(self, port: int, argv: list | None = None, version: str = VERSIONS[0], **env):
        self.version = version
        self.ids = itertools.count(1)
        self.inbox: queue.Queue = queue.Queue()
        self.others: list = []  # messages that were not the answer being waited for (notifications, answers to other requests)
        self.raw: list = []
        self.stderr = b""
        self.p = subprocess.Popen(argv or [sys.executable, str(MCP_PY)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=mcp_env(port, **env), cwd=str(S.HOME))
        self.threads = [threading.Thread(target=self._read_out, daemon=True), threading.Thread(target=self._read_err, daemon=True)]
        for t in self.threads:
            t.start()

    def _read_out(self):
        for line in iter(self.p.stdout.readline, b""):
            self.raw.append(line)
            try:
                self.inbox.put(json.loads(line))
            except ValueError:
                pass  # (assert_pure reports it)

    def _read_err(self):
        for line in iter(self.p.stderr.readline, b""):
            self.stderr += line

    # -- sending
    def send_raw(self, data: bytes) -> None:
        self.p.stdin.write(data)
        self.p.stdin.flush()

    def send(self, obj) -> None:
        self.send_raw(json.dumps(obj).encode() + b"\n")

    def notify(self, method: str, params: dict | None = None) -> None:
        self.send({"jsonrpc": "2.0", "method": method, **({"params": params} if params is not None else {})})

    def submit(self, method: str, params: dict | None = None) -> int:
        rid = next(self.ids)
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, **({"params": params} if params is not None else {})})
        return rid

    # -- receiving
    def response(self, rid, timeout: float = 30) -> dict:
        end = time.monotonic() + timeout
        while True:
            for m in self.others:
                if m.get("id") == rid and "method" not in m:
                    self.others.remove(m)
                    return m
            left = end - time.monotonic()
            if left <= 0:
                raise AssertionError(f"no answer to request {rid} in {timeout} s; other messages: {self.others}; stderr:\n{self.stderr.decode(errors='replace')}")
            try:
                self.others.append(self.inbox.get(timeout=min(left, 0.2)))
            except queue.Empty:
                pass

    def request(self, method: str, params: dict | None = None, timeout: float = 30) -> dict:
        return self.response(self.submit(method, params), timeout)

    def raw_request(self, line: bytes, rid=None, timeout: float = 30) -> dict:
        self.send_raw(line + b"\n")
        return self.response(rid, timeout)

    def silence(self, seconds: float = 0.4) -> list:
        """Everything that arrives in the next moments."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            with contextlib.suppress(queue.Empty):
                self.others.append(self.inbox.get(timeout=0.05))
        got, self.others = self.others, []
        return got

    def handshake(self) -> dict:
        r = self.request("initialize", {"protocolVersion": self.version, "capabilities": {}, "clientInfo": {"name": "unittest", "version": "1"}})
        assert "result" in r, r
        self.notify("notifications/initialized")
        return r["result"]

    def call(self, name: str, args: dict | None = None, timeout: float = 30, meta: dict | None = None) -> dict:
        """The raw JSON-RPC answer to a tools/call."""
        params: dict = {"name": name, "arguments": args if args is not None else {}}
        if meta:
            params["_meta"] = meta
        return self.request("tools/call", params, timeout)

    def tool(self, name: str, args: dict | None = None, timeout: float = 30, meta: dict | None = None) -> dict:
        """The result of a tools/call that must not be a protocol error."""
        r = self.call(name, args, timeout, meta)
        assert "error" not in r, r
        return r["result"]

    # -- finishing
    def close(self, timeout: float = 10) -> int:
        with contextlib.suppress(Exception):
            self.p.stdin.close()
        try:
            code = self.p.wait(timeout)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()
            raise AssertionError("the server did not exit after stdin was closed") from None
        self.release()
        return code

    def release(self):
        for t in self.threads:
            t.join(5)
        for pipe in (self.p.stdin, self.p.stdout, self.p.stderr):
            with contextlib.suppress(Exception):
                pipe.close()

    def kill(self):
        if self.p.poll() is None:
            self.p.kill()
            self.p.wait()
        self.release()

    def assert_pure(self, allow_traceback: bool = False) -> None:
        """stdout carried nothing but JSON-RPC messages (pure ASCII, one per line), and stderr no unexpected traceback."""
        for line in self.raw:
            assert line.endswith(b"\n"), line[:100]
            line.decode("ascii")  # (ensure_ascii: raises if anything else slipped in)
            msg = json.loads(line)  # (raises if it is not JSON)
            for m in msg if isinstance(msg, list) else [msg]:
                assert isinstance(m, dict) and m.get("jsonrpc") == "2.0", line[:200]
        assert allow_traceback or b"Traceback" not in self.stderr, self.stderr.decode(errors="replace")


class Proc(unittest.TestCase):
    """Helpers for tests that start MCP processes: each one is closed, and checked for a clean stdout, when the test ends."""

    def start(self, port: int | None = None, version: str = VERSIONS[0], handshake: bool = True, allow_traceback: bool = False, **kw) -> McpProc:
        proc = McpProc(port if port is not None else free_port(), version=version, **kw)
        self.addCleanup(self._end, proc, allow_traceback)
        if handshake:
            proc.handshake()
        return proc

    def _end(self, proc: McpProc, allow_traceback: bool):
        if proc.p.poll() is None:
            self.assertEqual(proc.close(), 0)
        proc.kill()
        proc.assert_pure(allow_traceback)


# ----------------------------------------------------------------------------------------- lifecycle and JSON-RPC
class Lifecycle(Proc):
    def test_initialize_answers_with_the_clients_version_when_supported(self):
        for v in VERSIONS:
            with self.subTest(v):
                proc = self.start(version=v, handshake=False)
                r = proc.request("initialize", {"protocolVersion": v, "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}})["result"]
                self.assertEqual(r["protocolVersion"], v)
                self.assertEqual(r["serverInfo"]["name"], "imageloom")
                self.assertEqual(r["serverInfo"]["version"], server.VERSION, "the version comes from core/server.py")
                self.assertIn("tools", r["capabilities"])
                self.assertEqual("title" in r["serverInfo"], v >= "2025-06-18")
                self.assertIn("wait_for_job", r["instructions"])

    def test_an_unknown_version_gets_the_newest_one(self):
        for v in ("2099-01-01", "1999-12-31", "2025-11-25", "latest"):
            with self.subTest(v):
                r = self.start(handshake=False).request("initialize", {"protocolVersion": v, "capabilities": {}})
                self.assertEqual(r["result"]["protocolVersion"], "2025-06-18")

    def test_bad_initialize_requests(self):
        proc = self.start(handshake=False)
        for params in (None, {}, {"protocolVersion": 5}, {"protocolVersion": ""}, {"protocolVersion": None}):
            with self.subTest(params):
                self.assertEqual(proc.request("initialize", params)["error"]["code"], -32602)
        self.assertIn("result", proc.request("initialize", {"protocolVersion": "2025-03-26"}))
        again = proc.request("initialize", {"protocolVersion": "2025-03-26"})
        self.assertEqual(again["error"]["code"], -32600, "initialize happens once per connection")

    def test_notifications_get_no_answer_and_unknown_ones_are_ignored(self):
        proc = self.start()
        proc.notify("notifications/initialized")
        proc.notify("notifications/roots/list_changed")
        proc.notify("notifications/something/new", {"x": 1})
        proc.notify("notifications/cancelled", {"requestId": 12345, "reason": "unknown request"})
        proc.send({"jsonrpc": "2.0", "method": "no/such/notification", "params": [1, 2]})
        self.assertEqual(proc.silence(), [], "a notification is never answered")
        self.assertEqual(proc.request("ping")["result"], {})

    def test_ping_works_at_any_time(self):
        proc = self.start(handshake=False)
        self.assertEqual(proc.request("ping"), {"jsonrpc": "2.0", "id": 1, "result": {}})

    def test_unknown_methods(self):
        proc = self.start()
        for method in ("nope", "resources/list", "prompts/list", "logging/setLevel", "tools/delete", ""):
            with self.subTest(method):
                err = proc.request(method)["error"]
                self.assertEqual(err["code"], -32601)
                self.assertIn(method, err["message"])

    def test_parse_errors_do_not_stop_the_server(self):
        proc = self.start()
        for junk in (b"{not json", b'{"jsonrpc": "2.0", "id": 1, "method": "ping"', b"\xff\xfe\xfd", b"nul", b"{'a': 1}"):
            with self.subTest(junk):
                r = proc.raw_request(junk, rid=None)
                self.assertEqual((r["id"], r["error"]["code"]), (None, -32700))
        self.assertEqual(proc.request("ping")["result"], {})

    def test_invalid_requests(self):
        proc = self.start()
        bad = [b'{"jsonrpc": "2.0", "id": 7}', b'{"jsonrpc": "1.0", "id": 7, "method": "ping"}', b'{"id": 7, "method": "ping"}', b'{"jsonrpc": "2.0", "id": 7, "method": 5}',
               b'{"jsonrpc": "2.0", "id": 7, "method": null}', b"5", b'"ping"', b"null", b"true", b'{"jsonrpc": "2.0", "id": {"a": 1}, "method": "ping"}',
               b'{"jsonrpc": "2.0", "id": true, "method": "ping"}', b'{"jsonrpc": "2.0", "id": 1.5, "method": "ping"}', b'{"jsonrpc": "2.0", "id": [1], "method": "ping"}']
        for line in bad:
            with self.subTest(line):
                r = proc.raw_request(line, rid=7 if b'"id": 7' in line else None)
                self.assertEqual(r["error"]["code"], -32600, r)
        r = proc.raw_request(b'{"jsonrpc": "2.0", "id": 8, "method": "ping", "params": 5}', rid=8)
        self.assertEqual(r["error"]["code"], -32602)
        r = proc.raw_request(b'{"jsonrpc": "2.0", "id": 9, "method": "ping", "params": [1]}', rid=9)
        self.assertEqual(r["error"]["code"], -32602)
        self.assertEqual(proc.request("ping")["result"], {})

    def test_string_and_integer_ids_are_echoed_exactly(self):
        proc = self.start()
        for rid in ("abc", "", 0, -4, 2**40, "7"):
            with self.subTest(rid):
                self.assertEqual(proc.raw_request(json.dumps({"jsonrpc": "2.0", "id": rid, "method": "ping"}).encode(), rid=rid)["id"], rid)

    def test_replies_from_the_client_are_ignored(self):
        proc = self.start()
        proc.send({"jsonrpc": "2.0", "id": 99, "result": {}})
        proc.send({"jsonrpc": "2.0", "id": 100, "error": {"code": -1, "message": "x"}})
        self.assertEqual(proc.silence(), [])

    def test_batch_answers_come_back_as_one_array(self):
        proc = self.start(version="2025-03-26")
        proc.send([{"jsonrpc": "2.0", "id": "a", "method": "ping"}, {"jsonrpc": "2.0", "method": "notifications/initialized"}, {"jsonrpc": "2.0", "id": "b", "method": "tools/list"},
                   {"jsonrpc": "2.0", "id": "c", "method": "nope"}, 5, {"jsonrpc": "2.0", "id": "d", "method": "tools/call", "params": {"name": "get_job", "arguments": {"job_id": "nope!"}}}])
        msg = proc.inbox.get(timeout=10)
        self.assertIsInstance(msg, list)
        by_id = {m.get("id"): m for m in msg}
        self.assertEqual(by_id["a"]["result"], {})
        self.assertEqual(len(by_id["b"]["result"]["tools"]), len(TOOL_NAMES))
        self.assertEqual(by_id["c"]["error"]["code"], -32601)
        self.assertEqual(by_id[None]["error"]["code"], -32600)
        self.assertTrue(by_id["d"]["result"]["isError"], "tools/call works inside a batch too")
        self.assertEqual(len(msg), 5, "the notification inside the batch has no answer")
        proc.send([])
        self.assertEqual(proc.inbox.get(timeout=10)["error"]["code"], -32600)

    def test_blank_lines_and_crlf_are_fine(self):
        proc = self.start()
        proc.send_raw(b"\n\n   \n")
        self.assertEqual(proc.raw_request(b'{"jsonrpc": "2.0", "id": 5, "method": "ping"}\r', rid=5)["result"], {})

    def test_a_huge_message_is_refused_and_the_server_carries_on(self):
        proc = self.start()
        proc.send_raw(b'{"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {"x": "' + b"a" * (mcp.MAX_LINE + 10) + b'"}}\n')
        self.assertEqual(proc.response(None)["error"]["code"], -32700)
        self.assertEqual(proc.request("ping")["result"], {})

    def test_tools_call_protocol_errors(self):
        proc = self.start()
        cases = [({}, "name"), ({"name": 5}, "name"), ({"name": ""}, "name"), ({"name": "no_such_tool"}, "Unknown tool"), ({"name": "list_models", "arguments": [1]}, "arguments"),
                 ({"name": "list_models", "arguments": "x"}, "arguments")]
        for params, text in cases:
            with self.subTest(params):
                err = proc.request("tools/call", params)["error"]
                self.assertEqual(err["code"], -32602)
                self.assertIn(text, err["message"])
        self.assertEqual(proc.request("tools/list", {"cursor": "abc"})["error"]["code"], -32602)

    def test_non_ascii_text_survives_and_the_wire_stays_ascii(self):
        proc = self.start()
        r = proc.call("get_job", {"job_id": "ёжик 🦊"})["result"]
        self.assertTrue(r["isError"])
        self.assertIn("ёжик 🦊", text_of(r))
        self.assertTrue(all(b < 128 for line in proc.raw for b in line), "ensure_ascii: no U+2028 or raw UTF-8 for line-based readers")

    def test_eof_ends_the_server_quietly(self):
        proc = self.start()
        t0 = time.monotonic()
        self.assertEqual(proc.close(), 0)
        self.assertLess(time.monotonic() - t0, 3)
        self.assertEqual(proc.silence(0.1), [])

    def test_eof_without_any_message(self):
        proc = self.start(handshake=False)
        self.assertEqual(proc.close(), 0)
        self.assertEqual(proc.raw, [])

    def test_requests_before_initialize_are_answered(self):
        proc = self.start(handshake=False)
        self.assertEqual(len(proc.request("tools/list")["result"]["tools"]), len(TOOL_NAMES))


# ----------------------------------------------------------------------------------------- entry points
class EntryPoints(Proc):
    def test_the_cli_subcommand_serves_the_same_tools(self):
        a = self.start(argv=[sys.executable, str(MCP_PY)])
        b = self.start(argv=[sys.executable, str(CLI), "mcp"])
        ta, tb = a.request("tools/list")["result"], b.request("tools/list")["result"]
        self.assertEqual(ta, tb)
        self.assertEqual({t["name"] for t in tb["tools"]}, TOOL_NAMES)

    def test_version_and_help(self):
        for argv in ([str(MCP_PY), "--version"], [str(CLI), "mcp", "--version"]):
            with self.subTest(argv):
                r = subprocess.run([sys.executable, *argv], capture_output=True, text=True, timeout=30, env=mcp_env(free_port()), stdin=subprocess.DEVNULL)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn(server.VERSION, r.stdout)
        r = subprocess.run([sys.executable, str(CLI), "mcp", "--help"], capture_output=True, text=True, timeout=30, env=mcp_env(free_port()), stdin=subprocess.DEVNULL)
        self.assertEqual(r.returncode, 0)
        self.assertIn("MCP", r.stdout)
        self.assertIn("imageloom mcp", r.stdout)
        r = subprocess.run([sys.executable, str(CLI), "--help"], capture_output=True, text=True, timeout=30, env=mcp_env(free_port()), stdin=subprocess.DEVNULL)
        self.assertIn("imageloom mcp", r.stdout, "the CLI's own help lists the subcommand")

    def test_a_bad_port_variable_is_reported_to_the_model_not_by_dying(self):
        proc = self.start(IMAGELOOM_PORT="http")
        r = proc.tool("list_models")
        self.assertTrue(r["isError"])
        self.assertIn("IMAGELOOM_PORT must be a port number", text_of(r))
        self.assertEqual(proc.request("ping")["result"], {})


class Hygiene(Proc):
    """stdout is for the protocol only; whatever else prints, and whatever crashes, must not break the stream."""

    SCRIPT = (
        "import os, sys\n"
        f"sys.path.insert(0, {str(CORE)!r})\n"
        "import imageloom_mcp as m\n"
        "mode = sys.argv[1]\n"
        "orig = m.TOOLS['list_jobs'].fn\n"
        "def noisy(a, ctx):\n"
        "    print('NOISE-FROM-PRINT')\n"
        "    os.write(1, b'NOISE-FROM-FD1\\n')\n"
        "    os.system('echo NOISE-FROM-CHILD')\n"
        "    return orig(a, ctx)\n"
        "def boom(a, ctx):\n"
        "    raise RuntimeError('boom from a tool')\n"
        "if mode == 'noise': m.TOOLS['list_jobs'].fn = noisy\n"
        "if mode == 'boom': m.TOOLS['list_jobs'].fn = boom\n"
        "if mode == 'describe': m.TOOLS['list_jobs'].describe = lambda: 1 / 0\n"
        "raise SystemExit(m.main([]))\n"
    )

    def wrapped(self, mode: str, **kw) -> McpProc:
        return self.start(argv=[sys.executable, "-c", self.SCRIPT, mode], **kw)

    def test_prints_and_child_processes_cannot_corrupt_stdout(self):
        proc = self.wrapped("noise")
        r = proc.tool("list_jobs")
        self.assertIn("not running", text_of(r))
        self.assertEqual(proc.request("ping")["result"], {})
        self.assertEqual(proc.close(), 0)
        err = proc.stderr.decode()
        for noise in ("NOISE-FROM-PRINT", "NOISE-FROM-FD1", "NOISE-FROM-CHILD"):
            self.assertIn(noise, err, "it went to stderr instead")
        self.assertFalse([ln for ln in proc.raw if b"NOISE" in ln])

    def test_a_crashing_tool_is_an_error_result_not_a_dead_server(self):
        proc = self.wrapped("boom", allow_traceback=True)
        r = proc.tool("list_jobs")
        self.assertTrue(r["isError"])
        self.assertIn("internal error in list_jobs", text_of(r))
        self.assertIn("boom from a tool", text_of(r))
        self.assertIn("Traceback", proc.stderr.decode(), "the details are logged to stderr")
        self.assertEqual(proc.request("ping")["result"], {})

    def test_a_crash_in_protocol_code_is_a_json_rpc_internal_error(self):
        proc = self.wrapped("describe", allow_traceback=True)
        err = proc.request("tools/list")["error"]
        self.assertEqual(err["code"], -32603)
        self.assertIn("ZeroDivisionError", err["message"])
        self.assertEqual(proc.request("ping")["result"], {})

    def test_startup_banner_goes_to_stderr(self):
        proc = self.start()
        proc.request("ping")
        deadline = time.monotonic() + 5
        while b"protocol 2025-06-18" not in proc.stderr and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIn(b"imageloom-mcp:", proc.stderr)
        self.assertIn(b"protocol 2025-06-18", proc.stderr)


# ----------------------------------------------------------------------------------------- tool definitions
class Definitions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proc = McpProc(free_port())
        cls.proc.handshake()
        cls.tools = {t["name"]: t for t in cls.proc.request("tools/list")["result"]["tools"]}

    @classmethod
    def tearDownClass(cls):
        cls.proc.close()
        cls.proc.kill()

    def test_the_tool_set(self):
        self.assertEqual(set(self.tools), TOOL_NAMES)

    def test_every_tool_is_described_for_a_model(self):
        for name, t in self.tools.items():
            with self.subTest(name):
                self.assertRegex(name, r"^[a-z][a-z0-9_]{2,40}$")
                self.assertGreater(len(t["description"]), 120, "a real description, not a label")
                self.assertTrue(t["title"])
                schema = t["inputSchema"]
                self.assertEqual(schema["type"], "object")
                self.assertIs(schema["additionalProperties"], False)
                self.assertLessEqual(set(schema["required"]), set(schema["properties"]))
                json.dumps(schema)
                for pname, p in schema["properties"].items():
                    self.assertIn("type", p, pname)
                    self.assertTrue(p.get("description"), f"{name}.{pname} needs a description")
                    if "enum" in p:
                        self.assertIn(p.get("default", p["enum"][0]), p["enum"])
                    if "default" in p and p["type"] == "integer" and "minimum" in p:
                        self.assertTrue(p["minimum"] <= p["default"] <= p["maximum"], pname)
                ann = t["annotations"]
                for hint in ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"):
                    self.assertIsInstance(ann[hint], bool, f"{name}.{hint}")

    def test_hints_tell_the_truth(self):
        read_only = {n for n, t in self.tools.items() if t["annotations"]["readOnlyHint"]}
        self.assertEqual(read_only, {"get_job", "wait_for_job", "list_jobs", "list_models", "recent_images"})
        for n in read_only:
            self.assertFalse(self.tools[n]["annotations"]["destructiveHint"], n)
            self.assertTrue(self.tools[n]["annotations"]["idempotentHint"], n)
        destructive = {n for n, t in self.tools.items() if t["annotations"]["destructiveHint"]}
        self.assertEqual(destructive, {"cancel_job"}, "only cancelling throws work away; nothing here deletes files")
        for n in ("generate_image", "edit_image", "upscale_image", "export_image"):
            self.assertFalse(self.tools[n]["annotations"]["idempotentHint"], f"{n} makes a new file every time")
        self.assertTrue(self.tools["add_model"]["annotations"]["openWorldHint"], "it reaches Hugging Face")
        self.assertEqual({n for n, t in self.tools.items() if t["annotations"]["openWorldHint"]}, {"add_model"})

    def test_slow_tools_explain_the_job_pattern(self):
        for n in ("generate_image", "edit_image", "upscale_image"):
            d = self.tools[n]["description"]
            self.assertIn("wait_for_job", d, n)
            self.assertIn("50", d, n)
            self.assertIn("job_id", d, n)
        self.assertIn("4 minutes", self.tools["generate_image"]["description"])
        self.assertIn("50", self.tools["wait_for_job"]["description"])

    def test_add_model_is_a_dry_run_unless_told_otherwise(self):
        t = self.tools["add_model"]
        self.assertIs(t["inputSchema"]["properties"]["dry_run"]["default"], True)
        self.assertEqual(t["inputSchema"]["required"], ["url"])
        self.assertIn("ask", t["description"].lower())
        self.assertIn("user", t["description"].lower())
        self.assertIn("gigabytes", t["description"].lower())

    def test_documented_limits(self):
        g = self.tools["generate_image"]["inputSchema"]["properties"]
        self.assertEqual((g["wait_seconds"]["minimum"], g["wait_seconds"]["maximum"]), (0, 50))
        self.assertEqual((g["count"]["minimum"], g["count"]["maximum"]), (1, 8))
        self.assertEqual(g["quality"]["enum"], ["fast", "standard", "hq", "max"])
        self.assertEqual(self.tools["wait_for_job"]["inputSchema"]["properties"]["timeout_seconds"]["default"], 30)
        self.assertEqual(self.tools["edit_image"]["inputSchema"]["properties"]["extra_references"]["maxItems"], 2)
        self.assertEqual(self.tools["upscale_image"]["inputSchema"]["properties"]["target"]["enum"], ["2x", "4x", "2k", "4k", "8k"])
        self.assertEqual(self.tools["upscale_image"]["inputSchema"]["properties"]["model"]["enum"], ["quality", "fast", "illustration"])
        self.assertEqual(self.tools["export_image"]["inputSchema"]["properties"]["format"]["enum"], ["png", "webp", "jpeg", "svg"])
        self.assertEqual(set(self.tools["upscale_image"]["inputSchema"]["required"]), {"image_path", "target"})

    def test_the_documentation_lists_every_tool(self):
        for doc in ("docs/MCP.md", "README.md", "README.ru.md"):
            text = (ROOT / doc).read_text(encoding="utf-8")
            self.assertIn("imageloom mcp", text, doc)
        doc = (ROOT / "docs" / "MCP.md").read_text(encoding="utf-8")
        for name in TOOL_NAMES:
            self.assertIn(f"`{name}`", doc, f"docs/MCP.md must document {name}")


class Validation(unittest.TestCase):
    """The argument checker, without a process."""

    def schema(self, name):
        return mcp.TOOLS[name].input_schema

    def check(self, name, args):
        return mcp.validate_args(name, self.schema(name), args)

    def test_defaults_and_nulls(self):
        a = self.check("generate_image", {"prompt": "x", "seed": None, "width": None})
        self.assertEqual((a["quality"], a["count"], a["wait_seconds"], a["upscale_model"], a["transparent"]), ("fast", 1, 0, "quality", False))
        self.assertNotIn("seed", a)

    def test_numbers_and_booleans_sent_as_text_are_accepted(self):
        a = self.check("generate_image", {"prompt": "x", "width": "768", "height": 1024.0, "count": " 3 ", "transparent": "TRUE", "quality": "Standard"})
        self.assertEqual((a["width"], a["height"], a["count"], a["transparent"], a["quality"]), (768, 1024, 3, True, "standard"))

    def test_every_problem_is_reported_at_once(self):
        with self.assertRaises(mcp.ToolError) as cm:
            self.check("generate_image", {"width": 1000, "height": 100, "count": 0, "quality": "ultra", "aspect_ratio": "wide", "wait_seconds": 51, "size": "1x1", "seed": -1, "upscale_to": "16k", "transparent": "maybe"})
        msg = str(cm.exception)
        for part in ("missing required argument 'prompt'", "width: must be a multiple of 32", "height: must be between 512 and 2752", "count: must be between 1 and 8", "quality: must be one of",
                     "aspect_ratio", "wait_seconds", "unknown argument 'size'", "seed", "upscale_to", "transparent: must be true or false"):
            self.assertIn(part, msg)

    def test_wrong_types(self):
        for args in ({"prompt": 5}, {"prompt": ""}, {"prompt": "x" * 4001}, {"prompt": ["a"]}, {"prompt": "x", "count": True}, {"prompt": "x", "count": 1.5}, {"prompt": "x", "count": "many"},
                     {"prompt": "x", "seed": 2**60}, {"prompt": "x", "width": [768]}, {"prompt": "x", "model": ""}):
            with self.subTest(args), self.assertRaises(mcp.ToolError):
                self.check("generate_image", args)
        with self.assertRaises(mcp.ToolError):
            self.check("edit_image", {"image_path": "/a.png", "instruction": "x", "extra_references": ["a", "b", "c"]})
        with self.assertRaises(mcp.ToolError):
            self.check("edit_image", {"image_path": "/a.png", "instruction": "x", "extra_references": "a.png"})

    def test_job_ids_are_constrained(self):
        for bad in ("../status", "ABC123", "abc/def", "", "xyz", "a" * 41, "12", "abc def", "abc\n", "abcd\n", "abcd\n\n", " abcd"):
            with self.subTest(bad), self.assertRaises(mcp.ToolError):
                self.check("get_job", {"job_id": bad})
        self.assertEqual(self.check("get_job", {"job_id": "0a1b2c3d4e5f"})["job_id"], "0a1b2c3d4e5f")
        self.assertEqual(self.check("wait_for_job", {"job_id": "0a1b2c3d4e5f"})["timeout_seconds"], 30)

    def test_the_default_of_add_model_is_a_dry_run(self):
        self.assertIs(self.check("add_model", {"url": "owner/repo"})["dry_run"], True)
        self.assertIs(self.check("add_model", {"url": "owner/repo", "dry_run": "false"})["dry_run"], False)

    def test_the_version_is_the_servers(self):
        self.assertEqual(mcp.VERSION, server.VERSION)

    def test_result_shapes(self):
        r = mcp.Result("hi", {"a": 1}, [{"type": "image", "data": "", "mimeType": "image/png"}])
        new, old = r.to_mcp("2025-06-18"), r.to_mcp("2024-11-05")
        self.assertEqual(new["structuredContent"], {"a": 1})
        self.assertNotIn("structuredContent", old)
        self.assertEqual([c["type"] for c in new["content"]], ["text", "image"])
        self.assertIs(new["isError"], False)
        self.assertIs(mcp.Result("no", is_error=True).to_mcp("2025-03-26")["isError"], True)

    def test_time_formats(self):
        self.assertEqual((mcp.human_time(0), mcp.human_time(45), mcp.human_time(240), mcp.human_time(7200)), ("0 s", "45 s", "4.0 min", "2.0 h"))


class Submit(unittest.TestCase):
    """What the model is told when the server refuses a job."""

    def refuse(self, code, body):
        with mock.patch.object(mcp.B, "api", return_value=(code, body)), self.assertRaises(mcp.ToolError) as cm:
            mcp.queue_job({}, {})
        return str(cm.exception)

    def test_the_servers_503_reaches_the_model_with_what_to_do(self):
        text = self.refuse(503, {"error": "the sd-cli engine is not installed yet: open the setup screen", "need_engine": True})
        self.assertIn("the sd-cli engine is not installed yet", text)
        self.assertIn("Only the user can finish the setup", text)
        self.assertIn("no model is ready", self.refuse(503, {"error": "no model is ready: open the setup screen"}))

    def test_a_full_queue_and_plain_rejections(self):
        self.assertIn("cancel_job", self.refuse(429, {"error": "the queue is full (50 jobs)"}))
        self.assertIn("width and height must be divisible by 32", self.refuse(400, {"error": "width and height must be divisible by 32 for Qwen-Image 2.1"}))
        self.assertIn("HTTP 500", self.refuse(500, {}))

    def test_a_job_is_remembered(self):
        with mock.patch.object(mcp.B, "api", return_value=(202, {"id": "abc123def456", "status": "queued"})):
            self.assertEqual(mcp.queue_job({}, {"upscale_to": "4k"})["id"], "abc123def456")
        self.assertEqual(mcp.META["abc123def456"], {"upscale_to": "4k"})


class InputFiles(unittest.TestCase):
    """image_arg: only existing image files, judged by their bytes as well as their name."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="mcp-in-", dir=S.HOME)).resolve()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def put(self, name, data):
        p = self.dir / name
        p.write_bytes(data)
        return p

    def rejects(self, value, text):
        with self.assertRaises(mcp.ToolError) as cm:
            mcp.image_arg(value, "image_path")
        self.assertIn(text, str(cm.exception))

    def test_accepts_real_pictures(self):
        png = self.put("a.png", png_bytes(8, 8))
        self.assertEqual(mcp.image_arg(str(png), "image_path"), png)
        jpg = self.put("b.JPG", b"\xff\xd8\xff\xe0" + b"\0" * 40)
        self.assertEqual(mcp.image_arg(str(jpg), "image_path"), jpg)
        self.put("c.webp", b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\0" * 20)
        mcp.image_arg(str(self.dir / "c.webp"), "image_path")
        self.assertEqual(mcp.image_arg(str(self.dir / ".." / self.dir.name / "a.png"), "image_path"), png, "paths are resolved")

    def test_rejects_everything_else(self):
        self.put("notes.txt", b"hello")
        self.put("fake.png", b"this is text, not a picture at all" * 3)
        self.put("empty.png", b"")
        self.put("script.py", b"\x89PNG\r\n\x1a\n" + b"\0" * 30)
        (self.dir / "folder.png").mkdir()
        self.rejects(str(self.dir / "missing.png"), "no such file")
        self.rejects(str(self.dir / "notes.txt"), "not an image file")
        self.rejects(str(self.dir / "fake.png"), "does not contain a picture")
        self.rejects(str(self.dir / "empty.png"), "empty")
        self.rejects(str(self.dir / "script.py"), "not an image file")
        self.rejects(str(self.dir / "folder.png"), "not a file")
        self.rejects(str(self.dir), "not a file")
        for bad in ("", "   ", None, 5, ["a.png"], "a\x00b.png"):
            with self.subTest(repr(bad)), self.assertRaises(mcp.ToolError):
                mcp.image_arg(bad, "image_path")

    @unittest.skipUnless(S.symlinks_work(), "symlinks")
    def test_a_symlink_to_a_secret_is_judged_by_what_it_points_at(self):
        secret = self.put("secret.txt", b"password=hunter2")
        link = self.dir / "innocent.png"
        os.symlink(secret, link)
        self.rejects(str(link), "not an image file")  # (resolved first: the target is a .txt)
        sneaky = self.dir / "sneaky.png"
        target = self.put("target-without-suffix", b"password=hunter2")
        os.symlink(target, sneaky)
        self.rejects(str(sneaky), "not an image file")

    def test_the_home_folder_is_expanded_and_relative_paths_say_where_they_looked(self):
        with mock.patch.dict(os.environ, {"HOME": str(self.dir), "USERPROFILE": str(self.dir)}):
            png = self.put("home.png", png_bytes(8, 8))
            if POSIX:
                self.assertEqual(mcp.image_arg("~/home.png", "image_path"), png)
        self.rejects("nowhere/else.png", str(Path.cwd()))

    def test_sniffing(self):
        for head, kind in ((b"\x89PNG\r\n\x1a\n", "png"), (b"\xff\xd8\xff\xdb", "jpeg"), (b"RIFF\0\0\0\0WEBP", "webp"), (b"GIF89a", "gif"), (b"BM\0\0", "bmp"), (b"II*\x00", "tiff"),
                           (b"\0\0\0\x18ftypheic", "heic"), (b"<svg xmlns", None), (b"%PDF-1.4", None), (b"", None), (b"PK\x03\x04", None)):
            self.assertEqual(mcp.sniff_image(head), kind, head)


class Previews(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="mcp-prev-", dir=S.HOME)).resolve()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_small_pngs_are_attached_as_they_are(self):
        p = self.dir / "s.png"
        p.write_bytes(png_bytes(64, 48))
        mime, data = mcp.make_preview(p)
        self.assertEqual((mime, data), ("image/png", p.read_bytes()))

    def test_big_pictures_are_scaled_down_to_768(self):
        p = self.dir / "big.png"
        p.write_bytes(png_bytes(1600, 1000))
        got = mcp.make_preview(p)
        if mcp.preview_backend() is None:
            self.assertIsNone(got, "no Pillow and no sips: no preview, the path is still in the text")
            return
        mime, data = got
        self.assertIn(mime, ("image/jpeg", "image/png"))
        w, h = image_dims(data)
        self.assertEqual(max(w, h), 768)
        self.assertLessEqual(len(data), mcp.PREVIEW_BYTES)

    @unittest.skipUnless(HAS_PIL, "needs Pillow to build a transparent picture")
    def test_transparency_survives_as_png(self):
        from PIL import Image
        p = self.dir / "alpha.png"
        Image.new("RGBA", (1500, 900), (200, 30, 30, 90)).save(p)
        mime, data = mcp.make_preview(p)
        self.assertEqual(mime, "image/png")
        self.assertEqual(max(image_dims(data)), 768)
        with Image.open(io.BytesIO(data)) as im:
            self.assertEqual(im.mode, "RGBA")
            self.assertEqual(im.getpixel((5, 5))[3], 90)

    def test_junk_and_svg_give_none(self):
        (self.dir / "x.png").write_bytes(b"not a picture" * 10)
        self.assertIsNone(mcp.make_preview(self.dir / "x.png"))
        (self.dir / "x.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
        self.assertIsNone(mcp.make_preview(self.dir / "x.svg"))
        self.assertIsNone(mcp.make_preview(self.dir / "missing.png"))

    def test_at_most_four_previews(self):
        paths = []
        for i in range(7):
            p = self.dir / f"{i}.png"
            p.write_bytes(png_bytes(16, 16, color=(i * 30, 0, 0)))
            paths.append(p)
        self.assertEqual(len(mcp.preview_items(paths)), 4)


# ----------------------------------------------------------------------------------------- the tools against the real server
class LiveCase(Proc):
    """One live ImageLoom server (this process, random port, throwaway folders) and one MCP process talking to it."""

    version = VERSIONS[0]

    @classmethod
    def setUpClass(cls):
        cls.live = LiveServer().__enter__()
        cls.mcp = McpProc(cls.live.port, version=cls.version)
        cls.mcp.handshake()

    @classmethod
    def tearDownClass(cls):
        code = cls.mcp.close()
        cls.mcp.kill()
        cls.live.__exit__(None, None, None)
        assert code == 0, f"the MCP server exited with {code}"
        cls.mcp.assert_pure()

    def setUp(self):
        self.mcp.others.clear()
        with server.LOCK:
            saved = (dict(server.JOBS), list(server.JOB_ORDER), list(server.QUEUE), list(server.HIST), json.dumps(server.REG.data), set(server.OUT.rglob("*")))
        self.addCleanup(self.restore, saved)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)

    def restore(self, saved):
        jobs, order, queue_, hist, reg, files = saved
        with server.LOCK:
            server.JOBS.clear(), server.JOB_ORDER.clear(), server.QUEUE.clear()
            server.JOBS.update(jobs), server.JOB_ORDER.extend(order), server.QUEUE.extend(queue_)
            server.HIST[:] = hist
            server.persist_history()
        data = json.loads(reg)
        server.REG.data["models"], server.REG.data["active"] = data["models"], data["active"]
        server.REG.save()
        for p in set(server.OUT.rglob("*")) - files:
            if p.is_file():
                p.unlink()
        self.mcp.others.clear()

    # -- fixtures
    def put_png(self, name: str, data: bytes | None = None) -> Path:
        p = server.OUT / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data or png_bytes(64, 48))
        return p

    def add_job(self, status: str, **fields) -> str:
        jid = uuid.uuid4().hex[:12]
        job = {"id": jid, "status": status, "mode": "generate", "prompt": "a test fox", "width": 768, "height": 768, "seeds": [11], "refs": [], "images": [], "eta": 240,
               "model": "qwen21-turbo-q8", "model_name": "Qwen-Image 2.1 Turbo · Q8", "queued": time.time(), "step": 0, "steps": 8, "phase": "loading model", **fields}
        if status == "running":
            job.setdefault("started", time.time() - 5)
        with server.LOCK:
            server.JOBS[jid] = job
            server.JOB_ORDER.append(jid)
            if status == "queued":
                server.QUEUE.append(jid)
        return jid

    def patch(self, **attrs):
        self.stack.enter_context(mock.patch.multiple(server, **attrs))

    def make_ready(self):
        """A pretend engine and models, so jobs are accepted. Nothing can run them: spawn() is a tripwire and no worker thread exists in the tests."""
        engine = S.HOME / "fake-engine" / "sd-cli"
        engine.parent.mkdir(exist_ok=True)
        engine.write_bytes(b"never executed")
        up = Path(server.CFG["upscale_dir"])
        up.mkdir(parents=True, exist_ok=True)
        weights = [up / v[0] for v in server.UPSCALE_MODELS.values()]
        for w in weights:
            if not w.exists():
                with open(w, "wb") as fh:
                    fh.truncate(6_000_000)
                self.addCleanup(w.unlink)
        self.patch(find_sd_cli=lambda: engine, model_ready=lambda m: True, model_can_edit=lambda m: True, any_model_ready=lambda: True)

    def jobs_now(self) -> set:
        return set(server.JOBS)


class Models(LiveCase):
    def test_list_models(self):
        r = self.mcp.tool("list_models")
        self.assertFalse(r["isError"])
        rows = {m["id"]: m for m in r["structuredContent"]["models"]}
        b = rows["qwen21-turbo-q8"]
        self.assertEqual((b["status"], b["ready"], b["edit"], b["alpha"], b["experimental"]), ("files_missing", False, False, True, False), "no engine, no files here")
        self.assertEqual(r["structuredContent"]["default_model"], server.REG.data["active"])
        self.assertTrue(rows[server.REG.data["active"]]["default"])
        self.assertGreater(r["structuredContent"]["free_disk_bytes"], 0)
        text = text_of(r)
        self.assertIn("qwen21-turbo-q8", text)
        self.assertIn("files_missing", text)
        self.assertIn("No model is ready yet", text)

    def test_ready_models_show_their_abilities(self):
        self.make_ready()
        b = {m["id"]: m for m in self.mcp.tool("list_models")["structuredContent"]["models"]}["qwen21-turbo-q8"]
        self.assertEqual((b["status"], b["ready"], b["edit"], b["alpha"]), ("ready", True, True, True))

    def test_a_downloading_model_shows_progress(self):
        server.REG.add({"id": "mcp-dl", "name": "Downloading", "family": "z-image", "variant": "turbo", "files": {}, "status": "downloading", "size": 4_000_000_000, "experimental": True})
        self.patch(SETUP={"running": True, "cancel": False, "items": {"m:mcp-dl:diffusion": {"bytes": 1_000, "total": 4_000, "state": "running"}}})
        r = self.mcp.tool("list_models")
        row = next(m for m in r["structuredContent"]["models"] if m["id"] == "mcp-dl")
        self.assertEqual((row["status"], row["download_progress"], row["experimental"]), ("downloading", 0.25, True))
        self.assertIn("downloading 25%", text_of(r))

    def test_set_default_model(self):
        server.REG.add({"id": "mcp-other", "name": "Other", "family": "z-image", "variant": "turbo", "files": {}, "status": "ready", "size": 1})
        before = server.REG.data["active"]
        r = self.mcp.tool("set_default_model", {"id": "mcp-other"})
        self.assertFalse(r["isError"])
        self.assertEqual(server.REG.data["active"], "mcp-other")
        self.assertIn("not ready", text_of(r), "a model without files is flagged")
        self.assertEqual(r["structuredContent"]["default_model"], "mcp-other")
        again = self.mcp.tool("set_default_model", {"id": "mcp-other"})
        self.assertFalse(again["isError"], "idempotent")
        self.mcp.tool("set_default_model", {"id": before})
        self.assertEqual(server.REG.data["active"], before)

    def test_set_default_model_with_an_unknown_id_changes_nothing(self):
        before = server.REG.data["active"]
        r = self.mcp.tool("set_default_model", {"id": "no-such-model"})
        self.assertTrue(r["isError"])
        self.assertIn("qwen21-turbo-q8", text_of(r), "the installed ids are listed")
        self.assertEqual(server.REG.data["active"], before)


class AddModel(LiveCase):
    ROUTES: ClassVar[dict] = {"someone/Qwen-Image-2.1-Turbo-GGUF/tree": "qwen21_tree.json", "someone/Qwen-Image-2.1-Turbo-GGUF": "qwen21_info.json",
                              "Qwen/Qwen3-VL-8B-Instruct-GGUF/tree": "qwen3vl_tree.json", "Comfy-Org/Qwen-Image-2.1/tree": "comfy_qwen21_tree.json",
                              "someone/SDXL-GGUF/tree": "sdxl_diffusers_tree.json", "someone/SDXL-GGUF": "sdxl_diffusers_info.json"}
    URL = "https://huggingface.co/someone/Qwen-Image-2.1-Turbo-GGUF"

    def setUp(self):
        super().setUp()
        self.hf = mock.Mock(side_effect=S.fake_hf(self.ROUTES))
        self.stack.enter_context(mock.patch.object(M, "hf_json", self.hf))
        self.registry = json.dumps(server.REG.data, sort_keys=True)

    def assert_nothing_installed(self):
        self.assertEqual(json.dumps(server.REG.data, sort_keys=True), self.registry)
        self.assertFalse(server.SETUP["running"])

    def test_the_default_is_a_dry_run_that_downloads_nothing(self):
        r = self.mcp.tool("add_model", {"url": self.URL})
        self.assertFalse(r["isError"])
        plan = r["structuredContent"]
        self.assertTrue(plan["dry_run"])
        self.assertEqual((plan["repo"], plan["family"], plan["variant"]), ("someone/Qwen-Image-2.1-Turbo-GGUF", "qwen-image-2.1", "turbo"))
        self.assertEqual(plan["weights"]["file"], "Qwen-Image-2.1-Turbo-Q8_0.gguf")
        self.assertGreater(plan["weights"]["size_bytes"], 1_000_000_000)
        self.assertEqual({c["file"] for c in plan["companions"]}, {"Qwen3VL-8B-Instruct-Q8_0.gguf", "vae/qwen_image_2.1_vae_bf16.safetensors", "mmproj-Qwen3VL-8B-Instruct-F16.gguf"})
        self.assertEqual(plan["engine_download_bytes"], 35_000_000, "this install has no engine yet: it would be fetched first")
        self.assertGreater(plan["total_download_bytes_max"], plan["weights"]["size_bytes"] + sum(c["size_bytes"] for c in plan["companions"]))
        self.assertIn("fits_on_disk", plan)
        text = text_of(r)
        self.assertIn("Nothing was downloaded", text)
        self.assertIn("ask the user", text)
        self.assertIn("dry_run=false", text)
        self.assert_nothing_installed()

    def test_an_explicit_dry_run_and_the_implicit_one_are_the_same(self):
        a = self.mcp.tool("add_model", {"url": self.URL})["structuredContent"]
        b = self.mcp.tool("add_model", {"url": "someone/Qwen-Image-2.1-Turbo-GGUF", "dry_run": True})["structuredContent"]
        self.assertEqual(a, b)
        self.assert_nothing_installed()

    def test_the_real_download_is_requested_only_with_dry_run_false(self):
        seen = []

        def fake_add(body):
            seen.append(body)
            return {"models": [{"id": "my-qwen-q8-0", "status": "downloading"}], "active": "qwen21-turbo-q8"}

        self.patch(models_add=fake_add)
        self.mcp.tool("add_model", {"url": self.URL})
        self.assertEqual(seen, [], "the default never starts a download")
        r = self.mcp.tool("add_model", {"url": self.URL, "dry_run": False})
        self.assertFalse(r["isError"])
        self.assertEqual(seen, [{"repo": "someone/Qwen-Image-2.1-Turbo-GGUF", "family": "qwen-image-2.1", "variant": "turbo", "file": "Qwen-Image-2.1-Turbo-Q8_0.gguf", "name": "Qwen-Image-2.1-Turbo-GGUF"}])
        self.assertEqual(r["structuredContent"]["model_id"], "my-qwen-q8-0")
        self.assertIn("list_models", text_of(r))
        self.assertIn("background", text_of(r))

    def test_choosing_a_file_family_and_variant(self):
        seen = []
        self.patch(models_add=lambda body: seen.append(body) or {"models": [{"id": "x", "status": "downloading"}]})
        plan = self.mcp.tool("add_model", {"url": self.URL})["structuredContent"]
        other = next(f["file"] for f in plan["other_files"])
        r = self.mcp.tool("add_model", {"url": self.URL, "file": other, "variant": "base", "family": "qwen-image-2.1", "dry_run": False})
        self.assertFalse(r["isError"], text_of(r))
        self.assertEqual((seen[0]["file"], seen[0]["variant"], seen[0]["family"]), (other, "base", "qwen-image-2.1"))

    def test_bad_requests_are_error_results_and_install_nothing(self):
        cases = [({"url": "https://evil.example.com/a/b"}, "does not look like a Hugging Face link"), ({"url": "not a link at all"}, "Hugging Face"),
                 ({"url": self.URL, "file": "nope.gguf"}, "choose `file` from"), ({"url": self.URL, "family": "dall-e"}, "unknown family"),
                 ({"url": "someone/missing-repo"}, "cannot inspect"), ({"url": "https://huggingface.co/datasets/a/b"}, "Hugging Face")]
        for args, text in cases:
            with self.subTest(args):
                r = self.mcp.tool("add_model", {**args, "dry_run": False})
                self.assertTrue(r["isError"], text_of(r))
                self.assertIn(text, text_of(r))
        self.assert_nothing_installed()

    def test_hugging_face_trouble_is_explained(self):
        self.hf.side_effect = urllib.error.URLError("unreachable")
        r = self.mcp.tool("add_model", {"url": self.URL})
        self.assertTrue(r["isError"])
        self.assertIn("could not reach Hugging Face", text_of(r))
        self.hf.side_effect = PermissionError("this repository is gated")
        r = self.mcp.tool("add_model", {"url": self.URL})
        self.assertTrue(r["isError"])
        self.assertIn("token", text_of(r))
        self.assertIn("gated", text_of(r))

    def test_a_repository_without_weights(self):
        self.hf.side_effect = S.fake_hf({"someone/Empty/tree": "empty_tree.json", "someone/Empty": "empty_info.json"})
        r = self.mcp.tool("add_model", {"url": "someone/Empty"})
        self.assertTrue(r["isError"])
        self.assertIn("no usable weights", text_of(r))


class Gallery(LiveCase):
    def row(self, name: str, prompt: str, kind: str = "gen", data: bytes | None = None, **extra) -> dict:
        self.put_png(name, data)
        row = {"id": uuid.uuid4().hex[:10], "file": name, "kind": kind, "prompt": prompt, "width": 64, "height": 48, "seed": 5, "seconds": 3, "ts": int(time.time()), "parent": None, "refs": [], "fav": False,
               "model": "qwen21-turbo-q8", **extra}
        server.add_rows([row])
        return row

    def test_recent_images_newest_first_with_absolute_paths(self):
        self.row("mcp-a.png", "first fox")
        self.row("mcp-b.png", "second fox", kind="edit")
        self.row("mcp-c-upload.png", "internal upload", kind="upload")
        r = self.mcp.tool("recent_images", {"limit": 2})
        self.assertFalse(r["isError"])
        items = r["structuredContent"]["images"]
        self.assertEqual([Path(i["path"]).name for i in items], ["mcp-b.png", "mcp-a.png"], "newest first, uploads are internal and hidden")
        self.assertEqual((items[0]["kind"], items[0]["prompt"], items[0]["seed"], items[0]["width"]), ("edit", "second fox", 5, 64))
        for i in items:
            self.assertTrue(os.path.isabs(i["path"]) and os.path.isfile(i["path"]), i)
            self.assertEqual(Path(i["path"]).resolve().parent, server.OUT.resolve())
        self.assertIn("second fox", text_of(r))
        self.assertIn(items[0]["path"], text_of(r))
        self.assertEqual(images_of(r), [], "no previews unless asked")

    def test_previews_on_request(self):
        self.row("mcp-p1.png", "one")
        self.row("mcp-p2.png", "two")
        r = self.mcp.tool("recent_images", {"limit": 5, "previews": 2})
        imgs = images_of(r)
        self.assertEqual(len(imgs), 2)
        for img in imgs:
            self.assertIn(img["mimeType"], ("image/png", "image/jpeg"))
            self.assertLessEqual(max(image_dims(base64.b64decode(img["data"]))), 768)

    def test_limits_and_defaults(self):
        for i in range(12):
            self.row(f"mcp-n{i:02d}.png", f"p{i}")
        self.assertEqual(len(self.mcp.tool("recent_images")["structuredContent"]["images"]), 10, "default limit")
        for bad in ({"limit": 0}, {"limit": 51}, {"limit": "many"}, {"previews": 5}, {"previews": -1}):
            with self.subTest(bad):
                self.assertTrue(self.mcp.tool("recent_images", bad)["isError"])

    def test_files_that_are_gone_are_not_listed(self):
        row = self.row("mcp-gone.png", "vanishing")
        (server.OUT / row["file"]).unlink()
        names = [Path(i["path"]).name for i in self.mcp.tool("recent_images", {"limit": 50})["structuredContent"]["images"]]
        self.assertNotIn("mcp-gone.png", names)

    def test_an_empty_gallery(self):
        with server.LOCK:
            server.HIST[:] = []
        r = self.mcp.tool("recent_images")
        self.assertFalse(r["isError"])
        self.assertEqual(r["structuredContent"]["images"], [])
        self.assertIn("empty", text_of(r))


class Jobs(LiveCase):
    def test_a_running_job(self):
        jid = self.add_job("running", step=3, steps=8, phase="denoising", started=time.time() - 12)
        r = self.mcp.tool("get_job", {"job_id": jid})
        self.assertFalse(r["isError"])
        d = r["structuredContent"]
        self.assertEqual((d["status"], d["mode"], d["phase"], d["step"], d["steps"]), ("running", "generate", "denoising", 3, 8))
        self.assertGreaterEqual(d["elapsed_seconds"], 12)
        self.assertEqual(d["estimated_seconds"], 240)
        self.assertLessEqual(d["remaining_seconds"], 228)
        self.assertEqual(d["next_action"], "wait_for_job")
        self.assertEqual(d["images"], [])
        text = text_of(r)
        for part in ("denoising 3/8", "elapsed", "wait_for_job", jid):
            self.assertIn(part, text)

    def test_a_queued_job(self):
        jid = self.add_job("queued")
        d = self.mcp.tool("get_job", {"job_id": jid})["structuredContent"]
        self.assertEqual((d["status"], d["queue_position"]), ("queued", server.QUEUE.index(jid) + 1))
        self.assertIn("queued", text_of(self.mcp.tool("get_job", {"job_id": jid})))

    def test_a_finished_job_returns_paths_and_previews(self):
        small, big = self.put_png("mcp-j1.png", png_bytes(64, 48)), self.put_png("mcp-j2.png", png_bytes(1600, 1000))
        jid = self.add_job("done", images=[small.name, big.name], seeds=[11, 12], seconds=241.2, step=8, phase="saving", width=1600, height=1000)
        r = self.mcp.tool("get_job", {"job_id": jid})
        self.assertFalse(r["isError"])
        d = r["structuredContent"]
        self.assertEqual(d["status"], "done")
        self.assertIsNone(d["next_action"])
        self.assertEqual([Path(i["path"]).name for i in d["images"]], ["mcp-j1.png", "mcp-j2.png"])
        self.assertEqual([i["seed"] for i in d["images"]], [11, 12])
        for i in d["images"]:
            self.assertTrue(os.path.isabs(i["path"]) and os.path.isfile(i["path"]))
        text = text_of(r)
        self.assertIn(d["images"][0]["path"], text)
        self.assertIn("done in 4.0 min", text)
        imgs = images_of(r)
        self.assertGreaterEqual(len(imgs), 1, "the small PNG is always attachable")
        for img in imgs:
            self.assertLessEqual(max(image_dims(base64.b64decode(img["data"]))), 768)
        if mcp.preview_backend():
            self.assertEqual(len(imgs), 2)
        self.assertEqual(r["content"][0]["type"], "text")

    def test_at_most_four_previews_for_many_images(self):
        names = [self.put_png(f"mcp-many{i}.png", png_bytes(16, 16)).name for i in range(6)]
        jid = self.add_job("done", images=names, seeds=list(range(6)), seconds=9.0)
        r = self.mcp.tool("get_job", {"job_id": jid})
        self.assertEqual(len(images_of(r)), 4)
        self.assertEqual(len(r["structuredContent"]["images"]), 6)
        for n in names:
            self.assertIn(n, text_of(r))

    def test_a_failed_job_is_an_error_result(self):
        jid = self.add_job("error", error="[E] out of memory", seconds=3.0)
        r = self.mcp.tool("get_job", {"job_id": jid})
        self.assertTrue(r["isError"])
        self.assertIn("FAILED", text_of(r))
        self.assertIn("out of memory", text_of(r))
        self.assertEqual(r["structuredContent"]["status"], "error")

    def test_a_cancelled_job_is_just_reported(self):
        jid = self.add_job("cancelled", seconds=3.0)
        r = self.mcp.tool("get_job", {"job_id": jid})
        self.assertFalse(r["isError"])
        self.assertIn("cancelled", text_of(r))

    def test_unknown_and_malformed_ids(self):
        r = self.mcp.tool("get_job", {"job_id": "deadbeefcafe"})
        self.assertTrue(r["isError"])
        self.assertIn("no job with id deadbeefcafe", text_of(r))
        self.assertIn("recent_images", text_of(r))
        for bad in ("../api/status", "NOPE", "a b", "1"):
            with self.subTest(bad):
                r = self.mcp.tool("get_job", {"job_id": bad})
                self.assertTrue(r["isError"])
                self.assertIn("job_id", text_of(r))

    def test_a_job_outside_the_library_is_refused(self):
        jid = self.add_job("done", images=["../../etc/hosts.png"], seconds=1.0)
        r = self.mcp.tool("get_job", {"job_id": jid})
        self.assertTrue(r["isError"])
        self.assertIn("outside", text_of(r))

    def test_list_jobs(self):
        a, b, c = self.add_job("done", images=[], seconds=2.0, prompt="old"), self.add_job("running", step=2, phase="denoising", prompt="busy fox"), self.add_job("queued", prompt="waiting fox")
        r = self.mcp.tool("list_jobs")
        self.assertFalse(r["isError"])
        ids = [j["job_id"] for j in r["structuredContent"]["jobs"]]
        self.assertEqual(ids[:3], [c, b, a], "newest first")
        row = r["structuredContent"]["jobs"][1]
        self.assertEqual((row["status"], row["prompt"], row["step"], row["steps"]), ("running", "busy fox", 2, 8))
        self.assertIn("busy fox", text_of(r))
        self.assertIn(b, text_of(r))

    def test_cancel_a_running_job(self):
        jid = self.add_job("running")
        r = self.mcp.tool("cancel_job", {"job_id": jid})
        self.assertFalse(r["isError"])
        self.assertEqual(server.JOBS[jid]["status"], "cancelled")
        self.assertEqual(r["structuredContent"]["cancelled"], [jid])
        again = self.mcp.tool("cancel_job", {"job_id": jid})
        self.assertFalse(again["isError"], "idempotent")
        self.assertIn("nothing to cancel", text_of(again))

    def test_cancel_a_queued_job_and_a_finished_one(self):
        q, d = self.add_job("queued"), self.add_job("done", seconds=1.0)
        self.mcp.tool("cancel_job", {"job_id": q})
        self.assertEqual(server.JOBS[q]["status"], "cancelled")
        r = self.mcp.tool("cancel_job", {"job_id": d})
        self.assertIn("already done", text_of(r))
        self.assertEqual(server.JOBS[d]["status"], "done", "finished work is left alone")

    def test_cancel_unknown(self):
        self.assertTrue(self.mcp.tool("cancel_job", {"job_id": "deadbeefcafe"})["isError"])

    def test_a_job_with_an_automatic_upscale_is_one_job(self):
        first = self.put_png("mcp-first.png")
        up = self.add_job("running", mode="upscale", step=1, steps=4, phase="upscaling", width=3840, height=3840, eta=100)
        jid = self.add_job("done", images=[first.name], seconds=200.0, next=[up], width=768, height=768)
        d = self.mcp.tool("get_job", {"job_id": jid})["structuredContent"]
        self.assertEqual((d["status"], d["phase"], d["step"], d["steps"]), ("running", "upscaling", 1, 4), "not done until the upscale is")
        self.assertEqual(d["images"], [])
        self.assertEqual(d["intermediate_images"], [str(first.resolve())])
        self.assertEqual(d["next_action"], "wait_for_job")
        server.JOBS[up].update(status="done", images=[self.put_png("mcp-up.png").name], seconds=60.0)
        done = self.mcp.tool("get_job", {"job_id": jid})
        d = done["structuredContent"]
        self.assertEqual(d["status"], "done")
        self.assertEqual([Path(i["path"]).name for i in d["images"]], ["mcp-up.png"], "the upscaled picture is the result")
        self.assertEqual([Path(p).name for p in d["originals"]], ["mcp-first.png"])
        self.assertEqual((d["width"], d["height"]), (3840, 3840))
        self.assertGreaterEqual(d["elapsed_seconds"], 260)
        self.assertIn("Before the upscale", text_of(done))

    def test_the_moment_between_done_and_the_chain_is_not_reported_as_done(self):
        """The server marks a job done a moment before it lists the chained upscale: that gap must not look like 'finished'."""
        self.make_ready()
        jid = self.mcp.tool("generate_image", {"prompt": "x", "upscale_to": "2x"})["structuredContent"]["job_id"]
        first = self.put_png("mcp-gap.png")
        server.JOBS[jid].update(status="done", images=[first.name], seconds=5.0)
        d = self.mcp.tool("get_job", {"job_id": jid})["structuredContent"]
        self.assertEqual((d["status"], d["phase"], d["images"]), ("running", "waiting for the upscale to start", []))
        server.JOBS[jid]["next"] = []  # (the server lists no follow-up when it could not queue one)
        d = self.mcp.tool("get_job", {"job_id": jid})["structuredContent"]
        self.assertEqual((d["status"], [Path(i["path"]).name for i in d["images"]]), ("done", ["mcp-gap.png"]))

    def test_a_failed_automatic_upscale_still_returns_the_first_picture(self):
        first = self.put_png("mcp-keep.png")
        up = self.add_job("error", mode="upscale", error="upscale exploded", seconds=1.0)
        jid = self.add_job("done", images=[first.name], seconds=100.0, next=[up])
        r = self.mcp.tool("get_job", {"job_id": jid})
        self.assertFalse(r["isError"])
        d = r["structuredContent"]
        self.assertEqual((d["status"], [Path(i["path"]).name for i in d["images"]]), ("done", ["mcp-keep.png"]))
        self.assertIn("upscale exploded", " ".join(d["warnings"]))
        self.assertIn("Warning", text_of(r))

    def test_cancelling_the_chain_cancels_the_follow_up(self):
        up = self.add_job("queued", mode="upscale")
        jid = self.add_job("done", images=[self.put_png("mcp-c.png").name], seconds=5.0, next=[up])
        r = self.mcp.tool("cancel_job", {"job_id": jid})
        self.assertEqual(r["structuredContent"]["cancelled"], [up])
        self.assertEqual((server.JOBS[jid]["status"], server.JOBS[up]["status"]), ("done", "cancelled"))

    def test_jobs_tools_do_not_start_a_server(self):
        proc = self.start(free_port())  # nothing listens there
        r = proc.tool("get_job", {"job_id": "deadbeefcafe"})
        self.assertTrue(r["isError"])
        self.assertIn("not running", text_of(r))
        r = proc.tool("list_jobs")
        self.assertFalse(r["isError"])
        self.assertEqual(r["structuredContent"]["jobs"], [])
        self.assertTrue(proc.tool("wait_for_job", {"job_id": "deadbeefcafe"})["isError"])
        self.assertTrue(proc.tool("cancel_job", {"job_id": "deadbeefcafe"})["isError"])


class Waiting(LiveCase):
    def test_a_finished_job_returns_at_once(self):
        jid = self.add_job("done", images=[self.put_png("mcp-w1.png").name], seconds=7.0)
        t0 = time.monotonic()
        r = self.mcp.tool("wait_for_job", {"job_id": jid, "timeout_seconds": 30})
        self.assertLess(time.monotonic() - t0, 3)
        self.assertEqual(r["structuredContent"]["status"], "done")
        self.assertEqual(len(images_of(r)), 1)

    def test_the_wait_ends_at_the_timeout_and_says_to_call_again(self):
        jid = self.add_job("running", step=2, phase="denoising")
        t0 = time.monotonic()
        r = self.mcp.tool("wait_for_job", {"job_id": jid, "timeout_seconds": 2})
        took = time.monotonic() - t0
        self.assertTrue(1.5 <= took < 6, took)
        self.assertFalse(r["isError"])
        self.assertEqual(r["structuredContent"]["status"], "running")
        self.assertIn("Still running", text_of(r))
        self.assertIn(f'job_id "{jid}"', text_of(r))

    def test_a_failure_that_happens_while_waiting(self):
        jid = self.add_job("running")
        threading.Timer(1.0, lambda: server.JOBS[jid].update(status="error", error="[E] the engine crashed", seconds=2.0)).start()
        r = self.mcp.tool("wait_for_job", {"job_id": jid, "timeout_seconds": 20})
        self.assertTrue(r["isError"])
        self.assertIn("the engine crashed", text_of(r))

    def test_progress_notifications(self):
        name = self.put_png("mcp-prog.png").name
        jid = self.add_job("running", step=0, steps=8, phase="denoising")
        job = server.JOBS[jid]

        def advance():
            for step in range(1, 6):
                time.sleep(0.7)
                job["step"] = step
            job.update(status="done", images=[name], seconds=9.0, step=8)

        threading.Thread(target=advance, daemon=True).start()
        r = self.mcp.tool("wait_for_job", {"job_id": jid, "timeout_seconds": 30}, meta={"progressToken": "tok-1"})
        self.assertEqual(r["structuredContent"]["status"], "done")
        notes = [m for m in self.mcp.others if m.get("method") == "notifications/progress"]
        self.assertGreaterEqual(len(notes), 3, notes)
        values = [n["params"]["progress"] for n in notes]
        self.assertEqual(values, sorted(set(values)), "progress strictly increases")
        for n in notes:
            self.assertEqual(n["params"]["progressToken"], "tok-1")
            self.assertEqual(n["params"]["total"], 100)
            self.assertIn("denoising", n["params"]["message"])
            self.assertLess(n["params"]["progress"], 100)
        self.assertNotIn("id", notes[0], "notifications have no id")

    def test_an_integer_token_works_and_no_token_means_no_notifications(self):
        jid = self.add_job("running", step=1, steps=8, phase="denoising")
        self.mcp.tool("wait_for_job", {"job_id": jid, "timeout_seconds": 1}, meta={"progressToken": 77})
        self.assertTrue([m for m in self.mcp.others if m.get("method") == "notifications/progress" and m["params"]["progressToken"] == 77])
        self.mcp.others.clear()
        self.mcp.tool("wait_for_job", {"job_id": jid, "timeout_seconds": 2})
        self.assertEqual([m for m in self.mcp.others if m.get("method") == "notifications/progress"], [])

    def test_cancelling_a_wait_stops_it_and_sends_no_answer(self):
        jid = self.add_job("running", step=1, phase="denoising")
        with mock.patch.object(server, "public", wraps=server.public) as pub:
            rid = self.mcp.submit("tools/call", {"name": "wait_for_job", "arguments": {"job_id": jid, "timeout_seconds": 50}})
            end = time.monotonic() + 10
            while pub.call_count < 2 and time.monotonic() < end:
                time.sleep(0.05)
            self.assertGreaterEqual(pub.call_count, 2, "the server is being polled")
            self.mcp.notify("notifications/cancelled", {"requestId": rid, "reason": "the user pressed stop"})
            time.sleep(1.5)
            polled = pub.call_count
            time.sleep(2.5)
            self.assertEqual(pub.call_count, polled, "no more polling after the cancellation")
        self.assertEqual([m for m in self.mcp.silence(0.3) if m.get("id") == rid], [], "a cancelled request gets no response")
        self.assertEqual(self.mcp.request("ping")["result"], {}, "and the server is fine")
        self.assertEqual(server.JOBS[jid]["status"], "running", "cancelling the wait does not cancel the job")

    def test_a_cancellation_right_behind_its_request_is_not_lost(self):
        jid = self.add_job("running", step=1, phase="denoising")
        with mock.patch.object(server, "public", wraps=server.public) as pub:
            for _ in range(5):
                rid = self.mcp.submit("tools/call", {"name": "wait_for_job", "arguments": {"job_id": jid, "timeout_seconds": 50}})
                self.mcp.notify("notifications/cancelled", {"requestId": rid})  # (back to back: the worker may not even have started)
            time.sleep(2.5)
            polled = pub.call_count
            time.sleep(1.5)
            self.assertEqual(pub.call_count, polled, "none of the five waits is still polling")
        self.assertEqual([m for m in self.mcp.silence(0.3) if "id" in m], [], "and none of them was answered")
        self.assertEqual(self.mcp.request("ping")["result"], {})

    def test_other_requests_are_served_while_a_wait_is_in_flight(self):
        jid = self.add_job("running", step=1, phase="denoising")
        rid = self.mcp.submit("tools/call", {"name": "wait_for_job", "arguments": {"job_id": jid, "timeout_seconds": 5}})
        t0 = time.monotonic()
        self.assertEqual(self.mcp.request("ping")["result"], {})
        self.assertEqual(len(self.mcp.request("tools/list")["result"]["tools"]), len(TOOL_NAMES))
        self.assertFalse(self.mcp.tool("list_jobs")["isError"])
        self.assertLess(time.monotonic() - t0, 3)
        self.mcp.notify("notifications/cancelled", {"requestId": rid})
        self.mcp.silence(0.3)

    def test_several_waits_at_once(self):
        jids = [self.add_job("done", images=[self.put_png(f"mcp-par{i}.png").name], seconds=1.0) for i in range(5)]
        rids = [self.mcp.submit("tools/call", {"name": "wait_for_job", "arguments": {"job_id": j, "timeout_seconds": 10}}) for j in jids]
        for rid, j in zip(rids, jids):
            self.assertEqual(self.mcp.response(rid)["result"]["structuredContent"]["job_id"], j)

    def test_eof_interrupts_a_wait_and_the_server_exits_promptly(self):
        jid = self.add_job("running", step=1, phase="denoising")
        proc = self.start(self.live.port)
        rid = proc.submit("tools/call", {"name": "wait_for_job", "arguments": {"job_id": jid, "timeout_seconds": 50}})
        time.sleep(1.0)
        t0 = time.monotonic()
        self.assertEqual(proc.close(), 0)
        self.assertLess(time.monotonic() - t0, 4, "a 50 s wait must not hold the exit")
        self.assertNotIn(rid, [m.get("id") for m in proc.others + list(proc.inbox.queue)], "no answer is written for an interrupted call")

    def test_everything_piped_in_before_eof_is_answered(self):
        proc = McpProc(self.live.port)
        self.addCleanup(proc.kill)
        lines = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "pipe"}}},
                 {"jsonrpc": "2.0", "method": "notifications/initialized"}, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                 {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "list_models", "arguments": {}}}]
        for m in lines:
            proc.send(m)
        self.assertEqual(proc.close(), 0)
        ids = sorted(m["id"] for m in list(proc.inbox.queue) + proc.others if "id" in m)
        self.assertEqual(ids, [1, 2, 3])
        proc.assert_pure()


class Submitting(LiveCase):
    def setUp(self):
        super().setUp()
        self.make_ready()

    def new_job(self, r: dict) -> dict:
        self.assertFalse(r["isError"], text_of(r))
        d = r["structuredContent"]
        self.assertIn(d["job_id"], server.JOBS)
        return server.JOBS[d["job_id"]]

    def test_generate_returns_at_once_with_a_job_and_an_estimate(self):
        t0 = time.monotonic()
        r = self.mcp.tool("generate_image", {"prompt": "a red fox in snow"})
        self.assertLess(time.monotonic() - t0, 5)
        job = self.new_job(r)
        d = r["structuredContent"]
        self.assertEqual((job["mode"], job["status"], job["width"], job["height"], job["prompt"], len(job["seeds"])), ("generate", "queued", 768, 768, "a red fox in snow", 1))
        self.assertEqual(job["model"], "qwen21-turbo-q8")
        self.assertEqual(d["status"], "queued")
        self.assertGreater(d["estimated_seconds"], 60)
        self.assertEqual(d["next_action"], "wait_for_job")
        self.assertEqual(d["requested"]["width"], 768)
        text = text_of(r)
        for part in (d["job_id"], "768x768", "wait_for_job", "estimate", "Do not submit the same request again"):
            self.assertIn(part, text)
        self.assertEqual(images_of(r), [])

    def test_sizes_come_from_the_clis_own_rules(self):
        cli = mcp.load_cli_module()
        cases = [({}, (768, 768)), ({"quality": "standard"}, cli.dims("1:1", "standard")), ({"aspect_ratio": "16:9"}, cli.dims("16:9", "fast")),
                 ({"aspect_ratio": "9:16", "quality": "standard"}, cli.dims("9:16", "standard")), ({"width": 512, "height": 512}, (512, 512)), ({"width": "640", "height": "512"}, (640, 512))]
        for args, want in cases:
            with self.subTest(args):
                job = self.new_job(self.mcp.tool("generate_image", {"prompt": "p", **args}))
                self.assertEqual((job["width"], job["height"]), want)
        self.assertEqual(cli.dims("16:9", "fast"), (1024, 576))

    def test_count_seed_transparency_and_model_choice(self):
        job = self.new_job(self.mcp.tool("generate_image", {"prompt": "a logo", "count": 3, "seed": 7, "transparent": True, "model": "qwen"}))
        self.assertEqual((job["seeds"], job["transparent"], job["model"]), ([7, 8, 9], True, "qwen21-turbo-q8"))
        r = self.mcp.tool("generate_image", {"prompt": "x", "model": "does-not-exist"})
        self.assertTrue(r["isError"])
        self.assertIn("no model matches 'does-not-exist'", text_of(r))
        self.assertIn("list_models", text_of(r))

    def test_a_model_without_transparency_is_flagged(self):
        server.REG.add({"id": "mcp-plain", "name": "Plain", "family": "z-image", "variant": "turbo", "status": "ready", "size": 1, "files": {}})
        r = self.mcp.tool("generate_image", {"prompt": "logo", "model": "mcp-plain", "transparent": True})
        self.assertFalse(r["isError"], text_of(r))
        self.assertIn("cannot draw transparent", text_of(r))
        self.assertFalse(self.new_job(r)["transparent"])

    def test_automatic_upscale_is_chained(self):
        r = self.mcp.tool("generate_image", {"prompt": "x", "quality": "fast", "upscale_to": "4k", "upscale_model": "fast"})
        job = self.new_job(r)
        self.assertEqual(job["then"], {"model": "fast", "target": "4k"})
        self.assertGreater(r["structuredContent"]["estimated_seconds"], job["eta"], "the upscale is part of the estimate")
        self.assertIn("upscale to 4k", text_of(r))

    def test_waiting_inline(self):
        t0 = time.monotonic()
        r = self.mcp.tool("generate_image", {"prompt": "x", "wait_seconds": 2}, timeout=30)
        took = time.monotonic() - t0
        self.assertTrue(1.5 <= took < 10, took)
        self.assertFalse(r["isError"])
        self.assertEqual(r["structuredContent"]["status"], "queued", "nothing can finish here: no worker")
        self.assertIn("Still queued", text_of(r))
        self.assertIn("wait_for_job", text_of(r))

    def test_waiting_inline_returns_the_images_when_the_job_ends(self):
        image = self.put_png("mcp-quick.png").name

        def finish():
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                fresh = [j for j in list(server.JOBS.values()) if j.get("prompt") == "quick one" and j["status"] == "queued"]
                if fresh:
                    time.sleep(0.8)
                    fresh[0].update(status="done", images=[image], seconds=3.0)
                    with server.LOCK:
                        server.QUEUE.clear()
                    return
                time.sleep(0.05)

        threading.Thread(target=finish, daemon=True).start()
        r = self.mcp.tool("generate_image", {"prompt": "quick one", "wait_seconds": 20})
        self.assertEqual(r["structuredContent"]["status"], "done")
        self.assertIn("mcp-quick.png", text_of(r))
        self.assertEqual(len(images_of(r)), 1)

    def test_heavy_and_slow_requests_get_warnings(self):
        r = self.mcp.tool("generate_image", {"prompt": "x", "quality": "max"})
        self.assertFalse(r["isError"], text_of(r))
        self.assertIn("heavy", text_of(r))
        self.assertIn("smaller quality", text_of(r))

    def test_the_queue_position_is_reported(self):
        running = self.add_job("running")
        self.patch(CURRENT={"id": running, "proc": None})
        queued = len(server.QUEUE)
        r = self.mcp.tool("generate_image", {"prompt": "x"})
        self.assertEqual(r["structuredContent"]["jobs_ahead"], 1 + queued)
        self.assertIn(f"{1 + queued} job", text_of(r))

    def test_bad_arguments_never_create_a_job(self):
        before = self.jobs_now()
        bad = [{}, {"prompt": ""}, {"prompt": "x", "width": 768}, {"prompt": "x", "height": 768}, {"prompt": "x", "width": 1000, "height": 1000}, {"prompt": "x", "width": 4096, "height": 4096},
               {"prompt": "x", "width": 768, "height": 768, "aspect_ratio": "16:9"}, {"prompt": "x", "aspect_ratio": "wide"}, {"prompt": "x", "aspect_ratio": "1:0"}, {"prompt": "x", "quality": "ultra"},
               {"prompt": "x", "count": 9}, {"prompt": "x", "wait_seconds": 51}, {"prompt": "x", "wait_seconds": -1}, {"prompt": "x", "upscale_to": "16k"}, {"prompt": "x", "bogus": 1},
               {"prompt": "x" * 4001}, {"prompt": "x", "width": 2752, "height": 2752}, {"prompt": "x", "aspect_ratio": "1000:1"}]
        for args in bad:
            with self.subTest(args):
                r = self.mcp.tool("generate_image", args)
                self.assertTrue(r["isError"], args)
                self.assertTrue(text_of(r))
        self.assertEqual(self.jobs_now(), before)

    def test_an_edit(self):
        if not HAS_CONVERTER:
            self.skipTest("needs macOS sips or Pillow, like the CLI")
        src = SANDBOX / "photo.png"
        src.write_bytes(png_bytes(96, 64))
        self.addCleanup(src.unlink)
        r = self.mcp.tool("edit_image", {"image_path": str(src), "instruction": "make the sky purple"})
        job = self.new_job(r)
        self.assertEqual((job["mode"], job["prompt"], job["model"], len(job["refs"])), ("edit", "make the sky purple", "qwen21-turbo-q8", 1))
        self.assertTrue(job["refs"][0].startswith("_uploads/"), job["refs"])
        self.assertTrue((server.OUT / job["refs"][0]).is_file(), "the picture was staged in the library")
        self.assertEqual((job["width"], job["height"]), mcp.load_cli_module().dims("96:64", "fast", edit=True), "the CLI's own edit size")
        self.assertLessEqual(job["width"] * job["height"], 512 * 512 * 1.3, "edit tiers are small")
        self.assertEqual(len(job["seeds"]), 1)
        self.assertIn("wait_for_job", text_of(r))
        self.assertTrue(src.exists(), "the original is never touched")

    def test_an_edit_with_extra_references(self):
        if not HAS_CONVERTER:
            self.skipTest("needs macOS sips or Pillow")
        paths = []
        for i, color in enumerate(((200, 10, 10), (10, 200, 10), (10, 10, 200))):
            p = SANDBOX / f"ref{i}.png"
            p.write_bytes(png_bytes(64, 64, color=color))
            self.addCleanup(p.unlink)
            paths.append(str(p))
        job = self.new_job(self.mcp.tool("edit_image", {"image_path": paths[0], "instruction": "combine", "extra_references": paths[1:], "seed": 3}))
        self.assertEqual(len(set(job["refs"])), 3)
        self.assertEqual(job["seeds"], [3])
        too_many = self.mcp.tool("edit_image", {"image_path": paths[0], "instruction": "x", "extra_references": paths})
        self.assertTrue(too_many["isError"])

    def test_edit_needs_an_edit_capable_model(self):
        self.patch(model_can_edit=lambda m: False)
        src = SANDBOX / "e.png"
        src.write_bytes(png_bytes(32, 32))
        self.addCleanup(src.unlink)
        r = self.mcp.tool("edit_image", {"image_path": str(src), "instruction": "x"})
        self.assertTrue(r["isError"])
        self.assertIn("cannot edit", text_of(r))

    def test_input_pictures_are_checked_before_anything_is_sent(self):
        d = Path(tempfile.mkdtemp(prefix="mcp-bad-", dir=S.HOME))
        self.addCleanup(shutil.rmtree, d, True)
        (d / "t.txt").write_text("hello")
        (d / "fake.png").write_bytes(b"text pretending to be a picture")
        before = self.jobs_now()
        uploads = set((server.OUT / "_uploads").glob("*")) if (server.OUT / "_uploads").exists() else set()
        for tool_name, base in (("edit_image", {"instruction": "x"}), ("upscale_image", {"target": "2x"}), ("export_image", {"format": "png"})):
            for path, text in ((str(d / "missing.png"), "no such file"), (str(d), "not a file"), (str(d / "t.txt"), "not an image file"), (str(d / "fake.png"), "does not contain a picture"),
                               ("/etc/hosts", "not an image file"), ("../../../etc/passwd", "image_path"), ("", "image_path")):
                with self.subTest(tool=tool_name, path=path):
                    r = self.mcp.tool(tool_name, {"image_path": path, **base})
                    self.assertTrue(r["isError"])
                    self.assertIn(text, text_of(r))
        self.assertEqual(self.jobs_now(), before)
        after = set((server.OUT / "_uploads").glob("*")) if (server.OUT / "_uploads").exists() else set()
        self.assertEqual(after, uploads, "nothing was uploaded")

    def test_upscale_a_picture_from_anywhere_and_one_from_the_library(self):
        if not HAS_CONVERTER:
            self.skipTest("needs macOS sips or Pillow")
        outside = SANDBOX / "small.png"
        outside.write_bytes(png_bytes(64, 48))
        self.addCleanup(outside.unlink)
        home_relative = "~/small.png" if POSIX else str(outside)
        r = self.mcp.tool("upscale_image", {"image_path": home_relative, "target": "2x"})
        job = self.new_job(r)
        self.assertEqual((job["mode"], job["model"], job["target"], (job["width"], job["height"])), ("upscale", "quality", "2x", (128, 96)))
        self.assertTrue(job["src"].startswith("_uploads/"))
        self.assertEqual(r["structuredContent"]["requested"]["result"], "128x96")
        inside = self.put_png("mcp-lib.png", png_bytes(100, 50))
        job2 = self.new_job(self.mcp.tool("upscale_image", {"image_path": str(inside), "target": "4x", "model": "fast"}))
        self.assertEqual((job2["src"], job2["model"], (job2["width"], job2["height"])), ("mcp-lib.png", "fast", (400, 200)), "a PNG already in the library is used in place")

    def test_upscale_limits_and_missing_weights(self):
        inside = self.put_png("mcp-lim.png", png_bytes(2000, 1000))
        for target in ("8k", "4x"):
            self.assertFalse(self.mcp.tool("upscale_image", {"image_path": str(inside), "target": target})["isError"], f"{target} is within the 8192 px limit")
        self.patch(upscale_available=lambda: {"quality": False, "fast": True, "illustration": False})
        r = self.mcp.tool("upscale_image", {"image_path": str(inside), "target": "2x"})
        self.assertTrue(r["isError"])
        self.assertIn("'quality' upscaler weights are not installed", text_of(r))
        self.assertIn("fast", text_of(r), "the ones that are installed are named")

    def test_upscale_beyond_the_limit(self):
        inside = self.put_png("mcp-huge.png", png_bytes(3000, 100))
        r = self.mcp.tool("upscale_image", {"image_path": str(inside), "target": "4x"})
        self.assertTrue(r["isError"])
        self.assertIn("8192", text_of(r))

    @unittest.skipUnless(HAS_PIL, "the in-process server needs Pillow to convert")
    def test_export_converts_and_returns_the_new_file(self):
        src = SANDBOX / "logo.png"
        src.write_bytes(png_bytes(120, 80))
        self.addCleanup(src.unlink)
        for fmt in ("webp", "jpeg", "png"):
            with self.subTest(fmt):
                r = self.mcp.tool("export_image", {"image_path": str(src), "format": fmt, "quality": 80})
                self.assertFalse(r["isError"], text_of(r))
                d = r["structuredContent"]
                out = Path(d["path"])
                self.assertTrue(out.is_file() and out.parent.name == "_exports" and out.resolve().parent.parent == server.OUT.resolve(), out)
                self.assertEqual((d["format"], d["width"], d["height"]), (fmt, 120, 80))
                self.assertGreater(d["bytes"], 0)
                self.assertIn(str(out), text_of(r))
                self.assertEqual(out.suffix, {"jpeg": ".jpg"}.get(fmt, "." + fmt))
        self.assertTrue(src.exists())

    @unittest.skipUnless(HAS_PIL, "the in-process server needs Pillow to convert")
    def test_export_of_a_library_picture_in_place_and_resizing(self):
        inside = self.put_png("mcp-exp.png", png_bytes(200, 100))
        r = self.mcp.tool("export_image", {"image_path": str(inside), "format": "webp", "lossless": True, "max_side": 100})
        d = r["structuredContent"]
        self.assertEqual((d["width"], d["height"]), (100, 50))
        self.assertTrue(Path(d["path"]).name.startswith("mcp-exp"), "named after the library file: it was not copied first")

    def test_export_svg(self):
        src = self.put_png("mcp-svg.png", png_bytes(60, 40))
        r = self.mcp.tool("export_image", {"image_path": str(src), "format": "svg", "preset": "logo"})
        try:
            import vtracer  # noqa: F401
            have = HAS_PIL
        except ImportError:
            have = False
        if have:
            self.assertFalse(r["isError"], text_of(r))
            self.assertTrue(Path(r["structuredContent"]["path"]).name.endswith(".svg"))
            self.assertEqual(images_of(r), [], "no preview for a vector file")
        else:
            self.assertTrue(r["isError"])
            self.assertTrue("vtracer" in text_of(r) or "Pillow" in text_of(r), text_of(r))

    def test_export_validates_its_options(self):
        src = self.put_png("mcp-opt.png")
        for args in ({"format": "gif"}, {"format": "png", "quality": 10}, {"format": "png", "max_side": -1}, {"format": "svg", "preset": "fancy"}, {}):
            with self.subTest(args):
                self.assertTrue(self.mcp.tool("export_image", {"image_path": str(src), **args})["isError"])


class NotReady(LiveCase):
    """A fresh install: no engine, no model. The tools must say what the user has to do and create nothing."""

    def check(self, r: dict, *words: str):
        self.assertTrue(r["isError"], text_of(r))
        text = text_of(r)
        for w in words:
            self.assertIn(w, text)
        self.assertNotIn("Traceback", text)

    def test_generate_without_an_engine(self):
        before = self.jobs_now()
        self.check(self.mcp.tool("generate_image", {"prompt": "a red fox"}), "engine", "imageloom ui", "Only the user can finish the setup")
        self.assertEqual(self.jobs_now(), before)

    def test_generate_with_an_engine_but_no_model(self):
        engine = S.HOME / "fake-engine2" / "sd-cli"
        engine.parent.mkdir(exist_ok=True)
        engine.write_bytes(b"x")
        self.patch(find_sd_cli=lambda: engine)
        before = self.jobs_now()
        self.check(self.mcp.tool("generate_image", {"prompt": "a red fox"}), "no model is installed", "add_model")
        self.assertEqual(self.jobs_now(), before)

    def test_edit_and_upscale_without_an_engine(self):
        src = SANDBOX / "n.png"
        src.write_bytes(png_bytes(32, 32))
        self.addCleanup(src.unlink)
        self.check(self.mcp.tool("edit_image", {"image_path": str(src), "instruction": "x"}), "engine")
        self.check(self.mcp.tool("upscale_image", {"image_path": str(src), "target": "2x"}), "engine")


class OldProtocol(LiveCase):
    version = "2024-11-05"

    def test_no_structured_content_for_old_clients_but_the_text_has_everything(self):
        r = self.mcp.tool("list_models")
        self.assertNotIn("structuredContent", r)
        self.assertIn("qwen21-turbo-q8", text_of(r))
        self.assertIs(r["isError"], False)

    def test_structured_content_is_back_from_2025_06_18_on_only(self):
        for v, want in (("2025-03-26", False), ("2025-06-18", True)):
            with self.subTest(v):
                self.assertEqual("structuredContent" in self.start(self.live.port, version=v).tool("list_models"), want)

    def test_progress_has_no_message_field_before_2025_03_26(self):
        jid = self.add_job("running", step=1, steps=8, phase="denoising")
        self.mcp.tool("wait_for_job", {"job_id": jid, "timeout_seconds": 1}, meta={"progressToken": "t"})
        notes = [m for m in self.mcp.others if m.get("method") == "notifications/progress"]
        self.assertTrue(notes)
        self.assertTrue(all("message" not in n["params"] for n in notes))


class StartsTheServer(Proc):
    """Nothing is listening: like the CLI, the first tool that needs ImageLoom starts its server, and restarts it after it quit."""

    @unittest.skipUnless(POSIX, "uses signals and POSIX paths")
    def test_the_server_is_started_on_demand_and_again_after_it_quit(self):
        home = Path(tempfile.mkdtemp(prefix="mcp-auto-")).resolve()
        self.addCleanup(shutil.rmtree, home, True)
        data, out, port = home / "data", home / "out", free_port()
        data.mkdir()
        (data / "config.json").write_text(json.dumps({"output_dir": str(out)}), encoding="utf-8")
        env = dict(HOME=str(home), USERPROFILE=str(home), IMAGELOOM_HOME=str(data), PATH="/usr/bin:/bin")
        self.addCleanup(self.stop_server, data, port)
        proc = self.start(port, **env)
        self.assertFalse(self.alive(port))
        r = proc.tool("list_models", timeout=60)
        self.assertFalse(r["isError"], text_of(r))
        self.assertTrue(self.alive(port), "the tool started the server on the port it was told to use")
        info = json.loads((data / "server.json").read_text())
        self.assertEqual(info["port"], port)
        self.assertIn("qwen21-turbo-q8", text_of(r))
        self.assertTrue(out.is_dir(), "its library was created in the sandbox, not in a real Pictures folder")
        r = proc.tool("generate_image", {"prompt": "x"}, timeout=60)
        self.assertTrue(r["isError"], "nothing is installed in the sandbox: no job is created")
        self.assertTrue("engine" in text_of(r) or "no model is installed" in text_of(r), text_of(r))
        self.post(port, "/api/quit")
        deadline = time.time() + 15
        while self.alive(port) and time.time() < deadline:
            time.sleep(0.1)
        self.assertFalse(self.alive(port))
        r = proc.tool("list_models", timeout=60)
        self.assertFalse(r["isError"], "the server is started again")
        self.assertTrue(self.alive(port))

    def alive(self, port: int) -> bool:
        try:
            return self.get(port, "/api/status") == 200
        except OSError:
            return False

    def get(self, port: int, path: str) -> int:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            c.request("GET", path, headers={"Host": f"127.0.0.1:{port}"})
            return c.getresponse().status
        finally:
            c.close()

    def post(self, port: int, path: str) -> None:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            c.request("POST", path, body=b"{}", headers={"Host": f"127.0.0.1:{port}", "X-ImageLoom": "1", "Content-Type": "application/json"})
            c.getresponse().read()
        except OSError:
            pass
        finally:
            c.close()

    def stop_server(self, data: Path, port: int):
        with contextlib.suppress(OSError):
            self.post(port, "/api/quit")
        try:
            pid = json.loads((data / "server.json").read_text())["pid"]
        except (OSError, ValueError, KeyError):
            return
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                return
            time.sleep(0.1)
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)


class InProcessProtocol(unittest.TestCase):
    """The server object on in-memory streams: protocol corners that need no subprocess."""

    def run_lines(self, *msgs) -> list:
        out = io.BytesIO()
        srv = mcp.McpServer(io.BytesIO(b"".join(json.dumps(m).encode() + b"\n" for m in msgs)), out)
        self.assertEqual(srv.run(), 0)
        return [json.loads(line) for line in out.getvalue().splitlines()]

    def test_a_round_trip(self):
        with contextlib.redirect_stderr(io.StringIO()):
            got = self.run_lines({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}}, {"jsonrpc": "2.0", "id": 2, "method": "ping"})
        self.assertEqual([m["id"] for m in got], [1, 2])
        self.assertEqual(got[0]["result"]["protocolVersion"], "2025-03-26")

    def test_an_unexpected_exception_is_32603(self):
        with mock.patch.object(mcp.McpServer, "initialize", side_effect=KeyError("boom")), contextlib.redirect_stderr(io.StringIO()):
            got = self.run_lines({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}})
        self.assertEqual(got[0]["error"]["code"], -32603)

    def test_the_writer_going_away_ends_the_loop(self):
        class Gone:
            def write(self, data):
                raise BrokenPipeError

            def flush(self):
                pass

        srv = mcp.McpServer(io.BytesIO(b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n' * 3), Gone())
        self.assertEqual(srv.run(), 0)
        self.assertTrue(srv.stopping.is_set())


if __name__ == "__main__":
    unittest.main()
