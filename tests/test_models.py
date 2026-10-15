"""models.py: family detection, quant parsing, default pick for a 16 GB machine, Hugging Face link handling, registry."""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
from _support import M, fake_hf, fixture  # noqa: E402  (also puts core/ on sys.path and installs the no-network guard)


class ParseUrl(unittest.TestCase):
    def test_accepted_forms(self):
        ok = {
            "owner/repo": ("owner/repo", None),
            "  owner/repo  ": ("owner/repo", None),
            "https://huggingface.co/owner/repo": ("owner/repo", None),
            "https://huggingface.co/owner/repo/": ("owner/repo", None),
            "https://www.huggingface.co/owner/repo": ("owner/repo", None),
            "HTTPS://HuggingFace.co/Owner/Repo.v2": ("Owner/Repo.v2", None),
            "huggingface.co/owner/repo": ("owner/repo", None),
            "https://huggingface.co/owner/repo?download=true": ("owner/repo", None),
            "https://huggingface.co/owner/repo/tree/main": ("owner/repo", None),
            "https://huggingface.co/owner/repo/blob/main/dir/model-Q4_0.gguf": ("owner/repo", "dir/model-Q4_0.gguf"),
            "https://huggingface.co/owner/repo/resolve/main/model%20x.gguf?download=true": ("owner/repo", "model x.gguf"),
        }
        for text, want in ok.items():
            with self.subTest(text):
                self.assertEqual(M.parse_hf_url(text), want)

    def test_rejected_forms(self):
        bad = [
            "", "   ", None, 5, [], "owner", "a/b/c", "../etc", "a/../b", "./a/b", "-x/y", ".hidden/y",
            "http://huggingface.co/owner/repo",  # plain http
            "ftp://huggingface.co/owner/repo", "file:///etc/passwd",
            "https://evil.com/owner/repo", "https://evil.com/huggingface.co/owner/repo",
            "https://huggingface.co.evil.com/owner/repo", "https://evilhuggingface.co/owner/repo", "https://hf.co/owner/repo",
            "https://user:pw@huggingface.co/owner/repo",  # userinfo
            "https://huggingface.co@evil.com/owner/repo", "https://huggingface.co:443/owner/repo", "https://huggingface.co:8443/owner/repo",
            "https://127.0.0.1/owner/repo", "https://[::1]/owner/repo", "https://localhost/owner/repo",
            "https://huggingface.co/datasets/owner/repo", "https://huggingface.co/spaces/owner/repo", "https://huggingface.co/owner",
            "https://huggingface.co/owner/repo/discussions", "https://huggingface.co/owner/repo/%2e%2e",
            "https://huggingface.co/owner/repo\nHost: evil.com", "https://huggingface.co/owner/repo evil", "https://huggingface.co\\@evil.com/o/r",
            "owner/re\npo", "own er/repo", "owner/rep o", "x" * 700,
        ]
        for text in bad:
            with self.subTest(text):
                with self.assertRaises(ValueError):
                    M.parse_hf_url(text)


class FileNames(unittest.TestCase):
    def test_safe_filename_keeps_one_harmless_component(self):
        cases = {
            "model-Q4_0.gguf": "model-Q4_0.gguf",
            "dir/sub/model.gguf": "model.gguf",
            "../../evil.gguf": "evil.gguf",
            "..\\..\\evil.gguf": "evil.gguf",
            "/etc/passwd": "passwd",
            "C:\\Windows\\system32\\x.gguf": "x.gguf",
            "-rf.gguf": "rf.gguf",  # never looks like an option
            ".hidden.gguf": "hidden.gguf",
            "..": "file",
            "": "file",
            "sp ace&semi;.gguf": "sp_ace_semi_.gguf",
            "ünï.gguf": "_n_.gguf",
            "CON.gguf": "_CON.gguf",
            "nul": "_nul",
        }
        for raw, want in cases.items():
            with self.subTest(raw):
                got = M.safe_filename(raw)
                self.assertEqual(got, want)
                self.assertNotIn("/", got)
                self.assertNotIn("\\", got)
                self.assertFalse(got.startswith((".", "-")))

    def test_long_names_keep_their_extension(self):
        got = M.safe_filename("a" * 400 + ".gguf")
        self.assertTrue(got.endswith(".gguf"))
        self.assertLessEqual(len(got), 170)

    def test_clean_repo_path(self):
        for p in ("a.gguf", "dir/a.gguf", "split_files/vae/ae.safetensors"):
            self.assertTrue(M.clean_repo_path(p), p)
        for p in ("../a.gguf", "a/../b.gguf", "/abs.gguf", "./a.gguf", "a//b.gguf", "a\\b.gguf", "a\x00b.gguf", "a\nb", "", None, 5, "x" * 400):
            self.assertFalse(M.clean_repo_path(p), repr(p))


class Quants(unittest.TestCase):
    def test_quant_of(self):
        cases = {
            "Qwen-Image-2.1-Turbo-Q8_0.gguf": "Q8_0", "model-Q4_K_M.gguf": "Q4_K_M", "flux1-dev-Q5_K_S.gguf": "Q5_K_S", "unet/a-Q3_K_L.gguf": "Q3_K_L",
            "z_image_turbo-Q4_K.gguf": "Q4_K", "a.Q6_K.gguf": "Q6_K", "x-F16.gguf": "F16", "x-bf16.safetensors": "BF16", "model_fp8_e4m3fn.safetensors": "FP8",
            "model-IQ4_XS.gguf": "IQ4_XS", "Q8_0.gguf": "Q8_0", "x.safetensors": "", "v1-5-pruned-emaonly.safetensors": "", "squash.gguf": "",
        }
        for name, want in cases.items():
            with self.subTest(name):
                self.assertEqual(M.quant_of(name), want)

    def test_classify(self):
        cases = {
            "flux1-dev-Q4_0.gguf": "diffusion", "sd_xl_base_1.0.safetensors": "diffusion", "ae.safetensors": "vae", "vae/x.safetensors": "vae", "flux-vae.gguf": "vae",
            "mmproj-Qwen3VL-F16.gguf": "llm_vision", "t5xxl_fp8_e4m3fn.safetensors": "text_encoder", "Qwen3-4B-Q4_K_M.gguf": "text_encoder",
            "lora-x.safetensors": None, "a/b/c-controlnet.safetensors": None, "model-00001-of-00003.safetensors": None, "README.md": None, "model.bin": None,
        }
        for name, want in cases.items():
            with self.subTest(name):
                self.assertEqual(M.classify(name), want)


class FamilyDetection(unittest.TestCase):
    def test_from_repo_names(self):
        cases = {
            "someone/Qwen-Image-2.1-Turbo-GGUF": ("qwen-image-2.1", "turbo"), "a/Qwen-Image-2.1": ("qwen-image-2.1", "base"),
            "x/z-image-turbo-gguf": ("z-image", "turbo"), "x/Z_Image_Base": ("z-image", "base"),
            "city96/FLUX.1-dev-gguf": ("flux1", "dev"), "x/FLUX.1-schnell": ("flux1", "schnell"),
            "stabilityai/sdxl-base": ("sdxl", "base"), "x/stable-diffusion-xl-base": ("sdxl", "base"),
            "runwayml/stable-diffusion-v1-5": ("sd15", "base"), "x/sd-1.5-pruned": ("sd15", "base"),
            "x/flux2-dev": ("custom", "base"), "x/mystery": ("custom", "base"),
        }
        for repo, want in cases.items():
            with self.subTest(repo):
                self.assertEqual(M.detect_family(repo, [], []), want)

    def test_from_file_names_and_tags(self):
        files = [{"path": "z_image_turbo-Q4_K.gguf", "size": 1}]
        self.assertEqual(M.detect_family("x/unknown", [], files), ("z-image", "turbo"))
        self.assertEqual(M.detect_family("x/unknown", ["flux"], []), ("flux1", "dev"))

    def test_recipes_are_complete(self):
        for key, fam in M.FAMILIES.items():
            with self.subTest(key):
                self.assertTrue(set(fam["required"]) <= set(fam["flags"]) | {"checkpoint"})
                for variant in fam["variants"]:
                    self.assertEqual(len(fam["efforts"].get(variant) or next(iter(fam["efforts"].values()))), 4)
                for level in next(iter(fam["efforts"].values())):
                    self.assertEqual(level["px"] % fam["multiple"], 0, f"{key}: effort sizes must fit the model's size multiple")
                    self.assertEqual(level["px_edit"] % fam["multiple"], 0)


class DefaultPick(unittest.TestCase):
    @staticmethod
    def cands(*pairs):
        return [{"path": f"m-{q}.gguf", "size": int(gb * 1e9), "quant": q} for q, gb in pairs]

    def test_biggest_quant_that_fits(self):
        c = self.cands(("Q4_K_M", 4.9), ("Q5_K_M", 5.8), ("Q8_0", 7.6), ("F16", 14.2))
        self.assertEqual(M.pick_default(c)["quant"], "Q8_0")  # F16 is too big for a 16 GB machine next to the encoder

    def test_skips_what_does_not_fit_the_budget(self):
        c = self.cands(("Q2_K", 4.0), ("Q4_0", 6.8), ("Q5_K_S", 8.3), ("Q6_K", 9.9), ("Q8_0", 12.7))
        self.assertEqual(M.pick_default(c)["quant"], "Q5_K_S")

    def test_smallest_when_nothing_fits(self):
        c = self.cands(("Q8_0", 12.7), ("Q4_0", 11.0), ("F16", 24.0))
        self.assertEqual(M.pick_default(c)["quant"], "Q4_0")

    def test_bigger_budget_allows_bigger_quant(self):
        c = self.cands(("Q4_0", 6.8), ("Q8_0", 12.7))
        self.assertEqual(M.pick_default(c, budget=14e9)["quant"], "Q8_0")

    def test_unknown_quant_falls_back_to_biggest_fitting_file(self):
        c = [{"path": "a.safetensors", "size": int(4e9), "quant": ""}, {"path": "b.safetensors", "size": int(7e9), "quant": ""}]
        self.assertEqual(M.pick_default(c)["path"], "b.safetensors")

    def test_prefers_clean_vae_over_baked_in_09vae(self):
        c = [{"path": "m-0.9vae-Q8_0.gguf", "size": int(7e9), "quant": "Q8_0"}, {"path": "m-Q6_K.gguf", "size": int(6e9), "quant": "Q6_K"}]
        self.assertEqual(M.pick_default(c)["path"], "m-Q6_K.gguf")

    def test_no_candidates(self):
        self.assertIsNone(M.pick_default([]))


class Inspect(unittest.TestCase):
    def inspect(self, url, routes):
        with mock.patch.object(M, "hf_json", fake_hf(routes)):
            return M.inspect_repo(url)

    QWEN: ClassVar[dict] = {"someone/Qwen-Image-2.1-Turbo-GGUF/tree": "qwen21_tree.json", "someone/Qwen-Image-2.1-Turbo-GGUF": "qwen21_info.json",
            "Qwen/Qwen3-VL-8B-Instruct-GGUF/tree": "qwen3vl_tree.json", "Comfy-Org/Qwen-Image-2.1/tree": "comfy_qwen21_tree.json"}

    def test_qwen_gguf_repo(self):
        plan = self.inspect("https://huggingface.co/someone/Qwen-Image-2.1-Turbo-GGUF", self.QWEN)
        self.assertEqual((plan["family"], plan["variant"], plan["supported"], plan["experimental"]), ("qwen-image-2.1", "turbo", True, False))
        paths = [c["path"] for c in plan["candidates"]]
        self.assertEqual(paths, ["Qwen-Image-2.1-Turbo-Q4_K_M.gguf", "Qwen-Image-2.1-Turbo-Q5_K_M.gguf", "Qwen-Image-2.1-Turbo-Q8_0.gguf", "Qwen-Image-2.1-Turbo-F16.gguf"])
        self.assertEqual(plan["default"], "Qwen-Image-2.1-Turbo-Q8_0.gguf")
        self.assertEqual(plan["license"], "other")
        self.assertEqual({c["role"]: c["size"] > 0 for c in plan["companions"]}, {"llm": True, "vae": True, "llm_vision": True})
        self.assertTrue(next(c for c in plan["companions"] if c["role"] == "llm_vision")["optional"])
        for c in plan["companions"]:
            self.assertTrue(c["url"].startswith("https://huggingface.co/"), c["url"])
        self.assertEqual(plan["warnings"], [])

    def test_default_fits_a_16gb_machine_for_flux(self):
        routes = {"city96/FLUX.1-dev-gguf/tree": "flux_tree.json", "city96/FLUX.1-dev-gguf": "flux_info.json", "comfyanonymous/flux_text_encoders/tree": "flux_companions_tree.json",
                  "Comfy-Org/z_image_turbo/tree": "flux_companions_tree.json"}
        plan = self.inspect("city96/FLUX.1-dev-gguf", routes)
        self.assertEqual((plan["family"], plan["variant"], plan["experimental"]), ("flux1", "dev", True))
        self.assertEqual(plan["default"], "flux1-dev-Q5_K_S.gguf")
        chosen = next(c for c in plan["candidates"] if c["path"] == plan["default"])
        self.assertLessEqual(chosen["size"], 9.5e9)
        self.assertEqual(sorted(c["role"] for c in plan["companions"]), ["clip_l", "t5xxl", "vae"])

    def test_picked_file_in_the_link_narrows_the_candidates(self):
        plan = self.inspect("https://huggingface.co/someone/Qwen-Image-2.1-Turbo-GGUF/blob/main/Qwen-Image-2.1-Turbo-Q5_K_M.gguf", self.QWEN)
        self.assertEqual([c["path"] for c in plan["candidates"]], ["Qwen-Image-2.1-Turbo-Q5_K_M.gguf"])

    def test_diffusers_layout_warns_and_ignores_component_folders(self):
        routes = {"stabilityai/stable-diffusion-xl-base-1.0/tree": "sdxl_diffusers_tree.json", "stabilityai/stable-diffusion-xl-base-1.0": "sdxl_diffusers_info.json"}
        plan = self.inspect("stabilityai/stable-diffusion-xl-base-1.0", routes)
        self.assertEqual(plan["family"], "sdxl")
        self.assertEqual([c["path"] for c in plan["candidates"]], ["sd_xl_base_1.0.safetensors"])  # unet/, vae/ ... are not loadable alone
        # the repo is Diffusers-tagged but ships a single-file checkpoint, so no "Diffusers" warning when a loadable file exists
        self.assertFalse([w for w in plan["warnings"] if "no usable" in w])

    def test_pure_diffusers_repo_gets_the_warning(self):
        info = dict(fixture("sdxl_diffusers_info.json"))
        tree = [f for f in fixture("sdxl_diffusers_tree.json") if "/" in f["path"] or f["path"] == "model_index.json"]
        def hf(path, token=None, timeout=40):
            return tree if "/tree/" in path else info
        with mock.patch.object(M, "hf_json", hf):
            plan = M.inspect_repo("stabilityai/stable-diffusion-xl-base-1.0")
        self.assertEqual(plan["candidates"], [])
        text = " ".join(plan["warnings"])
        self.assertIn("Diffusers", text)
        self.assertIn("no usable weight files", text)

    def test_gated_repo_warns_about_the_token(self):
        routes = {"black-forest-labs/FLUX.1-dev/tree": "gated_tree.json", "black-forest-labs/FLUX.1-dev": "gated_info.json", "comfyanonymous/flux_text_encoders/tree": "flux_companions_tree.json",
                  "Comfy-Org/z_image_turbo/tree": "flux_companions_tree.json"}
        plan = self.inspect("black-forest-labs/FLUX.1-dev", routes)
        self.assertTrue(plan["gated"])
        self.assertTrue(any("gated" in w and "token" in w for w in plan["warnings"]), plan["warnings"])

    def test_repo_without_weights(self):
        plan = self.inspect("a/docs-only", {"a/docs-only/tree": "empty_tree.json", "a/docs-only": "empty_info.json"})
        self.assertEqual((plan["candidates"], plan["default"], plan["family"], plan["supported"]), ([], None, "custom", False))
        self.assertTrue(any("no usable weight files" in w for w in plan["warnings"]))

    def test_hostile_file_names_never_become_candidates(self):
        plan = self.inspect("evil/model-GGUF", {"evil/model-GGUF/tree": "hostile_tree.json", "evil/model-GGUF": "hostile_info.json"})
        self.assertEqual([c["path"] for c in plan["candidates"]], ["good-model-Q4_0.gguf"])

    def test_sd15_single_file(self):
        plan = self.inspect("runwayml/stable-diffusion-v1-5", {"runwayml/stable-diffusion-v1-5/tree": "sd15_tree.json", "runwayml/stable-diffusion-v1-5": "sd15_info.json"})
        self.assertEqual(plan["family"], "sd15")
        self.assertEqual(plan["default"], "v1-5-pruned.safetensors")  # no quant names to rank: the biggest file that fits the 9.5 GB budget
        self.assertEqual(plan["companions"], [])

    def test_zimage(self):
        routes = {"leejet/Z-Image-Turbo-GGUF/tree": "zimage_tree.json", "leejet/Z-Image-Turbo-GGUF": "zimage_info.json",
                  "unsloth/Qwen3-4B-Instruct-2507-GGUF/tree": "qwen3_4b_tree.json", "Comfy-Org/z_image_turbo/tree": "comfy_zimage_tree.json"}
        plan = self.inspect("leejet/Z-Image-Turbo-GGUF", routes)
        self.assertEqual((plan["family"], plan["variant"], plan["default"]), ("z-image", "turbo", "z_image_turbo-Q8_0.gguf"))
        self.assertEqual(sorted(c["role"] for c in plan["companions"]), ["llm", "vae"])

    def test_bad_link_is_rejected_before_any_request(self):
        with mock.patch.object(M, "hf_json", side_effect=AssertionError("no request expected")):
            for url in ("http://huggingface.co/a/b", "https://evil.com/a/b", "https://user@huggingface.co/a/b"):
                with self.assertRaises(ValueError):
                    M.inspect_repo(url)


class HfJson(unittest.TestCase):
    def raises_http(self, code):
        def boom(url, headers=None, token=None, timeout=60):
            err = urllib.error.HTTPError(url, code, "x", {}, io.BytesIO(b""))
            self.addCleanup(err.close)
            raise err
        return mock.patch.object(M, "open_url", boom)

    def test_error_mapping(self):
        with self.raises_http(401), self.assertRaises(PermissionError):
            M.hf_json("/api/models/a/b")
        with self.raises_http(403), self.assertRaises(PermissionError):
            M.hf_json("/api/models/a/b")
        with self.raises_http(404), self.assertRaises(ValueError):
            M.hf_json("/api/models/a/b")
        with self.raises_http(500), self.assertRaises(urllib.error.HTTPError):
            M.hf_json("/api/models/a/b")

    def test_oversized_and_non_json_answers(self):
        class Resp(io.BytesIO):
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        with mock.patch.object(M, "open_url", lambda *a, **k: Resp(b"x" * (M.MAX_API_BYTES + 5))), self.assertRaises(ValueError):
            M.hf_json("/api/models/a/b")
        with mock.patch.object(M, "open_url", lambda *a, **k: Resp(b"<html>not json</html>")), self.assertRaises(ValueError):
            M.hf_json("/api/models/a/b")

    def test_list_files_skips_directories_and_bad_paths(self):
        tree = fixture("hostile_tree.json") + [{"type": "directory", "path": "d"}, "junk", {"type": "file"}]
        with mock.patch.object(M, "hf_json", lambda *a, **k: tree):
            self.assertEqual([f["path"] for f in M.list_files("evil/model")], ["good-model-Q4_0.gguf"])

    def test_list_files_follows_huggingface_cursor_pagination(self):
        class Resp(io.BytesIO):
            def __init__(self, body, headers=None):
                super().__init__(json.dumps(body).encode())
                self.headers = headers or {}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        next_url = f"{M.HF}/api/models/acme/large/tree/main?recursive=true&cursor=page-2"
        replies = [
            Resp(fixture("large_tree_page_1.json"), {"Link": f'<{next_url}>; rel="next"'}),
            Resp(fixture("large_tree_page_2.json")),
        ]
        with mock.patch.object(M, "open_url", side_effect=replies) as opened:
            files = M.list_files("acme/large")

        self.assertEqual([f["path"] for f in files], ["first.gguf", "second.safetensors"])
        self.assertEqual(opened.call_args_list[1].args[0], next_url)

    def test_hf_json_does_not_paginate_non_tree_responses(self):
        class Resp(io.BytesIO):
            headers = {"Link": f'<{M.HF}/api/models/acme/other?cursor=page-2>; rel="next"'}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        first_page = {"id": "acme/model", "tags": ["text-to-image"]}
        with mock.patch.object(M, "open_url", return_value=Resp(json.dumps(first_page).encode())) as opened:
            self.assertEqual(M.hf_json("/api/models/acme/model"), first_page)

        opened.assert_called_once()


# ----------------------------------------------------------------------------------------- network policy
class _Handler(BaseHTTPRequestHandler):
    """Records the headers it receives; /redirect/<port>/<path> answers 302 to another local server."""
    seen: ClassVar[list] = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        type(self).seen.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}})
        if self.path.startswith("/redirect/"):
            _, _, port, rest = self.path.split("/", 3)
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{port}/{rest}")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path.startswith("/to-http-localhost"):
            self.send_response(302)
            self.send_header("Location", "http://localhost:1/x")
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def _local_server():
    cls = type("H", (_Handler,), {"seen": []})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, cls


class RedirectPolicy(unittest.TestCase):
    """urllib re-sends custom headers (Authorization!) on a cross-host redirect; Hugging Face redirects downloads to CDNs."""

    class Local(M.SafeRedirect):  # the production handler with the https-only / no-IP rules relaxed so two local servers can talk
        schemes = ("http", "https")
        allow_local = True

    def setUp(self):
        self.a, self.ha = _local_server()
        self.b, self.hb = _local_server()
        self.opener = urllib.request.build_opener(self.Local)
        self.opener.add_handler(urllib.request.ProxyHandler({}))
        self.addCleanup(lambda: [s.shutdown() or s.server_close() for s in (self.a, self.b)])

    def fetch(self, url, **headers):
        with self.opener.open(urllib.request.Request(url, headers=headers), timeout=5) as r:
            return r.read()

    def test_authorization_is_dropped_when_the_redirect_changes_host(self):
        pa, pb = self.a.server_address[1], self.b.server_address[1]
        self.assertEqual(self.fetch(f"http://127.0.0.1:{pa}/redirect/{pb}/file", Authorization="Bearer SECRET", Cookie="c=1", **{"X-Keep": "1"}), b"ok")
        first, second = self.ha.seen[0], self.hb.seen[0]
        self.assertEqual(first["headers"]["authorization"], "Bearer SECRET")  # the original host still gets it
        self.assertNotIn("authorization", second["headers"], "the token leaked to another origin")
        self.assertNotIn("cookie", second["headers"])
        self.assertEqual(second["headers"].get("x-keep"), "1")  # other headers survive

    def test_authorization_is_kept_for_a_redirect_inside_the_same_origin(self):
        pa = self.a.server_address[1]
        self.fetch(f"http://127.0.0.1:{pa}/redirect/{pa}/other", Authorization="Bearer SECRET")
        self.assertEqual([h["headers"].get("authorization") for h in self.ha.seen], ["Bearer SECRET", "Bearer SECRET"])

    def test_production_policy_refuses_plain_http_and_local_targets(self):
        opener = urllib.request.build_opener(M.SafeRedirect)
        opener.add_handler(urllib.request.ProxyHandler({}))
        pa, pb = self.a.server_address[1], self.b.server_address[1]
        for target in (f"/redirect/{pb}/x", "/to-http-localhost"):
            with self.subTest(target), self.assertRaises(urllib.error.HTTPError) as cm:
                opener.open(urllib.request.Request(f"http://127.0.0.1:{pa}{target}", headers={"Authorization": "Bearer SECRET"}), timeout=5)
            self.addCleanup(cm.exception.close)
            self.assertIn(cm.exception.code, (301, 302))
        self.assertEqual(self.hb.seen, [], "the refused redirect must not have been followed")

    def test_token_only_goes_to_huggingface_over_https(self):
        tok = "hf_secret"
        for url, expect in {
            "https://huggingface.co/api/models/a/b": True, "https://www.huggingface.co/a/b": True, "https://HUGGINGFACE.CO/a/b": True,
            "http://huggingface.co/a/b": False, "https://huggingface.co.evil.com/a/b": False, "https://evilhuggingface.co/a/b": False,
            "https://cdn-lfs.huggingface.co/x": False, "https://github.com/x": False, "https://api.github.com/x": False, "https://127.0.0.1/x": False,
        }.items():
            with self.subTest(url):
                req = M.make_request(url, token=tok)
                self.assertEqual(req.get_header("Authorization") == f"Bearer {tok}", expect)
        self.assertIsNone(M.make_request("https://huggingface.co/x").get_header("Authorization"))

    def test_open_url_only_contacts_known_hosts(self):
        for url in ("http://huggingface.co/x", "https://evil.com/x", "https://user:pw@huggingface.co/x", "https://huggingface.co:444/x", "ftp://huggingface.co/x", "file:///etc/passwd", "https://127.0.0.1/x"):
            with self.subTest(url), self.assertRaises(ValueError):
                M.open_url(url, token="t")


# ----------------------------------------------------------------------------------------- registry
BUILTIN = {"id": "builtin", "name": "Built-in", "family": "qwen-image-2.1", "builtin": True, "files": {}}


def entry(mid, files, **extra):
    return {"id": mid, "name": mid, "family": "sdxl", "variant": "base", "files": files, "status": "ready", **extra}


class RegistryPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="reg-")).resolve()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.data = self.tmp / "data"
        (self.data / "models" / "m1").mkdir(parents=True)
        self.path = self.data / "models.json"

    def write(self, models, active=None, base=None):
        self.path.write_text(json.dumps({"models": models, "active": active or "builtin"}), encoding="utf-8")

    def on_disk(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_new_entries_are_stored_relative_and_resolved_in_memory(self):
        reg = M.Registry(self.path, BUILTIN, self.data)
        weights = self.data / "models" / "m1" / "w.safetensors"
        reg.add(entry("m1", {"checkpoint": str(weights)}))
        stored = self.on_disk()["models"][0]["files"]["checkpoint"]
        self.assertEqual(stored, "models/m1/w.safetensors")  # forward slashes, no drive letter, no home directory
        self.assertEqual(reg.get("m1")["files"]["checkpoint"], str(weights))
        self.assertTrue(os.path.isabs(M.sd_args(reg.get("m1"), "generate")[1]))

    def test_absolute_paths_inside_the_data_folder_are_migrated(self):
        inside = self.data / "models" / "m1" / "w.safetensors"
        self.write([entry("m1", {"checkpoint": str(inside), "vae": str(self.data / "models" / "_shared" / "ae.safetensors")})])
        reg = M.Registry(self.path, BUILTIN, self.data)
        self.assertEqual(self.on_disk()["models"][0]["files"], {"checkpoint": "models/m1/w.safetensors", "vae": "models/_shared/ae.safetensors"})
        self.assertEqual(reg.get("m1")["files"]["checkpoint"], str(inside))

    def test_absolute_paths_outside_keep_working(self):
        outside = self.tmp / "elsewhere" / "w.safetensors"
        self.write([entry("m1", {"checkpoint": str(outside)})])
        reg = M.Registry(self.path, BUILTIN, self.data)
        self.assertEqual(reg.get("m1")["files"]["checkpoint"], str(outside))
        self.assertEqual(self.on_disk()["models"][0]["files"]["checkpoint"], str(outside))  # untouched on disk too

    def test_moving_the_data_folder_keeps_relative_entries_working(self):
        weights = self.data / "models" / "m1" / "w.safetensors"
        weights.write_bytes(b"x" * 10)
        reg = M.Registry(self.path, BUILTIN, self.data)
        reg.add(entry("m1", {"checkpoint": str(weights)}))
        moved = self.tmp / "backup" / "ImageLoom"
        shutil.copytree(self.data, moved)
        shutil.rmtree(self.data)
        reg2 = M.Registry(moved / "models.json", BUILTIN, moved)
        self.assertEqual(reg2.get("m1")["files"]["checkpoint"], str(moved / "models" / "m1" / "w.safetensors"))
        self.assertTrue(Path(reg2.get("m1")["files"]["checkpoint"]).is_file())

    def test_old_absolute_entries_follow_a_moved_data_folder_when_the_file_is_missing_there(self):
        old = self.tmp / "old-home" / "ImageLoom"
        (self.data / "models" / "m1" / "w.safetensors").write_bytes(b"x")
        self.write([entry("m1", {"checkpoint": str(old / "models" / "m1" / "w.safetensors")})])  # recorded before the folder moved
        reg = M.Registry(self.path, BUILTIN, self.data)
        self.assertEqual(reg.get("m1")["files"]["checkpoint"], str(self.data / "models" / "m1" / "w.safetensors"))
        self.assertEqual(self.on_disk()["models"][0]["files"]["checkpoint"], "models/m1/w.safetensors")

    def test_dotdot_in_a_stored_path_is_normalised(self):
        self.write([entry("m1", {"checkpoint": "models/../models/m1/w.safetensors"})])
        reg = M.Registry(self.path, BUILTIN, self.data)
        self.assertEqual(reg.get("m1")["files"]["checkpoint"], str(self.data / "models" / "m1" / "w.safetensors"))

    def test_damaged_registry_is_set_aside_and_the_app_starts_empty(self):
        self.path.write_text("{ this is not json", encoding="utf-8")
        reg = M.Registry(self.path, BUILTIN, self.data)
        self.assertEqual(reg.data["models"], [])
        self.assertEqual(reg.data["active"], "builtin")
        self.assertTrue(list(self.data.glob("models.json.bad-*")), "the broken file must be kept")
        reg.add(entry("m1", {}))  # and it works afterwards
        self.assertEqual(self.on_disk()["models"][0]["id"], "m1")

    def test_wrong_shapes_and_unusable_entries(self):
        for content in ("[]", "5", '"x"', "null"):
            with self.subTest(content):
                self.path.write_text(content, encoding="utf-8")
                self.assertEqual(M.Registry(self.path, BUILTIN, self.data).data["models"], [])
        good = entry("good", {})
        self.write([good, {"id": "x", "family": "no-such-family"}, "junk", {"family": "sdxl"}, entry("builtin", {}), dict(good)], active="ghost")
        reg = M.Registry(self.path, BUILTIN, self.data)
        self.assertEqual([m["id"] for m in reg.data["models"]], ["good"])  # unknown family, junk, the built-in's id and the duplicate are dropped
        self.assertEqual(reg.data["active"], "builtin")  # an active id that does not exist falls back
        self.assertTrue(list(self.data.glob("models.json.bad-*")))

    def test_concurrent_writers_never_corrupt_the_file(self):
        reg = M.Registry(self.path, BUILTIN, self.data)
        errors = []

        def work(n):
            try:
                for i in range(15):
                    reg.add(entry(f"m{n}-{i}", {"checkpoint": str(self.data / "models" / f"m{n}-{i}" / "w.safetensors")}))
                    reg.update(f"m{n}-{i}", status="ready")
                    json.loads(self.path.read_text(encoding="utf-8"))  # a reader must always see a complete document
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(n,)) for n in range(6)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])
        self.assertEqual(len(self.on_disk()["models"]), 90)
        self.assertEqual(list(self.data.glob(".*.tmp")), [], "no temp files left behind")

    def test_remove_resets_the_active_model(self):
        reg = M.Registry(self.path, BUILTIN, self.data)
        reg.add(entry("m1", {}))
        reg.set_active("m1")
        self.assertEqual(self.on_disk()["active"], "m1")
        reg.remove("m1")
        self.assertEqual(self.on_disk()["active"], "builtin")


class AtomicFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="atomic-")).resolve()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_atomic_write_replaces_whole_and_leaves_no_temp(self):
        p = self.tmp / "state.json"
        M.atomic_write(p, "one")
        M.atomic_write(p, b"two")
        self.assertEqual(p.read_text(), "two")
        self.assertEqual([f.name for f in self.tmp.iterdir()], ["state.json"])

    @unittest.skipIf(os.name == "nt", "POSIX permissions")
    def test_mode_applies_from_the_first_byte(self):
        p = self.tmp / "secret"
        p.write_text("old")
        p.chmod(0o644)
        M.atomic_write(p, "token", mode=0o600)
        self.assertEqual(oct(p.stat().st_mode & 0o777), "0o600")  # an existing loose file is tightened too

    def test_failed_write_keeps_the_old_file(self):
        p = self.tmp / "state.json"
        M.atomic_write(p, "good")
        with mock.patch("os.replace", side_effect=OSError("disk gone")), self.assertRaises(OSError):
            M.atomic_write(p, "new")
        self.assertEqual(p.read_text(), "good")
        self.assertEqual([f.name for f in self.tmp.iterdir()], ["state.json"])

    def test_read_json_quarantines_damage(self):
        p = self.tmp / "x.json"
        p.write_text("{oops", encoding="utf-8")
        self.assertEqual(M.read_json(p, {"a": 1}), {"a": 1})
        self.assertFalse(p.exists())
        self.assertEqual(len(list(self.tmp.glob("x.json.bad-*"))), 1)
        p.write_text("[1]", encoding="utf-8")
        self.assertEqual(M.read_json(p, {}), {})  # wrong type counts as damage
        self.assertEqual(M.read_json(self.tmp / "missing.json", {"d": 1}), {"d": 1})

    def test_read_json_survives_binary_garbage(self):
        p = self.tmp / "x.json"
        p.write_bytes(b"\xff\xfe\x00garbage\x80")
        self.assertEqual(M.read_json(p, {}), {})


class SdArgs(unittest.TestCase):
    def test_arguments_stay_separate_list_items(self):
        m = {"family": "qwen-image-2.1", "variant": "turbo", "files": {"diffusion": "/m/a b.gguf", "llm": "/m/l.gguf", "vae": "/m/v.safetensors", "llm_vision": "/m/mm.gguf"}}
        args = M.sd_args(m, "edit")
        self.assertIn("/m/a b.gguf", args)  # a path with a space is one argument, never split or shell-parsed
        self.assertEqual(args[args.index("--llm_vision") + 1], "/m/mm.gguf")
        self.assertNotIn("--llm_vision", M.sd_args(m, "generate"))
        self.assertEqual(args[args.index("--steps") + 1], "8")


if __name__ == "__main__":
    unittest.main()
