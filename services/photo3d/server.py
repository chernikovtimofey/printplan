"""Photo-to-3D service for the AI Assistant plugin (Python 3.10+, CUDA GPU).

Turns 1-4 photos of a part into a mesh with Hunyuan3D-2 (shape only, no
texture) and returns it as a binary STL in STL axes (Z up), centred in XY,
bottom at z = 0 and the largest extent 1: the real size is unknown from a
photo, the plugin scales the mesh to the size the user gives.

    GET  /health    {"ready": bool, "error": str|null, "models": {...}}
    POST /generate  {"images": [{"view": "front"|"left"|"back"|"right", "data": base64}], "seed": int}
                    -> {"stl": base64, "triangles": n, "seconds": t, "model": str}

One photo goes to tencent/Hunyuan3D-2, several to tencent/Hunyuan3D-2mv
(views front/left/back/right). Each model runs on its own GPU, one request at a
time. Binds to 127.0.0.1 only; reach it from another computer through a
forwarded port (VS Code: Ports -> Forward a Port 17600).

    python services/photo3d/server.py [--port 17600] [--quality]
    python services/photo3d/server.py --generate front.jpg [left.jpg ...] --out part.stl
"""
import argparse
import base64
import io
import json
import logging
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

LOG = logging.getLogger("photo3d")
VIEWS = ("front", "left", "back", "right")
MAX_BODY = 64 * 1024 * 1024
MAX_IMAGE_SIDE = 1024
MAX_FACES = 200000

# (repository, turbo subfolder + steps, full-quality subfolder + steps)
MODELS = {
    "single": ("tencent/Hunyuan3D-2", ("hunyuan3d-dit-v2-0-turbo", 5), ("hunyuan3d-dit-v2-0", 30)),
    "multi": ("tencent/Hunyuan3D-2mv", ("hunyuan3d-dit-v2-mv-turbo", 5), ("hunyuan3d-dit-v2-mv", 30)),
}


class Generator(object):
    """Loads the pipelines in the background and runs one request per GPU at a time."""

    def __init__(self, quality=False, devices=("cuda:0", "cuda:1"), octree_resolution=320):
        self.quality = quality
        self.devices = {"single": devices[0], "multi": devices[-1]}
        self.octree_resolution = octree_resolution
        self.pipelines = {}
        self.steps = {}
        self.names = {}
        self.locks = {"single": threading.Lock(), "multi": threading.Lock()}
        self.error = None
        self.ready = threading.Event()
        self._rembg = None

    def load(self):
        try:
            import torch  # noqa: F401  (fail early with a clear message)
            from hy3dgen.rembg import BackgroundRemover
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            self._rembg = BackgroundRemover()
            for kind, (repo, turbo, full) in MODELS.items():
                subfolder, steps = full if self.quality else turbo
                LOG.info("loading %s/%s on %s", repo, subfolder, self.devices[kind])
                pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    repo, subfolder=subfolder, use_safetensors=True, device=self.devices[kind])
                if not self.quality:
                    try:
                        pipe.enable_flashvdm(mc_algo="mc")
                    except Exception:  # the turbo DiT still works with the standard decoder
                        LOG.exception("FlashVDM decoder unavailable, using the standard one")
                self.pipelines[kind] = pipe
                self.steps[kind] = steps
                self.names[kind] = "%s/%s" % (repo, subfolder)
        except Exception as e:
            LOG.exception("model loading failed")
            self.error = "%s: %s" % (type(e).__name__, e)
        finally:
            self.ready.set()

    def health(self):
        return {"ready": self.ready.is_set() and self.error is None, "loading": not self.ready.is_set(),
                "error": self.error, "models": self.names, "devices": self.devices}

    def generate(self, images, seed=1234):
        """images: list of (view, PIL.Image). Returns (STL bytes, triangle count, model name)."""
        import torch
        from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover

        if not self.ready.is_set():
            raise RuntimeError("модель ещё загружается, повторите через минуту")
        if self.error:
            raise RuntimeError("модель не загрузилась: %s" % self.error)
        kind = "multi" if len(images) > 1 else "single"
        prepared = [(view, self._prepare(image)) for view, image in images]
        cond = dict(prepared) if kind == "multi" else prepared[0][1]
        with self.locks[kind]:
            mesh = self.pipelines[kind](
                image=cond, num_inference_steps=self.steps[kind], octree_resolution=self.octree_resolution,
                num_chunks=20000, generator=torch.manual_seed(int(seed)), output_type="trimesh")[0]
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=MAX_FACES)
        triangles = _to_stl_axes(np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces)])
        return _stl_bytes(triangles), len(triangles), self.names[kind]

    def _prepare(self, image):
        """RGBA with a transparent background, at most MAX_IMAGE_SIDE pixels on a side."""
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        if image.mode == "RGBA" and np.asarray(image)[..., 3].min() < 250:
            return image  # already cut out
        return self._rembg(image.convert("RGB"))


def _to_stl_axes(triangles):
    """Hunyuan3D meshes are Y-up (glTF): rotate to Z-up, centre in XY, bottom at z = 0, largest extent 1."""
    t = np.stack([triangles[..., 0], -triangles[..., 2], triangles[..., 1]], axis=-1)
    pts = t.reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    t = t - np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2]])
    return t / float((hi - lo).max())


def _stl_bytes(triangles):
    tris = np.asarray(triangles, dtype=np.float32)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    records = np.zeros(len(tris), dtype=[("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    records["normal"], records["vertices"] = normals, tris
    return b"photo3d".ljust(80, b" ") + struct.pack("<I", len(tris)) + records.tobytes()


def _decode_images(body):
    items = body.get("images") or []
    if not 1 <= len(items) <= len(VIEWS):
        raise ValueError("нужно от 1 до %d фото" % len(VIEWS))
    views = [item.get("view") or VIEWS[i] for i, item in enumerate(items)]
    if any(v not in VIEWS for v in views) or len(set(views)) != len(views):
        raise ValueError("ракурсы должны быть разными, из %s" % ", ".join(VIEWS))
    images = []
    for view, item in zip(views, items):
        try:
            image = Image.open(io.BytesIO(base64.b64decode(item["data"])))
            image.load()
        except Exception:
            raise ValueError("не удалось прочитать фото (%s): нужен JPEG или PNG" % view)
        images.append((view, image))
    return images


def make_handler(generator):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                return self._json(200, generator.health())
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/generate":
                return self._json(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._json(413, {"error": "фото слишком большие"})
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                images = _decode_images(body)
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            started = time.time()
            try:
                stl, count, name = generator.generate(images, seed=body.get("seed", 1234))
            except RuntimeError as e:
                return self._json(503, {"error": str(e)})
            except Exception as e:
                LOG.exception("generation failed")
                return self._json(500, {"error": "генерация не удалась: %s: %s" % (type(e).__name__, e)})
            seconds = round(time.time() - started, 1)
            LOG.info("generated %d triangles from %d photo(s) in %.1f s", count, len(images), seconds)
            return self._json(200, {"stl": base64.b64encode(stl).decode("ascii"), "triangles": count,
                                    "seconds": seconds, "model": name})

        def _json(self, code, data):
            raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, fmt, *args):
            LOG.info("%s %s", self.address_string(), fmt % args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17600)
    parser.add_argument("--quality", action="store_true", help="full models (30 steps) instead of turbo (5 steps)")
    parser.add_argument("--devices", default="cuda:0,cuda:1", help="GPU for one photo, GPU for several")
    parser.add_argument("--octree-resolution", type=int, default=320)
    parser.add_argument("--generate", nargs="+", metavar="PHOTO", help="generate once from these photos and exit")
    parser.add_argument("--out", default="photo3d.stl")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    generator = Generator(quality=args.quality, devices=tuple(args.devices.split(",")),
                          octree_resolution=args.octree_resolution)
    if args.generate:
        generator.load()
        images = [(view, Image.open(path)) for view, path in zip(VIEWS, args.generate)]
        started = time.time()
        stl, count, name = generator.generate(images)
        with open(args.out, "wb") as f:
            f.write(stl)
        print("%s: %d triangles in %.1f s with %s" % (args.out, count, time.time() - started, name))
        return
    threading.Thread(target=generator.load, name="photo3d-load", daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(generator))
    LOG.info("photo3d service on http://%s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
