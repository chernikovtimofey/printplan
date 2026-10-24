"""Client of the photo-to-3D service (services/photo3d) and scaling of its meshes.

The slicer's Python cannot run an image-to-3D network, so the plugin sends the
photos to a separate service over HTTP:

    GET  /health    {"ready": bool, "models": {...}}
    POST /generate  {"images": [{"view": "front", "data": base64}], "seed": int}
                    -> {"stl": base64 binary STL, "triangles": n, "seconds": t, "model": ...}

The service returns the shape in STL axes (Z up) with the largest extent 1; the
real size is unknown from a photo, so `scale_to_size` applies the size the user
gave.
"""
import base64
import json
import socket
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from ..printplan.io_utils import parse_stl_bytes

VIEWS = ("front", "left", "back", "right")
MAX_PHOTOS = len(VIEWS)


class Photo3DError(Exception):
    pass


class Photo3DClient(object):
    def __init__(self, url, timeout=600.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        host = urllib.parse.urlparse(self.url).hostname or ""
        # A forwarded port on this machine must not go through a system proxy.
        handlers = [urllib.request.ProxyHandler({})] if host in ("127.0.0.1", "localhost", "::1") else []
        self._opener = urllib.request.build_opener(*handlers)

    def health(self):
        return self._request("GET", "/health", timeout=10.0)

    def generate(self, photos, views=None, seed=1234):
        """photos: list of image bytes (JPEG/PNG); views: front/left/back/right for each.
        Returns (triangles (N, 3, 3) with the largest extent ~1, service info)."""
        if not photos:
            raise Photo3DError("Нет фото для генерации модели")
        views = list(views or VIEWS[:len(photos)])
        images = [{"view": view, "data": base64.b64encode(data).decode("ascii")}
                  for view, data in zip(views, photos)]
        out = self._request("POST", "/generate", {"images": images, "seed": seed}, timeout=self.timeout)
        try:
            triangles = parse_stl_bytes(base64.b64decode(out["stl"]))
        except Exception as e:
            raise Photo3DError("Сервис вернул повреждённую модель: %s" % e)
        if len(triangles) == 0:
            raise Photo3DError("Сервис вернул пустую модель: деталь на фото не распознана")
        info = dict((k, v) for k, v in out.items() if k != "stl")
        return triangles, info

    def _request(self, method, path, body=None, timeout=None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.url + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with self._opener.open(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                message = json.loads(e.read().decode("utf-8")).get("error")
            except Exception:
                message = None
            raise Photo3DError("Сервис генерации моделей по фото: %s" % (message or "ошибка HTTP %d" % e.code))
        except (socket.timeout, TimeoutError):
            raise Photo3DError("Сервис генерации моделей по фото не ответил за %.0f с" % timeout)
        except (urllib.error.URLError, OSError) as e:
            raise Photo3DError("Сервис генерации моделей по фото недоступен по адресу %s (%s). Запустите "
                               "services/photo3d и, если он на другом компьютере, пробросьте его порт"
                               % (self.url, getattr(e, "reason", e)))


def scale_to_size(triangles, size_mm):
    """Scale so the largest extent equals size_mm, centre in XY and put the bottom at z = 0."""
    size_mm = float(size_mm)
    if not size_mm > 0:
        raise ValueError("Размер детали должен быть больше нуля")
    tris = np.asarray(triangles, dtype=np.float64)
    pts = tris.reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    extent = float((hi - lo).max())
    if extent <= 0:
        raise ValueError("Модель вырождена: нулевой размер")
    centre = np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2]])
    return (tris - centre) * (size_mm / extent)
