"""Input helpers: STL meshes and task texts (.txt / .md / .docx).

Everything returns plain numpy arrays / strings so the rest of the pipeline does
not care where the data came from (a file for the CLI, scene-node mesh data in
the plugin).
"""
import io
import os
import re
import struct
import zipfile

import numpy as np

_BINARY_RECORD = np.dtype([
    ("normal", "<f4", (3,)),
    ("vertices", "<f4", (3, 3)),
    ("attr", "<u2"),
])
_FLOAT_RE = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_VERTEX_RE = re.compile(
    r"vertex\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")")


def read_stl(path):
    """Read a binary or ASCII STL file into a (N, 3, 3) float64 array of triangles."""
    with open(path, "rb") as f:
        data = f.read()
    return parse_stl_bytes(data)


def parse_stl_bytes(data):
    if len(data) >= 84:
        count = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + 50 * count:
            records = np.frombuffer(data, dtype=_BINARY_RECORD, count=count, offset=84)
            return records["vertices"].astype(np.float64)
    head = data[:512].lstrip().lower()
    if head.startswith(b"solid"):
        text = data.decode("ascii", "replace")
        coords = np.array(_VERTEX_RE.findall(text), dtype=np.float64)
        if len(coords) % 3:
            raise ValueError("ASCII STL: vertex count is not a multiple of 3")
        return coords.reshape(-1, 3, 3)
    raise ValueError("Not a valid STL file (size does not match the binary layout and no 'solid' header)")


def write_stl(path, triangles, header=b"printplan"):
    """Write triangles as a binary STL (normals are computed from the vertex order)."""
    tris = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0] = 1.0
    records = np.zeros(len(tris), dtype=_BINARY_RECORD)
    records["normal"] = normals / lengths[:, None]
    records["vertices"] = tris
    with open(path, "wb") as f:
        f.write(header[:80].ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        f.write(records.tobytes())


def read_task(path_or_text):
    """Return the task text. Accepts a path to .txt/.md/.docx or the text itself."""
    if not isinstance(path_or_text, str) or "\n" in path_or_text or not os.path.isfile(path_or_text):
        return path_or_text
    ext = os.path.splitext(path_or_text)[1].lower()
    if ext == ".docx":
        return read_docx_text(path_or_text)
    with open(path_or_text, "rb") as f:
        raw = f.read()
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", "replace")


_PARA_RE = re.compile(r"<w:p[ >].*?</w:p>|<w:p/>", re.S)
_RUN_RE = re.compile(r"<w:t(?:\s[^>]*)?>([^<]*)</w:t>|<w:(tab|br)\b[^>]*/>")


def read_docx_text(path):
    """Extract paragraph text from a .docx without third-party libraries."""
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8")
    lines = []
    for para in _PARA_RE.findall(xml):
        parts = []
        for text, tag in _RUN_RE.findall(para):
            if tag == "tab":
                parts.append("\t")
            elif tag == "br":
                parts.append("\n")
            else:
                parts.append(text)
        line = _unescape_xml("".join(parts)).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _unescape_xml(text):
    return (text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
            .replace("&apos;", "'").replace("&amp;", "&"))


def read_json_resource(name):
    """Load a JSON file from the package data directory."""
    import json
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", name)
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)
