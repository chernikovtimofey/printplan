"""Plugin settings: LLM connection, server port, output folder, photo-to-3D service.

Inside STE Slicer the values live in the application preferences
(`ai_assistant/...`), so the API key is never stored in the sources; outside
the slicer (tests, dev server) an in-memory store is used. The key can also
come from the STE_LLM_API_KEY environment variable.
"""
import os
from collections import OrderedDict

from ..printplan.llm_client import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMClient

PREFIX = "ai_assistant/"
DEFAULT_PHOTO3D_URL = "http://127.0.0.1:17600"  # services/photo3d; reach a remote one through a forwarded port
DEFAULTS = OrderedDict([
    ("base_url", DEFAULT_BASE_URL),
    ("model", DEFAULT_MODEL),
    ("api_key", ""),
    ("port", 17530),
    ("output_dir", ""),
    ("photo3d_url", DEFAULT_PHOTO3D_URL),
])


class Settings(object):
    """In-memory settings store."""

    def __init__(self, values=None, use_env=True):
        self._values = dict(DEFAULTS)
        self._use_env = use_env
        if values:
            self.update(values)

    def get(self, key):
        value = self._read(key)
        return DEFAULTS.get(key) if value is None else value

    def set(self, key, value):
        if key not in DEFAULTS:
            raise KeyError(key)
        self._write(key, value)

    def update(self, values):
        for key, value in values.items():
            if key in DEFAULTS and value is not None:
                if key == "api_key" and value == "":
                    continue  # an empty field in the form means "keep the stored key"
                self.set(key, value)

    def _read(self, key):
        return self._values.get(key)

    def _write(self, key, value):
        self._values[key] = value

    # ------------------------------------------------------------------ derived values

    def api_key(self):
        env = os.environ.get("STE_LLM_API_KEY") if getattr(self, "_use_env", True) else None
        return self.get("api_key") or env or ""

    def port(self):
        try:
            return int(self.get("port"))
        except (TypeError, ValueError):
            return DEFAULTS["port"]

    def output_dir(self):
        folder = self.get("output_dir") or os.path.join(os.path.expanduser("~"), "STE-AI-Assistant")
        if not os.path.isdir(folder):
            os.makedirs(folder)
        return folder

    def photo3d_url(self):
        return (self.get("photo3d_url") or DEFAULT_PHOTO3D_URL).rstrip("/")

    def llm(self):
        """LLM client, or None when no API key is configured."""
        key = self.api_key()
        if not key:
            return None
        return LLMClient(base_url=self.get("base_url"), api_key=key, model=self.get("model"))

    def public(self):
        """Settings for the UI: the key itself is never sent back."""
        key = self.api_key()
        out = OrderedDict()
        out["base_url"] = self.get("base_url")
        out["model"] = self.get("model")
        out["has_api_key"] = bool(key)
        out["api_key_tail"] = key[-4:] if len(key) >= 8 else ""
        out["api_key_from_env"] = bool(not self.get("api_key") and key)
        out["port"] = self.port()
        out["output_dir"] = self.get("output_dir") or os.path.join(os.path.expanduser("~"), "STE-AI-Assistant")
        out["photo3d_url"] = self.photo3d_url()
        return out


class PreferencesSettings(Settings):
    """Settings stored in the slicer preferences (saved with the other preferences)."""

    def __init__(self, preferences, bridge=None, save=None):
        self._prefs = preferences
        self._bridge = bridge
        self._save = save
        for key, default in DEFAULTS.items():
            preferences.addPreference(PREFIX + key, default)

    def _read(self, key):
        return self._prefs.getValue(PREFIX + key)

    def _write(self, key, value):
        def write():
            self._prefs.setValue(PREFIX + key, value)
            if self._save is not None:
                self._save()
        if self._bridge is not None:
            self._bridge.call(write)
        else:
            write()
