"""What the slicer offers, and how setting values are validated.

`SlicerContext` is the interface the pipeline depends on. `CatalogSlicerContext`
implements it from data/slicer_catalog.json (built from the STE Slicer
definitions by tools/build_slicer_catalog.py); it is used by the CLI and tests.
Inside STE Slicer the plugin can subclass it and override
`validate_settings` to write the values into a temporary container and read
`stack.getProperty(key, "validationState")`, exactly as MachineErrorChecker
does — limits there depend on other settings, which the static catalog only
evaluates at their default values.
"""
from collections import OrderedDict

from .io_utils import read_json_resource

USED_MODES = ("classic", "cylindrical", "cylindrical_full")
# Settings the plan must never change (machine description, not print settings).
FORBIDDEN_PREFIXES = ("machine_", "extruder_nr", "material_diameter", "material_guid")


class SlicerContext(object):
    def machine(self):
        raise NotImplementedError

    def printing_modes(self):
        raise NotImplementedError

    def quality_profiles(self):
        raise NotImplementedError

    def materials(self):
        raise NotImplementedError

    def setting(self, key):
        raise NotImplementedError

    def validate_settings(self, settings):
        """List of issues: dicts with key, level ('error' | 'warning'), text."""
        raise NotImplementedError

    # helpers shared by implementations
    def quality_types(self):
        return [q["quality_type"] for q in self.quality_profiles()]

    def material_ids(self):
        return [m["id"] for m in self.materials()]

    def quality_layer_height(self, quality_type):
        for q in self.quality_profiles():
            if q["quality_type"] == quality_type:
                return q.get("layer_height")
        return None


class CatalogSlicerContext(SlicerContext):
    def __init__(self, catalog=None):
        self.catalog = catalog or read_json_resource("slicer_catalog.json")

    def machine(self):
        return self.catalog["machine"]

    def printing_modes(self):
        return OrderedDict((k, v) for k, v in self.catalog["printing_modes"].items() if k in USED_MODES)

    def quality_profiles(self):
        return self.catalog["quality_profiles"]

    def materials(self):
        return self.catalog["materials"]

    def setting(self, key):
        return self.catalog["settings"].get(key)

    def validate_settings(self, settings):
        issues = []
        for key, value in settings.items():
            issues.extend(check_value(key, value, self.setting(key)))
        return issues


def _issue(key, level, text):
    return OrderedDict([("key", key), ("level", level), ("text", text)])


def check_value(key, value, spec):
    """Static check of one value against its catalog entry."""
    if spec is None:
        return [_issue(key, "error", "неизвестный ключ настройки «%s»" % key)]
    if key.startswith(FORBIDDEN_PREFIXES):
        return [_issue(key, "error", "«%s» — параметр машины, план не должен его менять" % key)]
    stype = spec.get("type")
    if stype == "bool":
        if not isinstance(value, bool):
            return [_issue(key, "error", "«%s» должно быть true/false, получено %r" % (key, value))]
        return []
    if stype == "enum":
        options = spec.get("options") or []
        if value not in options:
            return [_issue(key, "error", "«%s»: значение %r не из списка %s" % (key, value, options))]
        return []
    if stype in ("float", "int"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [_issue(key, "error", "«%s» должно быть числом, получено %r" % (key, value))]
        if stype == "int" and float(value) != int(value):
            return [_issue(key, "error", "«%s» должно быть целым, получено %r" % (key, value))]
        out = []
        lo, hi = spec.get("min"), spec.get("max")
        if lo is not None and value < lo:
            out.append(_issue(key, "error", "«%s» = %g меньше минимума %g" % (key, value, lo)))
        if hi is not None and value > hi:
            out.append(_issue(key, "error", "«%s» = %g больше максимума %g" % (key, value, hi)))
        if not out:
            wlo, whi = spec.get("min_warning"), spec.get("max_warning")
            if wlo is not None and value < wlo:
                out.append(_issue(key, "warning", "«%s» = %g ниже рекомендуемого %g" % (key, value, wlo)))
            if whi is not None and value > whi:
                out.append(_issue(key, "warning", "«%s» = %g выше рекомендуемого %g" % (key, value, whi)))
        return out
    if stype == "str":
        return [] if isinstance(value, str) else [_issue(key, "error", "«%s» должно быть строкой" % key)]
    return [_issue(key, "error", "«%s»: тип %s нельзя задавать из плана" % (key, stype))]


def coerce_value(value, spec):
    """Best-effort conversion of LLM output ('true', '0.2', 3.0) to the setting's type."""
    if spec is None:
        return value
    stype = spec.get("type")
    if stype == "bool" and isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    if stype in ("float", "int") and isinstance(value, str):
        try:
            value = float(value.replace(",", "."))
        except ValueError:
            return value
    if stype == "int" and isinstance(value, float) and value == int(value):
        return int(value)
    if stype == "float" and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value
