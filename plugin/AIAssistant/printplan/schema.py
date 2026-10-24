"""Print-plan JSON schema and a minimal schema validator.

The same schema is the parameter schema of the `submit_print_plan` tool, so
the model is forced to produce it. The orientation is chosen by
`candidate_id` from the evidence table (the model never invents rotations),
and settings are a list of {key, value, reason} so every value carries its
reason.
"""
from collections import OrderedDict

VERDICTS = ["printable", "printable_with_tuning", "not_printable"]
SEVERITIES = ["high", "medium", "low"]
DEVIATION_KINDS = ["mode", "orientation", "material", "quality"]


def plan_schema(modes, quality_types, materials, orientation_ids):
    return OrderedDict([
        ("type", "object"),
        ("properties", OrderedDict([
            ("verdict", {"type": "string", "enum": VERDICTS}),
            ("mode", {"type": "string", "enum": list(modes)}),
            ("quality_type", {"type": "string", "enum": list(quality_types)}),
            ("material", {"type": "string", "enum": list(materials)}),
            ("orientation", OrderedDict([
                ("type", "object"),
                ("properties", OrderedDict([
                    ("candidate_id", {"type": "string", "enum": list(orientation_ids)}),
                    ("reason", {"type": "string"}),
                ])),
                ("required", ["candidate_id", "reason"]),
            ])),
            ("settings", OrderedDict([
                ("type", "array"),
                ("items", OrderedDict([
                    ("type", "object"),
                    ("properties", OrderedDict([
                        ("key", {"type": "string"}),
                        ("value", {"type": ["number", "boolean", "string"]}),
                        ("reason", {"type": "string"}),
                    ])),
                    ("required", ["key", "value", "reason"]),
                ])),
            ])),
            ("risks", OrderedDict([
                ("type", "array"),
                ("items", OrderedDict([
                    ("type", "object"),
                    ("properties", OrderedDict([
                        ("severity", {"type": "string", "enum": SEVERITIES}),
                        ("text", {"type": "string"}),
                        ("evidence", {"type": "string"}),
                    ])),
                    ("required", ["severity", "text"]),
                ])),
            ])),
            ("design_suggestions", {"type": "array", "items": {"type": "string"}}),
            ("deviations", OrderedDict([
                ("type", "array"),
                ("description", "Отказ от кандидата режима, лучшей ориентации, первого материала рейтинга "
                                "или базового профиля — с обоснованием по данным"),
                ("items", OrderedDict([
                    ("type", "object"),
                    ("properties", OrderedDict([
                        ("what", {"type": "string", "enum": DEVIATION_KINDS}),
                        ("reason", {"type": "string"}),
                    ])),
                    ("required", ["what", "reason"]),
                ])),
            ])),
            ("explanation", {"type": "string"}),
            ("questions_to_user", {"type": "array", "items": {"type": "string"}}),
        ])),
        ("required", ["verdict", "mode", "quality_type", "material", "orientation", "settings", "risks",
                      "design_suggestions", "explanation", "questions_to_user"]),
    ])


_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool, "null": type(None),
}


def _type_ok(value, name):
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, _TYPES[name])


def validate(instance, schema, path="$"):
    """Validate the subset of JSON Schema used here; returns a list of error strings."""
    errors = []
    expected = schema.get("type")
    if expected is not None:
        names = expected if isinstance(expected, list) else [expected]
        if not any(_type_ok(instance, n) for n in names):
            return ["%s: ожидался тип %s, получено %s" % (path, "/".join(names), type(instance).__name__)]
    if "enum" in schema and instance not in schema["enum"]:
        allowed = schema["enum"]
        shown = allowed if len(allowed) <= 12 else allowed[:12] + ["…"]
        errors.append("%s: значение %r не из списка %s" % (path, instance, shown))
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append("%s: %r меньше %r" % (path, instance, schema["minimum"]))
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append("%s: %r больше %r" % (path, instance, schema["maximum"]))
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append("%s: нет обязательного поля «%s»" % (path, key))
        for key, sub in schema.get("properties", {}).items():
            if key in instance:
                errors.extend(validate(instance[key], sub, "%s.%s" % (path, key)))
    if isinstance(instance, list) and "items" in schema:
        for i, item in enumerate(instance):
            errors.extend(validate(item, schema["items"], "%s[%d]" % (path, i)))
    return errors
