from typing import Literal

from pydantic.fields import FieldInfo


def get_dspy_field_type(field: FieldInfo) -> Literal["input", "output"]:
    """Get the DSPy field type ("input" or "output") stored on a pydantic field.

    Args:
        field: The pydantic `FieldInfo` to inspect. It must have been created via
            `dspy.InputField` or `dspy.OutputField`, which set the `__dspy_field_type`
            key in `json_schema_extra`.

    Returns:
        Either `"input"` or `"output"`, depending on how the field was declared.

    Raises:
        ValueError: If the field does not have a `__dspy_field_type` set.
    """
    field_type = field.json_schema_extra.get("__dspy_field_type")
    if field_type is None:
        raise ValueError(f"Field {field} does not have a __dspy_field_type")
    return field_type
