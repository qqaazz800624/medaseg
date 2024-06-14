from queue import Queue
from typing import Any, Hashable, Mapping, Sequence, TextIO, Union

import ruamel.yaml
from ruamel.yaml.scalarbool import ScalarBoolean
from ruamel.yaml.scalarint import ScalarInt
from ruamel.yaml.scalarfloat import (
    ScalarFloat,
    ExponentialFloat,
    ExponentialCapsFloat
)
from ruamel.yaml.scalarstring import ScalarString


"""
Try to get anchor value from a YAML node, if the `anchor` attribute does not
exists, this function will return None.

Args:
    node: A node from YAML dict.

Returns:
    The name of anchor or None.

"""
def get_anchor_value(node: Any) -> Union[str, None]:
    anchor = getattr(node, "anchor", None)
    if anchor is not None:
        return anchor.value
    return None

"""
This function creates raumel.yaml objects that preserves original attributes
with a given new value.

Args:
    orig_obj: original object for reference
    new_value: the new value for the output object

Returns:
    A new object with the same type as `orig_obj` and the value is `new_value`

"""
def create_yaml_obj(orig_obj, new_value):
    anchor = get_anchor_value(orig_obj)
    ObjType = type(orig_obj)
    if isinstance(orig_obj, ScalarBoolean):
        return ScalarBoolean(new_value, anchor=anchor)
    if isinstance(orig_obj, ScalarInt):
        return ObjType(
            new_value,
            width=orig_obj._width,
            underscore=orig_obj._underscore,
            anchor=anchor
        )
    if isinstance(orig_obj, ScalarFloat):
        if ObjType in [ExponentialFloat, ExponentialCapsFloat]:
            new_obj = ObjType(
                new_value,
                width=orig_obj._width,
                underscore=orig_obj._underscore
            )
            if anchor is not None:
                new_obj.yaml_set_anchor(anchor, always_dump=True)
            return new_obj
        return ScalarFloat(
            new_value,
            width=orig_obj._width,
            prec=orig_obj._prec,
            m_sign=orig_obj._m_sign,
            m_lead0=orig_obj._m_lead0,
            exp=orig_obj._exp,
            e_width=orig_obj._e_width,
            e_sign=orig_obj._e_sign,
            underscore=orig_obj._underscore,
            anchor=anchor
        )
    if isinstance(orig_obj, ScalarString):
        return ObjType(value=new_value, anchor=anchor)
    # For non-scalar type objects
    new_obj = ObjType(new_value)
    if anchor is not None:
        new_obj.yaml_set_anchor(anchor, always_dump=True)
    return new_obj

"""
An iterator interface for dict and list. If the input data is not a list or
dictionary, an empty generator will be returned.

Args:
    data: a list or a dictionary, or any other type of object

Returns:
    If input data is a list, an enumerated list generator will be returned,
    if input data is a dictionary, a generater that iterate thorough its keys
    and values will be returned.

"""
def iter_list_or_dict(data: Any):
    if isinstance(data, dict):
        yield from data.items()
    if isinstance(data, list):
        yield from enumerate(data)
    return

"""
Load and modify a YAML file according to given anchors, and write to output.

Args:
    in_file: a str that contains the path to input YAML file or a TextIO obj.
    anchors: a mapping that maps the anchor names to its new values.
    out_file: a str for output file name or a writeable TexTIO obj.

Returns:
    None
"""
def yaml_update_anchors(
    in_file: Union[str, TextIO],
    anchors: Mapping[Hashable, Any],
    out_file: Union[str, TextIO]
) -> None:
    # Try to load YAML data from input file or IO
    ryaml = ruamel.yaml.YAML(typ="rt")
    if isinstance(in_file, str):
        with open(in_file) as in_f:
            data = ryaml.load(in_f)
    elif isinstance(in_file, TextIO):
        data = ryaml.load(in_file)
    else:
        raise ValueError(f"Unsupported input type {type(in_file)}")

    updates = {}

    # Traverse the full YAML tree to construct updates
    nodes = Queue()
    nodes.put(data)
    while not nodes.empty():
        node = nodes.get()
        key = get_anchor_value(node)
        if key in anchors:
            updates[id(node)] = {
                "key": key,
                "old_value": node,
                "new_value": create_yaml_obj(node, anchors[key])
            }
        for _, v in iter_list_or_dict(node):
            nodes.put(v)

    # Traverse the YAML tree again to apply updates
    nodes.put(data)
    while not nodes.empty():
        node = nodes.get()
        for k, v in iter_list_or_dict(node):
            item_id = id(v)
            if item_id in updates:
                node[k] = updates[item_id]["new_value"]
            nodes.put(v)

    # Write updates YAML to output file or stream
    if isinstance(out_file, str):
        with open(out_file, "w") as out_f:
            ryaml.dump(data, out_f)
    elif isinstance(out_file, TextIO):
        ryaml.dump(data, out_file)
    else:
        raise ValueError(
            f"Expected output type PathLike or StringIO, get {type(out_file)}."
        )

