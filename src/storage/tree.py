"""Settings, results and plots as readable HDF5 groups (no class-by-class code).

The HDF5 hierarchy itself carries the structure, with the names of the dataclass fields and dict keys:

  dataclass   group, attribute _class = "analysis.filters.Settings"; one child per field
  dict        group, attribute _dict = 1; one child per key (attribute _order keeps the order)
  tuple/list  numbers or strings: one 1-D dataset (attribute _seq = "tuple" | "list");
              anything else: a group with attribute _seq and children "0", "1", ...
  ndarray     dataset named after the field (gzip above 1024 elements)
  scalar      attribute of the parent group, named after the field (bool, int, float, str)
  None        listed in the attribute _none of the parent
  anything else (threads, matrices, ...) is not stored

So `f["results/abel/a/abel/intensity"]` is an array and `f["settings/abel/a"].attrs["tikhonov"]` a number. Names
starting with "_" are structure. Classes are read back as the real objects, but only from the project's own packages (a file
must not be able to import arbitrary modules). Fields can be dropped when writing (SKIP): per-event arrays and
matrices that are rebuilt from the raw data anyway.

Files of format version 1 (one JSON string "_json" plus datasets a0, a1, ... per tree) are still read.
"""
import dataclasses
import importlib
import json

import numpy as np

ALLOWED_PREFIX = ("analysis.", "storage.", "gui.", "daq.", "session.", "paths.")   # the project's own packages (src/)

# (class name, field): not stored. Per-event arrays of the pipeline (rebuilt from the raw data) and the Abel matrices.
SKIP = frozenset({
    ("SideView", "x"), ("SideView", "y"), ("SideView", "weights"), ("SideView", "cycle"),
    ("AbelSide", "basis"), ("AbelSide", "binner"), ("AbelSide", "weights"), ("AbelSide", "cycle"),
    ("SelectionResult", "sum_a"), ("SelectionResult", "sum_b"), ("SelectionResult", "t_a"), ("SelectionResult", "t_b"),
    ("PosCorrState", "dx"), ("PosCorrState", "dy"),
})
COMPRESS_ABOVE = 1024
STR_ATTR_MAX = 4000                 # longer strings become datasets (attributes are limited to 64 kB)


def _string_dtype():
    import h5py
    return h5py.string_dtype()


def safe_name(key):
    """An HDF5 object name for a dict key ('/' separates paths, '_' starts structure)."""
    s = str(key).replace("/", "|")
    return ("k" + s) if s.startswith("_") or s in ("", ".") else s


# ====================================================================== writing
class _Writer:
    def __init__(self, skip):
        self.skip = skip

    def children(self, g, items):
        none, order = [], []
        for key, v in items:
            if self.put(g, key, v, none):
                order.append(key)
        if none:
            g.attrs["_none"] = json.dumps(none)
        g.attrs["_order"] = json.dumps(order)

    def put(self, g, key, v, none):
        """Store v as child `key` of group g. False if v is of a kind that is not stored."""
        if v is None:
            none.append(key)
        elif isinstance(v, (bool, np.bool_)):
            g.attrs[key] = bool(v)
        elif isinstance(v, (int, np.integer)):
            g.attrs[key] = int(v)
        elif isinstance(v, (float, np.floating)):
            g.attrs[key] = float(v)
        elif isinstance(v, str):
            if len(v) <= STR_ATTR_MAX:
                g.attrs[key] = v
            else:
                g.create_dataset(key, data=v, dtype=_string_dtype())
        elif isinstance(v, np.ndarray):
            self.array(g, key, v)
        elif isinstance(v, (tuple, list)):
            self.seq(g, key, v)
        elif isinstance(v, dict):
            sub = g.create_group(key)
            sub.attrs["_dict"] = 1
            keys = {}
            for k in v:
                name, n = safe_name(k), 1
                while name in keys:                               # two keys with the same safe name
                    n += 1
                    name = f"{safe_name(k)}~{n}"
                keys[name] = k
            if v and all(isinstance(k, int) and not isinstance(k, bool) for k in v):
                sub.attrs["_keytype"] = "int"
            self.children(sub, [(name, v[k]) for name, k in keys.items()])
        elif dataclasses.is_dataclass(v) and not isinstance(v, type):
            cls = type(v)
            sub = g.create_group(key)
            sub.attrs["_class"] = f"{cls.__module__}.{cls.__qualname__}"
            self.children(sub, [(f.name, getattr(v, f.name)) for f in dataclasses.fields(v)
                                if f.init and (cls.__name__, f.name) not in self.skip])
        else:
            return False
        return True

    def array(self, g, key, a):
        if a.dtype.kind in "biufc":
            if a.ndim > 0 and a.size > COMPRESS_ABOVE:
                g.create_dataset(key, data=a, compression="gzip", compression_opts=1, shuffle=True)
            else:
                g.create_dataset(key, data=a)
        else:                                                     # an array of strings / objects
            ds = g.create_dataset(key, data=[str(x) for x in a.ravel()], dtype=_string_dtype())
            ds.attrs["_array"] = 1

    def seq(self, g, key, v):
        kind = "tuple" if isinstance(v, tuple) else "list"
        items = list(v)
        isnum = all(isinstance(x, (bool, int, float, np.number, np.bool_)) for x in items)
        if isnum:                                                 # also the empty sequence
            if items and all(isinstance(x, (bool, np.bool_)) for x in items):
                a = np.array(items, bool)
            elif items and all(isinstance(x, (int, np.integer, bool, np.bool_)) for x in items):
                a = np.array(items, np.int64)
            else:
                a = np.array(items, np.float64)
            ds = g.create_dataset(key, data=a)
            ds.attrs["_seq"] = kind
        elif all(isinstance(x, str) for x in items):
            ds = g.create_dataset(key, data=items, dtype=_string_dtype())
            ds.attrs["_seq"] = kind
        else:
            sub = g.create_group(key)
            sub.attrs["_seq"] = kind
            sub.attrs["_len"] = len(items)
            self.children(sub, [(str(i), x) for i, x in enumerate(items)])


def write_tree(parent, name, obj, skip=SKIP):
    """Store obj as parent[name] (a group; a scalar or None becomes an attribute of parent)."""
    none = []
    _Writer(skip).put(parent, name, obj, none)
    if none:
        parent.attrs["_none"] = json.dumps(sorted(set(json.loads(parent.attrs.get("_none", "[]"))) | set(none)))


# ====================================================================== reading
def _native(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return x


def _is_string_ds(ds):
    return ds.dtype.kind in ("O", "S", "U")


class _Reader:
    def child(self, g, key):
        if key in g.attrs:
            return _native(g.attrs[key])
        if key in self.none(g):
            return None
        obj = g[key]
        return self.dataset(obj) if hasattr(obj, "dtype") else self.group(obj)

    @staticmethod
    def none(g):
        raw = g.attrs.get("_none") if hasattr(g.attrs, "get") else None
        return set(json.loads(_native(raw))) if raw is not None else set()

    def names(self, g):
        raw = g.attrs.get("_order")
        if raw is not None:
            return json.loads(_native(raw))
        found = [k for k in g.attrs if not k.startswith("_")] + list(g.keys()) + sorted(self.none(g))
        return list(dict.fromkeys(found))

    def dataset(self, ds):
        a = ds[()]
        seq = _native(ds.attrs["_seq"]) if "_seq" in ds.attrs else None
        if _is_string_ds(ds):
            if np.ndim(a) == 0:
                return _native(a)
            vals = [_native(x) for x in np.asarray(a, dtype=object).ravel()]
            if "_array" in ds.attrs:
                return np.array(vals)
            return tuple(vals) if seq == "tuple" else vals
        a = np.asarray(a)
        if seq is not None:
            lst = a.tolist()
            return tuple(lst) if seq == "tuple" else lst
        return a

    def group(self, g):
        cls = g.attrs.get("_class")
        if cls is not None:
            return self.dataclass(_native(cls), {k: self.child(g, k) for k in self.names(g)})
        seq = g.attrs.get("_seq")
        if seq is not None:
            items = [self.child(g, str(i)) for i in range(int(_native(g.attrs["_len"])))]
            return tuple(items) if _native(seq) == "tuple" else items
        conv = int if _native(g.attrs.get("_keytype", "")) == "int" else (lambda k: k)
        return {conv(k): self.child(g, k) for k in self.names(g)}

    @staticmethod
    def dataclass(path, values):
        if path.startswith("vmi."):                            # files written before src/vmi became src/
            path = path[4:]
        mod, _, name = path.rpartition(".")
        if not mod.startswith(ALLOWED_PREFIX):
            return values                                      # refuse to import anything else: keep the data
        try:
            cls = getattr(importlib.import_module(mod), name)
            known = {f.name: f for f in dataclasses.fields(cls) if f.init}
            kw = {k: v for k, v in values.items() if k in known}
            for k, f in known.items():                         # fields that were not stored (skipped / added later)
                if k not in kw and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                    kw[k] = None
            return cls(**kw)
        except Exception:                                      # class changed since: hand back the plain fields
            values["__class__"] = path
            return values


class _LegacyReader(_Reader):
    """Format version 1: a JSON string plus numbered array datasets in one group."""
    def __init__(self, group):
        self.g = group

    def dec(self, j):
        if isinstance(j, list):
            return [self.dec(x) for x in j]
        if not isinstance(j, dict):
            return j
        if "__array__" in j:
            return np.asarray(self.g[j["__array__"]][()])
        if "__list__" in j:
            return np.array(self.dec(j["__list__"]))
        if "__tuple__" in j:
            return tuple(self.dec(x) for x in j["__tuple__"])
        if "__dict__" in j:
            return {self.dec(k): self.dec(v) for k, v in j["__dict__"]}
        if "__skipped__" in j:
            return None
        if "__dc__" in j:
            return self.dataclass(j["__dc__"], {k: self.dec(v) for k, v in j["f"].items()})
        return {k: self.dec(v) for k, v in j.items()}


def read_tree(node):
    """The object stored by write_tree at this group (or dataset)."""
    if hasattr(node, "dtype"):
        return _Reader().dataset(node)
    if "_json" in node:
        raw = node["_json"][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return _LegacyReader(node).dec(json.loads(raw))
    return _Reader().group(node)
