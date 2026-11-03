"""Compact, frozen observation membership; project/partition IDs occur once.

Full partitions cost O(partitions). Irregular subsets use packed bits, never
JSON objects per cell. Iteration is provided for small legacy callers; scalable
consumers use partition masks, containment and set algebra directly.
"""
from __future__ import annotations

import base64
from bisect import bisect_right
from collections.abc import Sequence
from uuid import UUID

SCHEMA = "celltraj2.observation_scope.v1"


class ObservationScope(Sequence):
    def __init__(self, project_uuid, partitions=()):
        self.project_uuid = str(UUID(str(project_uuid)))
        records, seen = {}, set()
        for item in partitions:
            key = (str(UUID(str(item['dataset_uuid']))), str(UUID(str(item['roi_uuid']))), str(item['object_set']))
            if not key[2] or '/' in key[2] or '\\' in key[2] or key in seen:
                raise ValueError('Invalid or duplicate observation-scope partition')
            seen.add(key)
            size = int(item['size'])
            if size < 0 or size != item['size']:
                raise ValueError('Observation-scope size must be nonnegative')
            packed = item.get('bits')
            bits = None if packed is None else base64.b64decode(packed, validate=True)
            if bits is not None:
                if len(bits) != (size + 7) // 8 or (size % 8 and bits[-1] >> (size % 8)):
                    raise ValueError('Invalid observation-scope bitmap length/padding')
                count = sum(b.bit_count() for b in bits)
                if count == size:
                    bits = None
            else:
                count = size
            if count:
                records[key] = (size, bits, count)
        self._records = dict(sorted(records.items()))
        self.partitions = tuple(self._records)
        offsets = [0]
        for _, _, count in self._records.values():
            offsets.append(offsets[-1] + count)
        self._offsets = tuple(offsets)

    def __deepcopy__(self, memo):
        return self

    def __len__(self):
        return self._offsets[-1]

    def __contains__(self, key):
        if getattr(key, 'project_uuid', None) != self.project_uuid:
            return False
        record = self._records.get(key.partition)
        if record is None or not 1 <= key.observation_id <= record[0]:
            return False
        row = key.observation_id - 1
        return record[1] is None or bool(record[1][row // 8] & (1 << (row % 8)))

    def __iter__(self):
        from .interpretation import ProjectObservationKey
        for partition, (size, bits, _) in self._records.items():
            for start in range(1, size + 1, 65536):
                import numpy as np
                ids = np.arange(start, min(start + 65536, size + 1), dtype=np.int64)
                if bits is not None:
                    ids = ids[self.mask(partition, ids)]
                for identifier in ids:
                    yield ProjectObservationKey(self.project_uuid, *partition, int(identifier))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        part = bisect_right(self._offsets, index) - 1
        partition = self.partitions[part]
        offset = index - self._offsets[part]
        size, bits, _ = self._records[partition]
        if bits is None:
            identifier = offset + 1
        else:
            import numpy as np
            identifier = int(np.flatnonzero(np.unpackbits(np.frombuffer(bits, dtype=np.uint8), bitorder='little')[:size])[offset]) + 1
        from .interpretation import ProjectObservationKey
        return ProjectObservationKey(self.project_uuid, *partition, identifier)

    def mask(self, partition, observation_ids):
        import numpy as np
        ids = np.asarray(observation_ids, dtype=np.int64)
        record = self._records.get(tuple(partition))
        if record is None:
            return np.zeros(ids.shape, dtype=bool)
        size, bits, _ = record
        valid = (ids >= 1) & (ids <= size)
        if bits is not None and valid.any():
            rows = ids[valid] - 1
            valid[valid] = (np.frombuffer(bits, dtype=np.uint8)[rows // 8] & (1 << (rows % 8))) != 0
        return valid

    def restrict(self, partitions):
        selected = set(partitions)
        return ObservationScope(self.project_uuid, (r for r in self.to_dict()['partitions']
            if (r['dataset_uuid'], r['roi_uuid'], r['object_set']) in selected))

    def to_dict(self):
        return {'schema': SCHEMA, 'project_uuid': self.project_uuid, 'partitions': [
            dict(dataset_uuid=p[0], roi_uuid=p[1], object_set=p[2], size=size,
                 bits=None if bits is None else base64.b64encode(bits).decode('ascii'))
            for p, (size, bits, _) in self._records.items()]}

    @classmethod
    def from_dict(cls, value):
        if value.get('schema') != SCHEMA:
            raise ValueError('Unsupported observation-scope schema')
        return cls(value['project_uuid'], value['partitions'])

    @classmethod
    def from_masks(cls, project_uuid, masks):
        import numpy as np
        records = []
        for partition, mask in masks:
            mask = np.asarray(mask, dtype=bool)
            if mask.ndim != 1:
                raise ValueError('Observation mask must be one-dimensional')
            records.append(dict(dataset_uuid=partition[0], roi_uuid=partition[1], object_set=partition[2],
                size=len(mask), bits=None if mask.all() else base64.b64encode(np.packbits(mask, bitorder='little')).decode('ascii')))
        return cls(project_uuid, records)

    def _combine(self, other, operation):
        import numpy as np
        other = observation_scope(other, self.project_uuid)
        if other.project_uuid != self.project_uuid:
            raise ValueError('Observation scopes belong to different projects')
        records = []
        for partition in sorted(set(self.partitions) | set(other.partitions)):
            a, b = self._records.get(partition), other._records.get(partition)
            if a == b and a is not None:
                chosen = None if operation == 'difference' else a
            elif a is None:
                chosen = b if operation == 'union' else None
            elif b is None:
                chosen = None if operation == 'intersection' else a
            else:
                size = max(a[0], b[0])
                def packed(record):
                    n, bits, _ = record
                    out = np.zeros((size + 7) // 8, dtype=np.uint8)
                    if bits is None:
                        out[:n // 8] = 255
                        if n % 8: out[n // 8] = (1 << (n % 8)) - 1
                    else: out[:len(bits)] = np.frombuffer(bits, dtype=np.uint8)
                    return out
                left, right = packed(a), packed(b)
                values = left & right if operation == 'intersection' else left | right if operation == 'union' else left & ~right
                chosen = (size, values.tobytes(), None)
            if chosen is not None:
                size, bits, _ = chosen
                records.append(dict(dataset_uuid=partition[0], roi_uuid=partition[1], object_set=partition[2],
                    size=size, bits=None if bits is None else base64.b64encode(bits).decode('ascii')))
        return ObservationScope(self.project_uuid, records)

    def intersection(self, other): return self._combine(other, 'intersection')
    def union(self, other): return self._combine(other, 'union')
    def difference(self, other): return self._combine(other, 'difference')
    def issubset(self, other): return not self.difference(other)
    def __and__(self, other): return self.intersection(other)
    def __or__(self, other): return self.union(other)
    def __sub__(self, other): return self.difference(other)
    def __rsub__(self, other):
        return other.difference(self) if isinstance(other,ObservationScope) else {key for key in other if key not in self}
    def __eq__(self, other):
        if not isinstance(other, (ObservationScope, dict, Sequence, set, frozenset)):
            return NotImplemented
        if isinstance(other, ObservationScope) and self.project_uuid != other.project_uuid:
            return False
        return self.issubset(other) and observation_scope(other, self.project_uuid).issubset(self)


def observation_scope(keys, project_uuid=None):
    """Normalize legacy small key collections; compact scopes pass through."""
    if isinstance(keys, ObservationScope):
        if project_uuid is not None and keys.project_uuid != project_uuid:
            raise ValueError('Observation scope belongs to another project')
        return keys
    if isinstance(keys, dict):
        result = ObservationScope.from_dict(keys)
        return observation_scope(result, project_uuid)
    from .interpretation import ProjectObservationKey
    import numpy as np
    grouped = {}
    for key in keys:
        key = ProjectObservationKey.from_dict(key) if isinstance(key, dict) else key
        project_uuid = project_uuid or key.project_uuid
        if key.project_uuid != project_uuid:
            raise ValueError('Observation scope mixes projects')
        grouped.setdefault(key.partition, []).append(key.observation_id)
    masks = []
    for partition, ids in grouped.items():
        mask = np.zeros(max(ids), dtype=bool)
        mask[np.asarray(ids, dtype=np.int64) - 1] = True
        masks.append((partition, mask))
    if project_uuid is None:
        raise ValueError('Empty observation scope needs a project UUID')
    return ObservationScope.from_masks(project_uuid, masks)
