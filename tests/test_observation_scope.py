import base64
import json
import unittest
from uuid import uuid4

import numpy as np
from celltraj2.interpretation import ProjectObservationKey
from celltraj2.observation_scope import ObservationScope, observation_scope


class ObservationScopeTests(unittest.TestCase):
    def setUp(self):
        self.project = str(uuid4())
        self.partition = (str(uuid4()),str(uuid4()),'cells')

    def record(self, size, **extra):
        return dict(zip(('dataset_uuid','roi_uuid','object_set'),self.partition),size=size,**extra)

    def test_masks_algebra_and_legacy_normalization(self):
        rng = np.random.default_rng(11)
        for n in (0,1,7,8,9,33,257):
            for m in (0,n,n+5):
                a = ObservationScope.from_masks(self.project,[(self.partition,rng.random(n)>.4)])
                b = ObservationScope.from_masks(self.project,[(self.partition,rng.random(m)>.4)])
                ka,kb = set(a),set(b)
                self.assertEqual(set(a & b),ka & kb)
                self.assertEqual(set(a | b),ka | kb)
                self.assertEqual(set(a - b),ka - kb)
                self.assertEqual(a.issubset(b),ka.issubset(kb))
                self.assertEqual(a,observation_scope(ka,self.project))
                self.assertEqual(a,ObservationScope.from_dict(json.loads(json.dumps(a.to_dict()))))
                ids = np.arange(-2,max(n,m)+3)
                np.testing.assert_array_equal(a.mask(self.partition,ids),
                    [int(i) in {k.observation_id for k in ka} for i in ids])
                self.assertEqual(tuple(a),tuple(a[i] for i in range(len(a))))
                if a: self.assertEqual(a[-1],tuple(a)[-1])

    def test_large_full_scope_and_project_boundaries(self):
        a = ObservationScope(self.project,[self.record(30_000_000)])
        self.assertLess(len(json.dumps(a.to_dict())),400)
        self.assertEqual(a[-1].observation_id,30_000_000)
        self.assertEqual(a & a,a)
        self.assertFalse(a-a)
        foreign = ObservationScope(str(uuid4()),[self.record(30_000_000)])
        self.assertNotEqual(a,foreign)
        self.assertNotEqual(a,None)
        with self.assertRaisesRegex(ValueError,'project'): a.union(foreign)
        self.assertNotIn(ProjectObservationKey(foreign.project_uuid,*self.partition,1),a)

    def test_invalid_serialization(self):
        with self.assertRaisesRegex(ValueError,'schema'):
            ObservationScope.from_dict({'schema':'future'})
        for records in ([self.record(-1)],[self.record(1.5)],
                        [self.record(0),self.record(0)],
                        [self.record(9,bits=base64.b64encode(b'\x01').decode())],
                        [self.record(1,bits=base64.b64encode(b'\x80').decode())]):
            with self.assertRaises(ValueError): ObservationScope(self.project,records)


if __name__ == '__main__': unittest.main()
