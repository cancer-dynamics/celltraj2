import unittest
from dataclasses import replace
from uuid import uuid4
from celltraj2.interpretation import ProjectObservationKey
from celltraj2.state_coordinates import PartitionDomain
from celltraj2.state_contracts import (KineticModelBinding, kinetic_cohort_digest,
    compact_kinetic_cohort_digest, KINETIC_MODEL_BINDING_V2_SCHEMA)


class PartitionContractsTests(unittest.TestCase):
    def test_partition_rows_are_ordered_local_uuid_identity(self):
        identifier, a, b = (str(uuid4()) for _ in range(3))
        domain = PartitionDomain.create(identifier, [a, b])
        self.assertEqual(PartitionDomain.from_dict(domain.to_dict()), domain)
        self.assertNotEqual(domain.domain_id, PartitionDomain.create(identifier, [b, a]).domain_id)
        self.assertNotEqual(domain.domain_id, PartitionDomain.create(str(uuid4()), [a, b]).domain_id)
        with self.assertRaisesRegex(ValueError, 'unique'):
            PartitionDomain.create(identifier, [a, a])
        with self.assertRaisesRegex(ValueError, 'row-order'):
            replace(domain, cluster_ids=(b, a))
        with self.assertRaises(ValueError):
            replace(domain, schema='unknown.v9')

    def test_old_kinetic_serialization_has_no_compact_fields(self):
        key = ProjectObservationKey(str(uuid4()), str(uuid4()), str(uuid4()), 'cells', 1)
        leaf = str(uuid4())
        time = {'unit': 's', 'source': 'test'}
        old = KineticModelBinding(str(uuid4()), str(uuid4()), leaf, (key,),
            kinetic_cohort_digest(leaf, (key,), timebase=time), time)
        self.assertNotIn('pair_set_id', old.to_dict())
        self.assertEqual(KineticModelBinding.from_dict(old.to_dict()), old)
        with self.assertRaisesRegex(ValueError, 'v1'):
            replace(old, pair_set_id=str(uuid4()))

    def test_compact_binding_rejects_wrong_versions_and_expanded_pair_ids(self):
        key = ProjectObservationKey(str(uuid4()), str(uuid4()), str(uuid4()), 'cells', 1)
        leaf, pair, app, state = (str(uuid4()) for _ in range(4))
        time = {'unit': 's', 'source': 'validated_pair_set', 'lag_s': 1.}
        binding = KineticModelBinding(str(uuid4()), str(uuid4()), leaf, (key,),
            compact_kinetic_cohort_digest(leaf, (key,), pair, app, time, (state,)), time,
            partition_id=str(uuid4()), operator_id=str(uuid4()), assessment_id=str(uuid4()),
            schema=KINETIC_MODEL_BINDING_V2_SCHEMA, pair_set_id=pair, assignment_id=app,
            pair_set_digest='a'*64, assignment_digest='b'*64, vocabulary=(state,))
        self.assertEqual(KineticModelBinding.from_dict(binding.to_dict()), binding)
        self.assertIn(pair, binding.artifact_ids)
        with self.assertRaisesRegex(ValueError, 'PairSets'):
            replace(binding, pair_ids=(str(uuid4()),))
        with self.assertRaisesRegex(ValueError, 'physical'):
            replace(binding, timebase={'unit': 'frame', 'source': 'unknown'})
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            replace(binding, schema='site.kinetic_model_binding.v99')


if __name__ == '__main__':
    unittest.main()
