import unittest
from uuid import uuid4

from celltraj2.interpretation import TypeNode, TypeTaxonomy, canonical_json_digest
from celltraj2.state_coordinates import SAMPLE_DOMAIN_SCHEMA
from celltraj2.state_pairs import PAIR_RECIPE_SCHEMA, validate_pair_recipe


class PairRecipeTests(unittest.TestCase):
    def setUp(self):
        self.parent, self.leaf = str(uuid4()), str(uuid4())
        self.taxonomy = TypeTaxonomy(str(uuid4()), 'Types', (
            TypeNode(self.parent, 'Parent', 'parent', '#336699'),
            TypeNode(self.leaf, 'Leaf', 'leaf', '#669933', parent_type_id=self.parent)))
        domain = {'schema': SAMPLE_DOMAIN_SCHEMA, 'domain_kind': 'delay', 'row_count': 3,
            'length': 2, 'snapshot_id': str(uuid4()), 'snapshot_digest': 'a'*64,
            'snapshot_row_order_digest': 'b'*64, 'dependency_digest': 'c'*64,
            'row_order_digest': 'd'*64}
        domain['domain_id'] = canonical_json_digest(domain)
        self.recipe = {'schema': PAIR_RECIPE_SCHEMA, 'row_domain': domain,
            'exact_leaf_type_id': self.leaf, 'taxonomy_id': self.taxonomy.taxonomy_id,
            'type_release_id': str(uuid4()), 'type_classification_id': str(uuid4()),
            'type_assignments_digest': 'e'*64, 'split_digest': None,
            'history_length': 2, 'lag_frames': 3, 'time_mode': 'physical',
            'interval_s': 600., 'tolerance_s': 1e-6, 'physical_time_claim': True,
            'anchor': 'final', 'support_policy': 'complete_combined_path'}

    def test_exact_leaf_and_explicit_time_contract(self):
        self.assertEqual(validate_pair_recipe(self.recipe,taxonomy=self.taxonomy),self.recipe)
        frame = {**self.recipe,'time_mode':'frame','interval_s':None,'physical_time_claim':False}
        self.assertEqual(validate_pair_recipe(frame,taxonomy=self.taxonomy),frame)
        for change in ({'exact_leaf_type_id':self.parent}, {'history_length':3}, {'lag_frames':True},
                       {'time_mode':'frame'}, {'interval_s':0}, {'tolerance_s':float('nan')},
                       {'type_assignments_digest':'missing'}, {'split_digest':'missing'},
                       {'anchor':'initial'}, {'support_policy':'endpoints'}, {'schema':'future'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_pair_recipe({**self.recipe,**change},taxonomy=self.taxonomy)


if __name__ == '__main__':
    unittest.main()
