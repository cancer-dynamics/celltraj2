import unittest
from types import SimpleNamespace
import numpy as np
from celltraj2.tracking import SparseAdjacency, TrackGraph
from celltraj2.trajectory_index import GraphIndex, Runs


def graph(parents):
    n=len(parents)
    edges=sorted((p-1,c) for c,p in enumerate(parents) if p)
    counts=np.bincount([p for p,c in edges],minlength=n)
    indptr=np.r_[0,np.cumsum(counts)]
    assignments=np.zeros(n,dtype=[('observation_id','i8'),('parent_observation_id','i8')])
    assignments['observation_id']=np.arange(1,n+1)
    assignments['parent_observation_id']=parents
    return TrackGraph(adjacency=SparseAdjacency(indptr=indptr,indices=np.array([c for p,c in edges],dtype='i8'),data=np.ones(len(edges)),shape=(n,n)),links=np.array([]),assignments=assignments)


class GraphIndexTests(unittest.TestCase):
    def test_branch_partition_and_barriers(self):
        index=GraphIndex.from_graph(graph([0,1,2,2,3,4]))
        self.assertEqual(len(np.unique(index.order)),6)
        self.assertEqual(len(index.order),6)
        np.testing.assert_equal(index.trace(4),[0,1,2,4])
        np.testing.assert_equal(index.trace(0),[0,1])
        frames=np.array([1,2,3,3,4,4])
        runs=index.project(np.arange(6),frames,np.ones(6,dtype=bool),np.zeros(6))
        windows=runs.windows(2)
        self.assertEqual(len(windows.starts),3)
        self.assertTrue(all(len(set(windows.members(i)))==2 for i in range(3)))
        # Absent interior observations never become a skip edge.
        mapping=np.arange(6);mapping[2]=-1
        filtered=index.project(mapping,frames,np.ones(6,dtype=bool))
        self.assertNotIn((1,4),[tuple(filtered.windows(2).members(i)) for i in range(len(filtered.windows(2).starts))])

    def test_type_and_split_barriers_and_gap(self):
        index=GraphIndex.from_graph(graph([0,1,2,3,4]))
        rows=np.arange(5)
        runs=index.project(rows,np.array([1,2,3,5,6]),np.ones(5,dtype=bool),np.array(['a','a','b','b','b']))
        self.assertEqual([tuple(runs.windows(2).members(i)) for i in range(2)],[(0,1),(3,4)])

    def test_cycles_and_disagreement(self):
        with self.assertRaisesRegex(ValueError,'Cycle'):
            GraphIndex.from_graph(graph([2,1]))
        g=graph([0,1]);g.assignments['parent_observation_id'][1]=0
        with self.assertRaisesRegex(ValueError,'disagree'):
            GraphIndex.from_graph(g)

    def test_long_delay_linear_storage_and_bounded_batches(self):
        n=100000
        index=GraphIndex.from_graph(graph(np.r_[0,np.arange(1,n)]))
        runs=index.project(np.arange(n),np.arange(n),np.ones(n,dtype=bool))
        small,big=runs.windows(2),runs.windows(1000)
        self.assertLessEqual(big.starts.nbytes,small.starts.nbytes)
        self.assertIs(small.runs,big.runs)
        for ids,batch in big.batches([np.arange(n)],indices=np.arange(25),max_batch_bytes=32000):
            self.assertLessEqual(batch.nbytes,32000)
            np.testing.assert_equal(batch[:,0],ids)
            np.testing.assert_equal(batch[:,-1],ids+999)
        a,b,dt=big.pairs(np.arange(n),lag_frames=100)
        np.testing.assert_equal(b-a,100)
        np.testing.assert_equal(dt,100)
        self.assertEqual(len(a),n-999-100)

    def test_missing_mean_not_zero_filled(self):
        windows=Runs(np.arange(3),np.array([0,3])).windows(2)
        self.assertTrue(np.isnan(windows.summarize([1,np.nan,3],'mean')).all())
        np.testing.assert_equal(windows.summarize([1,2,3],'mean'),[1.5,2.5])

if __name__=='__main__':unittest.main()
