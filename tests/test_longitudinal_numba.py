import numpy as np, pytest
from snfs_traffic.core.longitudinal_kernels import advance_positions_kernel
from snfs_traffic.core.longitudinal_numba import NUMBA_AVAILABLE, advance_positions_numba

def test_longitudinal_numba_optional_import(): assert isinstance(NUMBA_AVAILABLE, bool)

@pytest.mark.skipif(not NUMBA_AVAILABLE, reason='numba unavailable')
def test_advance_positions_numba_equivalent():
 pos=np.array([0,5,9],dtype=np.int64); vel=np.array([1,2,0],dtype=np.int64); alive=np.array([True,True,False])
 assert np.array_equal(advance_positions_kernel(pos,vel,alive,road_length=10), advance_positions_numba(pos,vel,alive,road_length=10))
