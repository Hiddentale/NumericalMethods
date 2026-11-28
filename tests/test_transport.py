import numpy as np
import pytest
from num_pde_project import evaluate_empirical_convergence


@pytest.mark.parametrize("c, cfl", [(1.0, 0.8), (2.0, 0.5)])
def test_transport_second_order(c, cfl):
    u_exact = lambda x, t: np.sin(2 * np.pi * (x - c * t))
    _, _, orders = evaluate_empirical_convergence(
        u_0=lambda x: u_exact(x, 0),
        g0_u=lambda t: u_exact(0, t),
        cfl_number=cfl,
        u_exact=u_exact,
        final_time=1.0,
        c=c,
        initial_number_of_spatial_points=(20, 40, 80, 160),
    )
    assert orders[-1] > 1.9
