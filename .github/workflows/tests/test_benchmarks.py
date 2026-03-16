# tests/test_benchmarks.py
import numpy as np
from scripts.utils.right_triangle_benchmarks import (
    RIGHT_TRIANGLE_ANGLES, 
    TETRAHEDRAL_ANGLES, 
    get_benchmark_label
)

def test_pythagorean_angles():
    """Verify Pythagorean triple angles are computed correctly."""
    # 3-4-5 triangle: arctan(3/4) ≈ 36.87°
    assert np.isclose(RIGHT_TRIANGLE_ANGLES['3-4-5'][0], 36.87, atol=0.01)
    assert np.isclose(RIGHT_TRIANGLE_ANGLES['3-4-5'][1], 53.13, atol=0.01)

def test_tetrahedral_angles():
    """Verify tetrahedral projection angles."""
    assert np.isclose(TETRAHEDRAL_ANGLES[0], 35.264, atol=0.01)  # arctan(1/√2)
    assert np.isclose(TETRAHEDRAL_ANGLES[1], 54.736, atol=0.01)  # complement

def test_benchmark_label_matching():
    """Test angle-to-label matching within tolerance."""
    label = get_benchmark_label(36.9, tolerance=0.1)
    assert label == 'Pythagorean_3-4-5'
    
    # Outside tolerance → None
    assert get_benchmark_label(40.0, tolerance=1.0) is None
