#!/usr/bin/env python3
"""
branching_angle_analysis.py
===========================
Test the Right-Triangle Code Hypothesis:
Do natural branching systems converge on right-triangle angular ratios?

Usage:
    python branching_angle_analysis.py --input data/branches.csv --output results/
    python branching_angle_analysis.py --swc examples/example_neuron.swc --scale neuronal
    python branching_angle_analysis.py --geojson examples/example_river.geojson --scale fluvial

Author: Umair Abbas Siddiquie
License: MIT (see ../LICENSE_CODE.md)
DOI: 10.5281/zenodo.19038531
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, permutation_test
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

# Local imports
from utils.data_loader import load_branching_data
from utils.angle_extractor import compute_bifurcation_angles
from utils.clustering import cluster_angles, evaluate_clusters
from utils.right_triangle_benchmarks import RIGHT_TRIANGLE_ANGLES, TETRAHEDRAL_ANGLES, GOLDEN_RATIO_ANGLES
from utils.stats import permutation_test_significance, generate_null_distribution

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Test right-triangle code hypothesis in branching systems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input branches.csv --output results/
  %(prog)s --swc neuron.swc --scale neuronal --tolerance 2.0
  %(prog)s --geojson river.geojson --scale fluvial --test-scale-invariance
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str, help='CSV file with branching data')
    input_group.add_argument('--swc', type=str, help='SWC neuronal reconstruction file')
    input_group.add_argument('--geojson', type=str, help='GeoJSON river/lightning network')
    
    # Analysis parameters
    parser.add_argument('--scale', type=str, default='unknown',
                       choices=['neuronal', 'vascular', 'fluvial', 'lightning', 'galactic', 'unknown'],
                       help='Physical scale of the dataset')
    parser.add_argument('--tolerance', type=float, default=2.0,
                       help='Angle matching tolerance in degrees (default: 2.0)')
    parser.add_argument('--min-cluster-size', type=int, default=10,
                       help='Minimum samples per DBSCAN cluster (default: 10)')
    parser.add_argument('--eps', type=float, default=3.0,
                       help='DBSCAN neighborhood radius in degrees (default: 3.0)')
    
    # Testing options
    parser.add_argument('--test-scale-invariance', action='store_true',
                       help='Test if angle ratios persist across scales')
    parser.add_argument('--permutation-tests', type=int, default=1000,
                       help='Number of permutations for significance testing (default: 1000)')
    
    # Output options
    parser.add_argument('--output', type=str, default='results/',
                       help='Output directory for results (default: results/)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()


def match_to_benchmarks(angles: np.ndarray, tolerance: float) -> pd.DataFrame:
    """
    Match observed angles to right-triangle benchmark angles.
    
    Parameters
    ----------    angles : np.ndarray
        Array of bifurcation angles in degrees
    tolerance : float
        Matching tolerance in degrees
    
    Returns
    -------
    pd.DataFrame
        DataFrame with match results and statistics
    """
    # Combine all benchmark angles with labels
    benchmarks = []
    for label, angle_list in [
        ('Pythagorean_3-4-5', RIGHT_TRIANGLE_ANGLES['3-4-5']),
        ('Pythagorean_5-12-13', RIGHT_TRIANGLE_ANGLES['5-12-13']),
        ('Pythagorean_7-24-25', RIGHT_TRIANGLE_ANGLES['7-24-25']),
        ('Tetrahedral', TETRAHEDRAL_ANGLES),
        ('Golden_Ratio', GOLDEN_RATIO_ANGLES),
        ('Special_30-45-60', [30.0, 45.0, 60.0])
    ]:
        for angle in angle_list:
            benchmarks.append({'label': label, 'angle': angle})
    
    benchmarks_df = pd.DataFrame(benchmarks)
    
    # Match each observed angle to nearest benchmark
    results = []
    for angle in angles:
        diffs = np.abs(benchmarks_df['angle'] - angle)
        min_idx = diffs.idxmin()
        if diffs[min_idx] <= tolerance:
            results.append({
                'observed_angle': angle,
                'matched_benchmark': benchmarks_df.loc[min_idx, 'label'],
                'benchmark_angle': benchmarks_df.loc[min_idx, 'angle'],
                'deviation': diffs[min_idx],
                'matched': True
            })
        else:
            results.append({
                'observed_angle': angle,
                'matched_benchmark': None,
                'benchmark_angle': None,
                'deviation': None,
                'matched': False
            })
    
    return pd.DataFrame(results)

def test_scale_invariance(angles_by_scale: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Test if angle distributions are consistent across physical scales.
    
    Parameters
    ----------
    angles_by_scale : dict
        Dictionary mapping scale names to angle arrays
    
    Returns
    -------
    dict
        Kolmogorov-Smirnov test results for pairwise scale comparisons
    """
    results = {}
    scales = list(angles_by_scale.keys())
    
    for i, scale1 in enumerate(scales):
        for scale2 in scales[i+1:]:
            stat, p_value = ks_2samp(
                angles_by_scale[scale1],
                angles_by_scale[scale2]
            )
            results[f'{scale1}_vs_{scale2}'] = {
                'ks_statistic': stat,
                'p_value': p_value,
                'same_distribution': p_value > 0.05
            }
    
    return results


def main():
    """Main execution pipeline."""
    args = parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load data
    logger.info(f"Loading data from {args.input or args.swc or args.geojson}")
    df = load_branching_data(
        input_file=args.input or args.swc or args.geojson,
        file_type='csv' if args.input else ('swc' if args.swc else 'geojson'),
        scale=args.scale    )
    logger.info(f"Loaded {len(df)} nodes")
    
    # Extract bifurcation angles
    logger.info("Extracting bifurcation angles...")
    angles = compute_bifurcation_angles(df, dimension=2 if args.scale != 'galactic' else 3)
    logger.info(f"Extracted {len(angles)} bifurcation angles")
    
    # Save raw angles
    pd.Series(angles, name='bifurcation_angle_deg').to_csv(
        output_dir / 'raw_angles.csv', index=False
    )
    
    # Cluster angles to find dominant modes
    logger.info("Clustering angles...")
    cluster_results = cluster_angles(
        angles, 
        eps=args.eps, 
        min_samples=args.min_cluster_size
    )
    
    # Evaluate cluster quality
    if len(set(cluster_results['labels'])) > 1:
        sil_score = silhouette_score(
            angles.reshape(-1, 1), 
            cluster_results['labels']
        )
        logger.info(f"Silhouette score: {sil_score:.3f}")
    
    # Match to right-triangle benchmarks
    logger.info(f"Matching angles to benchmarks (tolerance
