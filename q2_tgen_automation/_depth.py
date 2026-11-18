import biom
import numpy as np
from qiime2 import CategoricalMetadataColumn
from scipy.stats import hypergeom


STRATEGIES = (
    'weight-sample-and-features',
    'proportional-richness-retention',
    'group-retention',
    'guarantee-samples',
)


def select_sampling_depth(
    strategy: str,
    group: CategoricalMetadataColumn | None = None,
    sample_weight: float | None = None,
    feature_weight: float | None = None,
    min_sample_proportion: float | None = None,
) -> int:
    '''
    '''
    pass


def weight_sample_and_features(
    table: biom.Table,
    sampling_depth: int,
    sample_weight: float,
    feature_weight: float,
) -> float:
    '''
    Score a depth by weighted sample retention and feature retention.

    Samples with at least `sampling_depth` counts are retained. A feature is
    considered retained if it appears in any retained sample.

    Per sample feature retention is an expected value determined
    probabilistically:
        P(present) = 1 - ((N - c) choose d) / (N choose d),
        where N = total reads in a sample, c = feature count in the same
        sample, and d = sampling depth

    The probability that the feature is retained anywhere is then 1 minus the
    probability that it is absent in every retained sample.
    '''
    if not np.isclose(sample_weight + feature_weight, 1, rtol=0):
        raise ValueError('The sample and feature weights must sum to one.')

    totals = table.sum(axis='sample')
    retained = totals >= sampling_depth

    expected_features = 0.0
    observed_features = 0
    for feature_counts in table.iter_data(axis='observation', dense=True):
        if not feature_counts.any():
            continue

        observed_features += 1

        eligible = retained & (feature_counts > 0)
        if not eligible.any() or feature_weight == 0:
            continue

        # probability of sampling this feature at least once in each retained
        # sample
        presence = hypergeom.sf(
            0, totals[eligible], feature_counts[eligible], sampling_depth
        )

        # probability of sampling this feature in any retained sample
        expected_features += 1 - np.prod(1 - presence)

    if observed_features == 0:
        raise ValueError('The table must contain at least one positive count.')

    sample_proportion = retained.sum() / table.shape[1]
    feature_proportion = expected_features / observed_features

    return float(
        sample_weight * sample_proportion + feature_weight * feature_proportion
    )
