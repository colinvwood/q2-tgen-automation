from fractions import Fraction
from math import ceil

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


def proportional_richness_retention(
    table: biom.Table,
    sampling_depth: int,
) -> float:
    '''
    Score a depth by average proportional richness retention.

    Expected proportional richness is determined probabilistically per sample.
    For a sample with N reads, a feature with count c is present after
    rarefaction without replacement to depth d with probability:
        P(present) = 1 - ((N - c) choose d) / (N choose d)

    Summing all feature probabilities gives the whole samples expected
    richness.
    '''
    total_retention = 0.0
    nonempty_samples = 0
    for sample_counts in table.iter_data(axis='sample', dense=True):
        sample_counts = sample_counts[sample_counts > 0]
        observed_features = sample_counts.size
        if observed_features == 0:
            continue

        nonempty_samples += 1
        total_reads = sample_counts.sum()
        if total_reads < sampling_depth:
            continue

        presence = hypergeom.sf(
            0, total_reads, sample_counts, sampling_depth
        )
        expected_features = presence.sum()
        total_retention += expected_features / observed_features

    if nonempty_samples == 0:
        raise ValueError('The table must contain at least one positive count.')

    return float(total_retention / table.shape[1])


def guarantee_samples(
    table: biom.Table,
    min_sample_proportion: float,
) -> int:
    '''
    Return the highest sampling depth retaining the requested proportion of
    samples.
    '''
    totals = table.sum(axis='sample')
    if totals.size == 0:
        raise ValueError('The table must contain at least one sample.')

    # avoid rounding e.g. ceil(0.07 * 100) up to 8 instead of 7
    required_samples = ceil(Fraction(str(min_sample_proportion)) * totals.size)
    ranked_totals = np.sort(totals)[::-1]
    sampling_depth = int(ranked_totals[required_samples - 1])

    if sampling_depth < 1:
        raise ValueError(
            'No positive sampling depth can retain the requested proportion '
            'of samples because some samples are empty.'
        )

    return sampling_depth


def group_retention(
    table: biom.Table,
    group: CategoricalMetadataColumn,
    min_sample_proportion: float | None = None,
    min_sample_count: int | None = None,
) -> int:
    '''
    Return the highest sampling depth retaining at least the requested
    proportion or count of samples in every unqiue group of a categorical
    metadata column.

    Group sizes include empty samples in the table. Metadata is matched by
    sample ID; extra metadata rows are ignored. Every table sample must have
    a nonmissing group assignment. Missing assignments, a table with no
    samples, a group with too few samples, or a target requiring empty samples
    to survive raise ValueError.
    '''
    if (min_sample_proportion is None) == (min_sample_count is None):
        raise ValueError(
            'Specify one of `min_sample_proportion` and `min_sample_count`.'
        )

    totals = table.sum(axis='sample')
    if totals.size == 0:
        raise ValueError('The table must contain at least one sample.')

    assignments = group.to_series().reindex(table.ids(axis='sample'))
    missing_samples = assignments.index[assignments.isna()]
    if len(missing_samples) > 0:
        raise ValueError(
            'The following samples appear in the table but have no value in '
            f'the group column: {missing_samples}.'
        )

    group_indices = assignments.groupby(assignments, observed=True).indices
    group_depths = []
    for group_name, indices in group_indices.items():
        group_totals = totals[indices]
        if min_sample_count is not None:
            required_samples = min_sample_count
        else:
            required_samples = ceil(
                Fraction(str(min_sample_proportion)) * group_totals.size
            )

        if required_samples > group_totals.size:
            raise ValueError(
                f'Group {group_name} contains {group_totals.size} samples, '
                f'but at least {required_samples} are required.'
            )

        ranked_totals = np.sort(group_totals)[::-1]
        sampling_depth = int(ranked_totals[required_samples - 1])

        if sampling_depth < 1:
            raise ValueError(
                'No positive sampling depth can retain the requested '
                f'number of samples in group {group_name} because '
                'some samples are empty.'
            )
        group_depths.append(sampling_depth)

    return min(group_depths)
