from multiprocessing.pool import Pool

from typing import Tuple, Optional

from spdivik import distance as dst, kmeans as km


def assert_configured(config, name):
    assert name in config, 'Missing "' + name + '" field in config.'


def parse_distance(config) -> dst.ScipyDistance:
    assert_configured(config, 'distance')
    known_distances = {metric.value: metric for metric in dst.KnownMetric}
    assert config['distance'] in known_distances, \
        'Unknown distance {0}. Known: {1}'.format(
            config['distance'], list(known_distances.keys()))
    distance = dst.ScipyDistance(known_distances[config['distance']])
    return distance


def parse_initialization(config, distance: dst.DistanceMetric) \
        -> km.PercentileInitialization:
    assert_configured(config, 'initialization_percentile')
    initialization_percentile = float(config['initialization_percentile'])
    assert 0 <= initialization_percentile <= 100
    initialization = km.PercentileInitialization(
        distance=distance, percentile=initialization_percentile)
    return initialization


def parse_number_of_iterations(config) -> int:
    assert_configured(config, 'number_of_iterations')
    number_of_iterations = int(config['number_of_iterations'])
    assert 0 <= number_of_iterations
    return number_of_iterations


Min, Max = int, int


def parse_clustering_limits(config) -> Tuple[Min, Max]:
    assert_configured(config, 'maximal_number_of_clusters')
    maximal_number_of_clusters = int(config['maximal_number_of_clusters'])
    assert 1 < maximal_number_of_clusters, maximal_number_of_clusters
    minimal_number_of_clusters = int(config.get('minimal_number_of_clusters', 1))
    assert 0 < minimal_number_of_clusters < maximal_number_of_clusters, \
        (minimal_number_of_clusters, maximal_number_of_clusters)
    return minimal_number_of_clusters, maximal_number_of_clusters


def spawn_pool(config) -> Tuple[Pool, Optional[Pool]]:
    pool_max_tasks = int(config.get('pool_max_tasks', 4))
    assert 0 <= pool_max_tasks
    pool = Pool(maxtasksperchild=pool_max_tasks)
    use_pool_for_grouping = bool(config.get('use_pool_for_grouping', False))
    return pool, pool if use_pool_for_grouping else None


def parse_gap_trials(config) -> int:
    assert_configured(config, 'gap_trials')
    gap_trials = int(config['gap_trials'])
    assert 0 <= gap_trials
    return gap_trials
