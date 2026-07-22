import json
from types import SimpleNamespace


def read_json_conf(path):
    with open(path) as f:
        return json.load(f, object_hook=lambda d: SimpleNamespace(**d))


def read_split_months_conf(path):
    return read_json_conf(path)


def read_split_seasons_conf(path):
    return read_json_conf(path)


def read_compute_monthly_stats_conf(path):
    return read_json_conf(path)


def read_compute_season_field_stats_conf(path):
    return read_json_conf(path)


def read_create_structure_conf(path):
    return read_json_conf(path)


def read_gpu_operations_conf(path):
    return read_json_conf(path)


def read_multithread_operations_conf(path):
    return read_json_conf(path)


def read_mpi_operations_conf(path):
    return read_json_conf(path)
