import json
import os

from divik._inspect.app import divik_result_path


def store_path():
    return os.path.join(divik_result_path(), '.inspect')


def initialize():
    os.makedirs(store_path(), exist_ok=True)


def find_preserved():
    initialize()
    return [
        {
            'label': os.path.splitext(path)[0],
            'value': os.path.splitext(path)[0],
        }
        for path
        in os.listdir(store_path())
    ]


def load(name):
    path = os.path.join(store_path(), '{0}.json'.format(name))
    with open(path) as infile:
        return json.load(infile)


def preserve_by_name(name, disabled, color, level):
    path = os.path.join(store_path(), '{0}.json'.format(name))
    with open(path, 'w') as outfile:
        json.dump({
            'color_overrides': color,
            'disabled_clusters': disabled,
            'level': level
        }, outfile, sort_keys=True)


def restore_level(name):
    return load(name)['level']


def restore_disabled_clusters(name):
    if not name:
        return []
    disabled_clusters = load(name)['disabled_clusters']
    storage = json.loads(disabled_clusters)
    return storage['excluded']


def restore_color_overrides(name):
    return load(name)['color_overrides']
