import json

import divik._inspect.color as color

DEFAULT_COLOR = 128


def as_rgb(selected_point, figure):
    try:
        cluster = int(json.loads(selected_point)['cluster'])
    except json.JSONDecodeError:
        return DEFAULT_COLOR, DEFAULT_COLOR, DEFAULT_COLOR
    except ValueError:
        return DEFAULT_COLOR, DEFAULT_COLOR, DEFAULT_COLOR
    colormap = color.make_colormap(figure['data'][0]['z'])
    selected_color = colormap[cluster][1]
    jsonified = selected_color.replace('rgb(', '[').replace(')', ']')
    return json.loads(jsonified)


def update_color_overrides(overrides, new_color, level, selected_point):
    if not overrides:
        overrides = '{}'

    try:
        cluster = str(int(json.loads(selected_point)['cluster']))
    except json.JSONDecodeError:
        return overrides
    except TypeError:
        return overrides

    overrides = json.loads(overrides)
    level = str(level)
    if level not in overrides:
        overrides[level] = {}
    overrides[level][cluster] = new_color
    return json.dumps(overrides)
