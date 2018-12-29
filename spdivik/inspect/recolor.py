import json

import spdivik.inspect.color as color

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


def update_color_overrides(overrides, r, g, b, level, selected_point):
    if not overrides:
        overrides = '{}'

    try:
        cluster = int(json.loads(selected_point)['cluster'])
    except json.JSONDecodeError:
        return overrides
    except TypeError:
        return overrides

    overrides = json.loads(overrides)
    if level not in overrides:
        overrides[level] = {}
    overrides[level][cluster] = 'rgb({0},{1},{2})'.format(r, g, b)
    return json.dumps(overrides)
