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
