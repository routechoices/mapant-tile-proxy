import math


def latlon_to_tile_xy(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def latlon_to_tile_xy_offset(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    xx = (lon_deg + 180.0) / 360.0 * n
    yy = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    xoff = int((xx % 1.0) * 256)
    yoff = int((yy % 1.0) * 256)
    return (xoff, yoff)


def tile_xy_to_north_west_latlon(xtile, ytile, zoom):
    n = 2.0**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)
