from flask import Flask, send_file
from pyproj import Transformer
from io import BytesIO
from PIL import Image
import requests
import numpy as np
import cv2
from flask_caching import Cache
from slippy_tiles import tile_xy_to_north_west_latlon, latlon_to_tile_xy

cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})


app = Flask(__name__)
cache.init_app(app)

wgs84_to_crs3006 = Transformer.from_crs(
    "+proj=latlon",
    "+proj=utm +zone=33 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs",
)

wgs84_to_crs2056 = Transformer.from_crs(
    "+proj=latlon",
    "+proj=somerc +lat_0=46.9524055555556 +lon_0=7.43958333333333 +k_0=1 +x_0=2600000 +y_0=1200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs +type=crs"
)


def crs3006_to_tile_xy(x, y, zoom):
    scale = (2.0 ** (zoom)) / ((2**15) * 0.5) / 256
    xtile = int((x - 265000) * scale)
    ytile = int((7680000 - y) * scale) 
    return (xtile, ytile)


def crs2056_to_tile_xy(x, y, zoom):
    scale = 2 ** zoom / (2 ** 14) / 1000
    xtile = int((x - 248e4) * scale)
    ytile = int((1302e3 - y) * scale) 
    return (xtile, ytile)


def swedish_tile_xy_to_crs3006_north_west_xy(x_tile, y_tile, zoom):
    scale = (2.0 ** (zoom)) / ((2**15) * 0.5)/256
    x = x_tile / scale + 265000
    y = 7680000 - y_tile / scale 
    return (x, y)


def swiss_tile_xy_to_crs2056_north_west_xy(x_tile, y_tile, zoom):
    scale = 2 ** zoom / (2 ** 14) / 1000
    x = x_tile / scale + 248e4
    y = 1302e3 - y_tile / scale 
    return (x, y)


def latlon_to_crs3006_tile_coordinates(lat, lon, z):
    zoom = z - 2
    x, y = wgs84_to_crs3006.transform(lon, lat)
    tile_x, tile_y = crs3006_to_tile_xy(x, y, zoom)
    x_min, y_max = swedish_tile_xy_to_crs3006_north_west_xy(tile_x, tile_y, zoom)
    x_max, y_min = swedish_tile_xy_to_crs3006_north_west_xy(tile_x + 1, tile_y + 1, zoom)
    
    tile_height = y_max - y_min
    tile_width = x_max - x_min

    offset_x = (x - x_min) / tile_width * 256
    offset_y = (y_max - y) / tile_height * 256

    return offset_x, offset_y, tile_x, tile_y


def latlon_to_crs2056_tile_coordinates(lat, lon, z):
    zoom = z - 2
    x, y = wgs84_to_crs2056.transform(lon, lat)
    tile_x, tile_y = crs2056_to_tile_xy(x, y, zoom)
    x_min, y_max = swiss_tile_xy_to_crs2056_north_west_xy(tile_x, tile_y, zoom)
    x_max, y_min = swiss_tile_xy_to_crs2056_north_west_xy(tile_x + 1, tile_y + 1, zoom)
    
    tile_height = y_max - y_min
    tile_width = x_max - x_min

    offset_x = (x - x_min) / tile_width * 1000
    offset_y = (y_max - y) / tile_height * 1000

    return offset_x, offset_y, tile_x, tile_y


@cache.memoize(5*60)
def get_gokartor_tile(z, y, x):
    url = f'https://kartor.gokartor.se/Master/{z}/{y}/{x}.png'
    res = requests.get(url, stream=True)
    if res.status_code == 200:
        data = BytesIO(res.raw.read())
        return Image.open(data)
    return None


@cache.memoize(5*60)
def get_mapantch_tile(z, y, x):
    url = f'https://www.mapant.ch/wmts.php?layer=MapAnt%20Switzerland&style=default&tilematrixset=2056&Service=WMTS&Request=GetTile&Version=1.0.0&Format=image%2Fpng&TileMatrix={z}&TileCol={x}&TileRow={y}'
    #raise Exception(url)
    res = requests.get(url, stream=True)
    if res.status_code == 200:
        data = BytesIO(res.raw.read())
        return Image.open(data)
    return None


@app.route('/se/<int:z>/<int:x>/<int:y>.jpg')
@cache.memoize(7*24*3600)
def get_tile_se(z, x, y):
    north, west = tile_xy_to_north_west_latlon(x, y, z)
    south, east = tile_xy_to_north_west_latlon(x + 1, y + 1, z)
    zoom = z - 2
    nw_x, nw_y, nw_tile_x, nw_tile_y = latlon_to_crs3006_tile_coordinates(north, west, z)
    ne_x, ne_y, ne_tile_x, ne_tile_y = latlon_to_crs3006_tile_coordinates(north, east, z)
    se_x, se_y, se_tile_x, se_tile_y = latlon_to_crs3006_tile_coordinates(south, east, z)
    sw_x, sw_y, sw_tile_x, sw_tile_y = latlon_to_crs3006_tile_coordinates(south, west, z)
    
    tile_min_x = min(nw_tile_x, ne_tile_x, se_tile_x, sw_tile_x)
    tile_min_y = min(nw_tile_y, ne_tile_y, se_tile_y, sw_tile_y)
    tile_max_x = max(nw_tile_x, ne_tile_x, se_tile_x, sw_tile_x)
    tile_max_y = max(nw_tile_y, ne_tile_y, se_tile_y, sw_tile_y)

    src_tile_size = 256
    dst_tile_size = 256

    img_width = (tile_max_x - tile_min_x + 1) * src_tile_size
    img_height = (tile_max_y - tile_min_y + 1) * src_tile_size

    p1 = np.float32(
        [
            [0, 0],
            [dst_tile_size, 0],
            [dst_tile_size, dst_tile_size],
            [0, dst_tile_size],
        ]
    )

    p2 = np.float32(
        [
            [nw_x + (nw_tile_x - tile_min_x) * src_tile_size, nw_y + (nw_tile_y - tile_min_y) * src_tile_size],
            [ne_x + (ne_tile_x - tile_min_x) * src_tile_size, ne_y + (ne_tile_y - tile_min_y) * src_tile_size],
            [se_x + (se_tile_x - tile_min_x) * src_tile_size, se_y + (se_tile_y - tile_min_y) * src_tile_size],
            [sw_x + (sw_tile_x - tile_min_x) * src_tile_size, sw_y + (sw_tile_y - tile_min_y) * src_tile_size],
        ]
    )
    im = Image.new(mode="RGB", size=(img_width, img_height))
    for yy in range(tile_min_y, tile_max_y+1):
        for xx in range(tile_min_x, tile_max_x+1):
            tile = get_gokartor_tile(zoom, yy, xx)
            if tile:
                Image.Image.paste(im, tile, (int(src_tile_size * (xx - tile_min_x)), int(src_tile_size * (yy - tile_min_y))))
    coeffs, mask = cv2.findHomography( p2, p1,cv2.RANSAC, 5.0)
    img_alpha = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGRA)
    img = cv2.warpPerspective(
        img_alpha,
        coeffs,
        (dst_tile_size, dst_tile_size),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255, 0),
    )
    _, buffer = cv2.imencode(".jpeg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    data_out = BytesIO(buffer)
    return send_file(data_out, mimetype="image/jpeg")


@app.route('/ch/<int:z>/<int:x>/<int:y>.jpg')
@cache.memoize(7*24*3600)
def get_tile_ch(z, x, y):
    north, west = tile_xy_to_north_west_latlon(x, y, z)
    south, east = tile_xy_to_north_west_latlon(x + 1, y + 1, z)
    zoom = z - 7
    nw_x, nw_y, nw_tile_x, nw_tile_y = latlon_to_crs2056_tile_coordinates(north, west, z)
    ne_x, ne_y, ne_tile_x, ne_tile_y = latlon_to_crs2056_tile_coordinates(north, east, z)
    se_x, se_y, se_tile_x, se_tile_y = latlon_to_crs2056_tile_coordinates(south, east, z)
    sw_x, sw_y, sw_tile_x, sw_tile_y = latlon_to_crs2056_tile_coordinates(south, west, z)
    
    tile_min_x = min(nw_tile_x, ne_tile_x, se_tile_x, sw_tile_x)
    tile_min_y = min(nw_tile_y, ne_tile_y, se_tile_y, sw_tile_y)
    tile_max_x = max(nw_tile_x, ne_tile_x, se_tile_x, sw_tile_x)
    tile_max_y = max(nw_tile_y, ne_tile_y, se_tile_y, sw_tile_y)

    src_tile_size = 1000
    dst_tile_size = 256

    img_width = (tile_max_x - tile_min_x + 1) * src_tile_size
    img_height = (tile_max_y - tile_min_y + 1) * src_tile_size

    p1 = np.float32(
        [
            [0, 0],
            [dst_tile_size, 0],
            [dst_tile_size, dst_tile_size],
            [0, dst_tile_size],
        ]
    )

    p2 = np.float32(
        [
            [nw_x + (nw_tile_x - tile_min_x) * src_tile_size, nw_y + (nw_tile_y - tile_min_y) * src_tile_size],
            [ne_x + (ne_tile_x - tile_min_x) * src_tile_size, ne_y + (ne_tile_y - tile_min_y) * src_tile_size],
            [se_x + (se_tile_x - tile_min_x) * src_tile_size, se_y + (se_tile_y - tile_min_y) * src_tile_size],
            [sw_x + (sw_tile_x - tile_min_x) * src_tile_size, sw_y + (sw_tile_y - tile_min_y) * src_tile_size],
        ]
    )
    im = Image.new(mode="RGB", size=(img_width, img_height))
    for yy in range(tile_min_y, tile_max_y+1):
        for xx in range(tile_min_x, tile_max_x+1):
            tile = get_mapantch_tile(zoom, yy, xx)
            if tile:
                Image.Image.paste(im, tile, (int(src_tile_size * (xx - tile_min_x)), int(src_tile_size * (yy - tile_min_y))))
    coeffs, mask = cv2.findHomography( p2, p1,cv2.RANSAC, 5.0)
    img_alpha = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGRA)
    img = cv2.warpPerspective(
        img_alpha,
        coeffs,
        (dst_tile_size, dst_tile_size),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255, 0),
    )
    _, buffer = cv2.imencode(".jpeg", img_alpha, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    data_out = BytesIO(buffer)
    return send_file(data_out, mimetype="image/jpeg")
