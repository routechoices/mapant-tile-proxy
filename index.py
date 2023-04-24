from flask import Flask, send_file
from pyproj import Transformer
from io import BytesIO
from PIL import Image
import requests
import numpy as np
import cv2
from flask_caching import Cache
from slippy_tiles import tile_xy_to_north_west_latlon, latlon_to_tile_xy

#cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})


app = Flask(__name__)
#cache.init_app(app)

wgs82_to_crs3006 = Transformer.from_crs(
    "+proj=latlon",
    "+proj=utm +zone=33 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs",
)

crs3006_to_wgs84 = Transformer.from_crs(
    "+proj=utm +zone=33 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs",
    "+proj=latlon",
)

def crs3006_to_tile_xy(x, y, zoom):
    scale = (2.0 ** (zoom)) / ((2**15) * 0.5) / 256
    xtile = int((x - 265000) * scale)
    ytile = int((7680000 - y) * scale) 
    return (xtile, ytile)


def swedish_tile_xy_to_north_west_latlon(x_tile, y_tile, zoom):
    scale = (2.0 ** (zoom)) / ((2**15) * 0.5)/256
    x = x_tile / scale + 265000
    y = 7680000 - y_tile / scale 
    lon, lat = crs3006_to_wgs84.transform(x, y)
    return (lat, lon)


def latlon_to_tile_coordinates(lat, lon, z):
    zoom = z - 2
    x, y = wgs82_to_crs3006.transform(lon, lat)
    tile_x, tile_y = crs3006_to_tile_xy(x, y, zoom)
    lat_max, lon_min = swedish_tile_xy_to_north_west_latlon(tile_x, tile_y, zoom)
    lat_min, lon_max = swedish_tile_xy_to_north_west_latlon(tile_x + 1, tile_y + 1, zoom)
    x_min, y_max = wgs82_to_crs3006.transform(lon_min, lat_max)
    x_max, y_min = wgs82_to_crs3006.transform(lon_max, lat_min)
    
    tile_height = y_max - y_min
    tile_width = x_max - x_min

    offset_x = (x - x_min) / tile_width * 256
    offset_y = (y_max - y) / tile_height * 256

    return offset_x, offset_y, tile_x, tile_y


def get_gokartor_tile(z, y, x):
    url = f'https://kartor.gokartor.se/Master/{z}/{y}/{x}.png'
    res = requests.get(url, stream=True)
    if res.status_code == 200:
        data = BytesIO(res.raw.read())
        return Image.open(data)
    return None


@app.route('/')
def home():
    return "hello"

@app.route('/<int:z>/<int:x>/<int:y>.jpg')
def get_tile(z, x, y):
    north, west = tile_xy_to_north_west_latlon(x, y, z)
    south, east = tile_xy_to_north_west_latlon(x + 1, y + 1, z)
    zoom = z - 2
    nw_x, nw_y, nw_tile_x, nw_tile_y = latlon_to_tile_coordinates(north, west, z)
    ne_x, ne_y, ne_tile_x, ne_tile_y = latlon_to_tile_coordinates(north, east, z)
    se_x, se_y, se_tile_x, se_tile_y = latlon_to_tile_coordinates(south, east, z)
    sw_x, sw_y, sw_tile_x, sw_tile_y = latlon_to_tile_coordinates(south, west, z)
    
    tile_min_x = min(nw_tile_x, ne_tile_x, se_tile_x, sw_tile_x)
    tile_min_y = min(nw_tile_y, ne_tile_y, se_tile_y, sw_tile_y)
    tile_max_x = max(nw_tile_x, ne_tile_x, se_tile_x, sw_tile_x)
    tile_max_y = max(nw_tile_y, ne_tile_y, se_tile_y, sw_tile_y)

    img_width = (tile_max_x - tile_min_x + 1) * 256
    img_height = (tile_max_y - tile_min_y + 1) * 256

    p1 = np.float32(
        [
            [0, 0],
            [256, 0],
            [256, 256],
            [0, 256],
        ]
    )

    p2 = np.float32(
        [
            [nw_x + (nw_tile_x - tile_min_x) * 256, nw_y + (nw_tile_y - tile_min_y) * 256],
            [ne_x + (ne_tile_x - tile_min_x) * 256, ne_y + (ne_tile_y - tile_min_y) * 256],
            [se_x + (se_tile_x - tile_min_x) * 256, se_y + (se_tile_y - tile_min_y) * 256],
            [sw_x + (sw_tile_x - tile_min_x) * 256, sw_y + (sw_tile_y - tile_min_y) * 256],
        ]
    )
    im = Image.new(mode="RGB", size=(img_width, img_height))
    for yy in range(tile_min_y, tile_max_y+1):
        for xx in range(tile_min_x, tile_max_x+1):
            tile = get_gokartor_tile(zoom, yy, xx)
            if tile:
                Image.Image.paste(im, tile, (int(256 * (xx - tile_min_x)), int(256 * (yy - tile_min_y))))
    coeffs, mask = cv2.findHomography( p2, p1,cv2.RANSAC, 5.0)
    img_alpha = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGRA)
    img = cv2.warpPerspective(
        img_alpha,
        coeffs,
        (256, 256),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255, 0),
    )
    _, buffer = cv2.imencode(".jpeg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    data_out = BytesIO(buffer)
    return send_file(data_out, mimetype="image/jpeg")
