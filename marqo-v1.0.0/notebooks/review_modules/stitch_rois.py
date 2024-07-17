import os
import sys
import cv2
import yaml
import json
import csv
import large_image
import numpy as np
import pandas as pd
import seaborn as sns
from plotly.offline import iplot
from plotly import offline
import plotly.graph_objects as go
from ipycanvas import Canvas
from matplotlib import colors
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from matplotlib.patches import Rectangle
from matplotlib.backend_bases import MouseButton
from IPython.display import display, HTML, clear_output, Markdown
from PIL.Image import open as open_pil
from PIL.Image import fromarray
import tifffile as tiff

from ipywidgets import Button, HBox, VBox, interact, interactive, fixed, interact_manual, Image, Layout, interactive_output, Output, HTML, GridBox, Box
from .modulewidgets import *
from time import time
import traceback

offline.init_notebook_mode(connected=True)


# evaluate the need to have an ABC to inherit from
class RGBStitcher:
    def __init__(self):
        self.button_name = 'Stitch ROIs together'
        self.selected_rois = []
        self.selected_points = []

    def _set_parameters(self, sample_path, user_id):
        self.config_file = os.path.join(sample_path, user_id, 'config.yaml')

        with open(self.config_file, 'r') as stream:
            config_data = yaml.safe_load(stream)
       
        try:
            self.markers = config_data['sample']['markers']
            self.sample_name = config_data['sample']['name']
            self.rgb_dir = config_data['directories']['rgb']
            self.geojson_dir = config_data['directories']['geojson'].replace('geojsons_perROI', 'geoJSONS_perROI')
            self.sample_dir = config_data['directories']['marco_output']
            self.output_resolution = config_data['parameters']['image']['output_resolution']
            self.marker_index_maskforclipping = config_data['parameters']['tiling']['marker_index_maskforclipping']
            self.marker_tile = config_data['sample']['markers'][self.marker_index_maskforclipping]
            self.scale_factor = config_data['parameters']['image']['scale_factor']

        except:
            self.markers = config_data['sample']['markers']
            self.sample_name = config_data['sample']['sample_name']
            self.rgb_dir = config_data['directories']['product_directories']['rgb']
            self.sample_dir = config_data['directories']['sample_directory']
            self.geojson_dir = config_data['directories']['product_directories']['geojson'].replace('geojsons_perROI', 'geoJSONS_perROI')
            self.output_resolution = config_data['parameters']['output_resolution']
            self.marker_index_maskforclipping = config_data['tiling_parameters']['marker_index_maskforclipping']
            self.marker_tile = config_data['sample']['markers'][self.marker_index_maskforclipping]
            self.scale_factor = config_data['parameters']['SF']

        self.tilestats_file = os.path.join(sample_path, user_id, f'{self.sample_name}_tilestats.csv')
        self.tilemap_file = os.path.join(sample_path, f'{self.sample_name}_tilemap.ome.tif')
        self.registered_mask = self.get_registered_mask()
        self.tilestats_map = self.open_tilestats()

        self.stitch_dir = self._create_dir()


    def _create_dir(self):
        stitch_dir = os.path.join(self.sample_dir, 'stitched_files')

        if not os.path.exists(stitch_dir):
            os.mkdir(stitch_dir)

        return stitch_dir


    def get_registered_mask(self):
        mask_name = f'{self.sample_name}_{self.marker_tile}_thumbnail.png'
        mask_path = os.path.join(self.sample_dir, mask_name)

        image = open_pil(mask_path)
        metadata = image.text


        registered_mask = [float(value) if key != 'linear_shift' else json.loads(value) for key, value in metadata.items()]
        return registered_mask


    def open_tilestats(self):
        """
        Instead of using spamreader you can open the file using pandas.read_csv, but the overhead for creating a dataframe is bigger
        """
        self.tilestats_20x = {}

        stream = open(self.tilestats_file)
        csv_data = csv.reader(stream)
        next(csv_data)

        tiles_map = {}
        for line in csv_data:
            tile_index = line[0]
            top_left_coords = line[3]

            top_x = int(top_left_coords.split(',')[1].strip(']'))
            top_y = int(top_left_coords.split(',')[0].strip('['))

            self.tilestats_20x[tile_index] = [top_x, top_y]

            top_x = top_x * (1.25 / self.output_resolution) + (self.registered_mask[4][1] * (1.25 / self.output_resolution))
            top_y = top_y * (1.25 / self.output_resolution) + (self.registered_mask[4][0] * (1.25 / self.output_resolution))

            tiles_map[tile_index] = [top_x, top_y]

        return tiles_map


    def _get_initialize_widgets(self):
        widgs = {'output_name': output_name(self.sample_name),
                 'marker': marker_channel(self.markers)}

        return widgs


    def _open_json(self, json_file):
        with open(json_file, 'r') as stream:
            json_data = json.load(stream)

        return json_data

    
    def _create_figure(self, roi_img, x, y, scale_factor=1.0):
        # Constants
        img_width = roi_img.shape[1]
        img_height = roi_img.shape[0]

        fig = go.FigureWidget()

        fig.add_trace(
            go.Scatter(
                x=np.round((x * scale_factor), decimals=1),
                y=np.round((y * scale_factor), decimals=1),
                mode="markers",
                marker_opacity=0
            )
        )

        fig.data[0].marker.color = ['#36454f'] * len(x)
        fig.data[0].marker.opacity = [0] * len(x)


        # Configure axes
        fig.update_xaxes(
            visible=False,
            range=[0, img_width * scale_factor]
        )

        fig.update_yaxes(
            visible=False,
            range=[0, img_height * scale_factor],
            # the scaleanchor attribute ensures that the aspect ratio stays constant
            scaleanchor="x"
        )

        # Overlay ROI img onto scatter plot
        fig.add_layout_image(
            dict(
                x=0,
                sizex=img_width * scale_factor,
                y=img_height * scale_factor,
                sizey=img_height * scale_factor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=fromarray(roi_img))
        )

        # Configure other layout
        fig.update_layout(
            width=img_width * scale_factor,
            height=img_height * scale_factor,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            hovermode='closest',
        )

        return fig


    def _interact_figure(self, centroid, point_id, tilestats_map, tilemap):
        roi = tilestats_map[centroid]

        c = list(tilemap.data[0].marker.color)
        o = list(tilemap.data[0].marker.opacity)

        if roi in self.selected_rois:
            self.selected_rois.remove(roi)
            last_selection = point_id
            o[last_selection] = 0

        else:
            self.selected_rois.append(roi)
            last_selection = point_id
            c[last_selection] = '#1FFF5E'
            o[last_selection] = 1

        with tilemap.batch_update():
            tilemap.data[0].marker.color = c
            tilemap.data[0].marker.opacity = o



    def display_tilemap(self):
        def handle_click(trace, points, state):
            point_id = points.point_inds[-1]
            centroid = Point(points.xs[0], points.ys[0])
            self._interact_figure(centroid, point_id, tilestats_map, tilemap)

        def handle_selection(trace, points, selector):
            coords = [(x, y) for x, y in zip(points.xs, points.ys)]
            points_ids = [_id for _id in points.point_inds]
            for coord, point_id in zip(coords, points_ids):
                centroid = Point(coord[0], coord[1])
                self._interact_figure(centroid, point_id, tilestats_map, tilemap)

        roi_img = np.array(open_pil(self.tilemap_file))

        x = np.array([coord[0] for coord in self.tilestats_map.values()]) + (500 * 1.25 / self.output_resolution)
        y = np.abs(np.array([coord[1] for coord in self.tilestats_map.values()]) - roi_img.shape[0]) - (500 * 1.25 / self.output_resolution)

        scale_factor = 1.0
        tilemap = self._create_figure(roi_img, x, y, scale_factor=scale_factor)
        
        x = np.round((x * scale_factor), decimals=1)
        y = np.round((y * scale_factor), decimals=1)    
        #tilestats_map = {Point(coord[0], coord[1]): roi_index for roi_index, coord in enumerate(zip(x, y))}
        tilestats_map = {Point(coord[0], coord[1]): int(roi_index) for roi_index, coord in zip(self.tilestats_map.keys(), zip(x, y))}
        
        scatter = tilemap.data[0]
        scatter.on_click(handle_click)
        scatter.on_selection(handle_selection)

        return tilemap


    def setup(self, sample_path, user_id):
        self._set_parameters(sample_path, user_id)
        widgs = self._get_initialize_widgets()

        self.output_name_widg = widgs['output_name']
        self.marker_widg = widgs['marker']

        # tilemap to select ROIs
        tilemap = self.display_tilemap()
        box_layout = Layout(overflow_x='scroll',
                            flex_flow='column',
                            display='flex',
                            height=f'{tilemap.layout.height}px',
                            align_items='flex-start')

        box = Box(children=[tilemap], layout=box_layout)

        ui = VBox([self.output_name_widg, self.marker_widg])

        display(box)
        display(ui) 

        widgs_ = {'output_name': fixed(self.output_name_widg),
                  'marker': fixed(self.marker_widg)}

        return widgs_ 


    def normalize_coords(self):
        i = 0
        coords = []
        aux = self.selected_rois.copy()  # creating aux variable because the selected_rois can change it length during the iteration
        for roi in aux:
            if str(roi) not in self.tilestats_20x:
                self.selected_rois.remove(roi)
                print(f'WARNING: ROI #{roi} skipped. No tilestats found for it')
                continue

            top_left = self.tilestats_20x[str(roi)]
            if i == 0:
                min_x = top_left[0]
                min_y = top_left[1]
                i += 1

            else:
                current_x = top_left[0]
                current_y = top_left[1]

                if current_x < min_x:
                    min_x = current_x

                if current_y < min_y:
                    min_y = current_y

            coords.append(top_left)    
        
        norm_coords = {roi: [coord[0] - min_x, coord[1] - min_y] for roi, coord in zip(self.selected_rois, coords)}

        return norm_coords


    def _get_max(self, norm_coords):
        max_x = 0
        max_y = 0
        for roi, coords in norm_coords.items():
            current_x = coords[0]
            current_y = coords[1]

            if current_x > max_x:
                max_x = current_x
    
            if current_y > max_y:
                max_y = current_y
        
        return max_x, max_y

    def update_geojson(self, roi_geojson_path, x, y):
        cells = []
        geojson_data = self._open_json(roi_geojson_path)

        for cell in geojson_data:
            cyto_coords = np.array(cell['geometry']['coordinates'][0]) + np.array([x, y])
            nuc_coords = np.array(cell['nucleusGeometry']['coordinates'][0]) + np.array([x, y])

            cell['geometry']['coordinates'][0] = cyto_coords.tolist()
            cell['nucleusGeometry']['coordinates'][0] = nuc_coords.tolist()

            cells.append(cell)
        
        return cells


    def _open_roi(self, roi_img_path):
        return plt.imread(roi_img_path)


    def stitch(self, norm_coords, marker):

        max_x, max_y = self._get_max(norm_coords)
        canvas = np.zeros((max_y + 1000, max_x + 1000, 3))

        all_cells = []
        for roi in self.selected_rois:
            roi_img_path = os.path.join(self.rgb_dir, f'{self.sample_name}_ROI#{roi}_{marker}.ome.tif')
            roi_geojson_path = os.path.join(self.geojson_dir, f'{self.sample_name}_ROI#{roi}_{marker}.json')
            x, y = norm_coords[roi]

            cells = self.update_geojson(roi_geojson_path, x, y)

            roi_img = self._open_roi(roi_img_path)

            canvas[y: y + 1000, x: x + 1000, :] = roi_img
            all_cells.extend(cells)

        return canvas, all_cells


    def execute(self, **kwargs):
        print(f'Stitched image is being created for the tiles: {self.selected_rois}')

        output_name = kwargs['output_name'].value
        marker = kwargs['marker'].value

        norm_coords = self.normalize_coords()

        canvas, cells = self.stitch(norm_coords, marker)

        # save canvas as stitched rois
        save_path = os.path.join(self.stitch_dir, f'{output_name}.ome.tif')
        with tiff.TiffWriter(save_path) as tif:
            tif.write(canvas, resolution=(1e4 / self.scale_factor, 1e4 / self.scale_factor, 'CENTIMETER'), metadata={'PhysicalSizeX': self.scale_factor, 'PhysicalSizeY': self.scale_factor}, compression='zlib')

        # save cells as new geojson
        save_path = os.path.join(self.stitch_dir, f'{output_name}.json')
        with open(save_path, 'w+') as fout:
            json.dump(cells, fout)

        print(f'{output_name} has been successfully stitched.')



