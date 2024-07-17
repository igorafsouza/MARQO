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

from ipywidgets import Button, HBox, VBox, interact, interactive, fixed, interact_manual, Image, Layout, interactive_output, Output, HTML, GridBox, Box
from .modulewidgets import *
from time import time
import traceback

offline.init_notebook_mode(connected=True)


# evaluate the need to have an ABC to inherit from
class VisualizePopulations:
    def __init__(self):
        self.button_name = 'Save Changes'
        self.selected_cells = []
        self.points_ids = []
        self.selection = []
        self.roi_selected = 0
        self.roi_scale_factor = 0.7
        self.roi_output = Output()
        self.tilemap_output = Output(layout={'align-items': 'center'})
        self.metrics_output = Output()

    def _set_parameters(self, sample_path, user_id):
        self.config_file = os.path.join(sample_path, user_id, 'config.yaml')

        with open(self.config_file, 'r') as stream:
            config_data = yaml.safe_load(stream)
       
        try:
            self.markers = config_data['sample']['markers']
            self.sample_name = config_data['sample']['name']
            self.technology = config_data['sample']['technology']
            self.rgb_dir = config_data['directories']['rgb']
            self.geojson_dir = config_data['directories']['geojson'].replace('geojsons_perROI', 'geoJSONS_perROI')
            self.marker_index_maskforclipping = config_data['parameters']['tiling']['marker_index_maskforclipping']
            self.marker_tile = config_data['sample']['markers'][self.marker_index_maskforclipping]
            self.sample_dir = config_data['directories']['marco_output']
            self.output_resolution = config_data['parameters']['image']['output_resolution']

        except:
            self.markers = config_data['sample']['markers']
            self.sample_name = config_data['sample']['sample_name']
            self.technology = 'micsss'
            self.rgb_dir = config_data['directories']['product_directories']['rgb']
            self.sample_dir = config_data['directories']['sample_directory']
            self.geojson_dir = config_data['directories']['product_directories']['geojson'].replace('geojsons_perROI', 'geoJSONS_perROI')
            self.marker_index_maskforclipping = config_data['tiling_parameters']['marker_index_maskforclipping']
            self.marker_tile = config_data['sample']['markers'][self.marker_index_maskforclipping]
            self.output_resolution = config_data['parameters']['output_resolution']

        if self.technology == 'tma':
            self.cores = config_data['sample']['tma_cores']

        elif self.technology == 'if':
            channel = config_data['sample']['dna_marker_channel']
            self.dna_marker = self.markers[channel]
            del self.markers[channel]

        self.finalcellinfo_file = os.path.join(sample_path, user_id, f'{self.sample_name}_finalcellinfo.csv')
        self.tilestats_file = os.path.join(sample_path, user_id, f'{self.sample_name}_tilestats.csv')
        self.tilemap_file = os.path.join(sample_path, f'{self.sample_name}_tilemap.ome.tif')
        self.tilemap = os.path.join(sample_path, f'{self.sample_name}_tilemap.ome.tif')
        self.pathology_summary = os.path.join(sample_path, user_id, f'{self.sample_name}_pathologysummary.csv')
        self.registered_mask = self.get_registered_mask()
        self.tilestats_map = self.open_tilestats()

        self._read()

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
        stream = open(self.tilestats_file)
        csv_data = csv.reader(stream)
        next(csv_data)

        tiles_map = {}
        for line in csv_data:
            tile_index = line[0]
            top_left_coords = line[3]

            top_x = int(top_left_coords.split(',')[1].strip(']'))
            top_y = int(top_left_coords.split(',')[0].strip('['))

            top_x = (top_x * (1.25 / self.output_resolution)) + (self.registered_mask[4][1] * (1.25 / self.output_resolution))
            top_y = (top_y * (1.25 / self.output_resolution)) + (self.registered_mask[4][0] * (1.25 / self.output_resolution))

            tiles_map[tile_index] = [top_x, top_y]

        return tiles_map


    def _read(self):
        finalcellinfo_df = pd.read_csv(self.finalcellinfo_file, sep=',', header=0, index_col=['Cell index'])

        #if self.technology == 'tma':
        #    finalcellinfo_df = finalcellinfo_df.loc[finalcellinfo_df['TMA Core'] == self.core]

        # Load the ROI dataframes
        rois_data = {}
        tiles = finalcellinfo_df['Tile index'].unique().tolist()
        for roi in tiles:
            roi_cells = finalcellinfo_df.loc[finalcellinfo_df['Tile index'] == roi].reset_index()
            rois_data[roi]= roi_cells
            
        # Load global data
        global_markers_data = {}
        self.global_cells_indices = set(finalcellinfo_df.index)
        for marker in self.markers:
            global_markers_data[marker] = {'pos': None, 'neg': None}
            global_pos = finalcellinfo_df.loc[finalcellinfo_df[f'{marker}_expression'] == True]
            global_neg = finalcellinfo_df.loc[finalcellinfo_df[f'{marker}_expression'] == False]
    
            global_markers_data[marker]['pos'] = set(global_pos.index)
            global_markers_data[marker]['neg'] = set(global_neg.index)


        self.rois_data = rois_data
        self.global_markers_data = global_markers_data

        self.annotations_indices = {}
        if os.path.exists(self.pathology_summary):
            path_summary = pd.read_csv(self.pathology_summary, index_col=['tissue_type'])
            tissue_types = path_summary.index

            for tissue in tissue_types:
                if tissue == 'whole_tissue':
                    continue

                annotation_columns = finalcellinfo_df['Tissue annotation'].str.split(',', expand=True)
                all_cells_tissue = finalcellinfo_df.loc[annotation_columns.apply(lambda row: tissue in row.values, axis=1)]

                self.annotations_indices[tissue] = set(all_cells_tissue.index)

        self.finalcellinfo_df = finalcellinfo_df


    def _get_initialize_widgets(self):
        #rois = self.finalcellinfo_df['Tile index'].unique().tolist()
        #max_roi_index = max(rois)
        #min_roi_index = min(rois)

        widgs = {'pos_markers': pos_markers(self.markers),
                 'neg_markers': neg_markers(self.markers),
                 'background_channel': background_channel(self.markers),
                 'show_all_cells': show_all_cells(),
                 'display_morph': display_morph(),
                 'fill_cells': fill_cells()}

        if self.technology == 'tma':
            widgs.update({'core': select_core(self.cores)})

        return widgs


    def _open_json(self, json_file):
        with open(json_file, 'r') as stream:
            json_data = json.load(stream)

        return json_data

    
    def _create_figure(self, roi_img, x, y, scale_factor=1.0, obj='roi'):
        # Constants
        img_width = roi_img.shape[1]
        img_height = roi_img.shape[0]

        fig = go.FigureWidget()

        fig.add_trace(
            go.Scatter(
                x=np.round((x * scale_factor), decimals=2),
                y=np.round((y * scale_factor), decimals=2),
                mode="markers",
                marker_opacity=0
            )
        )

        if obj == 'tilemap':
            c = ['#36454f'] * len(x)
            c[self.roi_selected] = '#1FFF5E'
            fig.data[0].marker.color = c

            o = [0] * len(x)
            o[self.roi_selected] = 1
            fig.data[0].marker.opacity = o

        else:
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


    def get_cells(self, pos_markers, neg_markers, marker, roi, finalcellinfo_df):
        #all_cells = finalcellinfo_df.loc[finalcellinfo_df['Tile index'] == roi].reset_index()
        all_cells = finalcellinfo_df[roi]
        pos_indices = self._subset(pos_markers, all_cells, expression=True)
        neg_indices = self._subset(neg_markers, all_cells, expression=False)

        if pos_indices and neg_indices:
            final_indices = list(set(pos_indices) & set(neg_indices))

        else:
            final_indices = pos_indices if pos_indices else neg_indices

        
        pos_cells = all_cells.loc[all_cells.index.isin(final_indices), :]
        neg_cells = all_cells.loc[~all_cells.index.isin(final_indices), :]

        return pos_cells, neg_cells, all_cells        


    def _interact_visualize(self, change):
        if self.technology == 'tma':
            self.core = self.select_core_widg.value

        self.selected_cells = []

        display_morph = self.display_morph_widg.value
        marker = self.background_channel_widg.value
        fill_cells = self.fill_cells_widg.value
        pos_markers = self.pos_markers_widg.value
        neg_markers = self.neg_markers_widg.value
        roi = self.roi_selected
        centroid_size = 4

        self._open()
        roi_img = self.roi_img

        geojson_data = self.geojson_data
        if geojson_data == []:
            print('No cells founds in the geojson file. The cells potentially were removed by denoising step or the tissue suffered too much damage in the remaining slides.')

        has_annotation = True
        if not os.path.exists(self.pathology_summary):
            has_annotation = False

        # grabing positive and negative cells
        #self.finalcellinfo_df = pd.read_csv(self.finalcellinfo_file, sep=',', header=0, index_col=['Cell index'])
        pos_cells, neg_cells, all_cells = self.get_cells(pos_markers, neg_markers, marker, roi, self.rois_data)

        x = np.array(all_cells['Relative x_coord'].round(2))
        y = np.abs(np.array(all_cells['Relative y_coord'].round(2)) - 1000)

        if display_morph == 'None':
            fig = self._create_figure(np.array(roi_img), x, y, scale_factor=self.roi_scale_factor)

        elif display_morph == 'Centroid':

            roi_img = self._plot_centroids(pos_cells, roi_img, geojson_data, centroid_size=centroid_size, color=[200, 0, 0])
            if self.show_all_cells_widg.value:
                roi_img = self._plot_centroids(neg_cells, roi_img, geojson_data, centroid_size=centroid_size, color=[0, 0, 200])

            fig = self._create_figure(roi_img, x, y, scale_factor=self.roi_scale_factor)

        else:
            morph_key = {'Nucleus': 'nucleusGeometry', 'Cytoplasm': 'geometry'}[display_morph]

            roi_img = self._plot_boundaries(pos_cells, geojson_data, morph_key, fill_cells, roi_img, color=[200, 0, 0])
            if self.show_all_cells_widg.value:
                roi_img = self._plot_boundaries(neg_cells, geojson_data, morph_key, fill_cells, roi_img, color=[0, 0, 200])
                
            fig = self._create_figure(roi_img, x, y, scale_factor=self.roi_scale_factor)

        #TODO: Table with metrics relative to the current ROI

        self.show_metrics(all_cells, pos_cells, neg_cells, marker, pos_markers, neg_markers, roi, has_annotation)

        self.fig = fig

        def handle_click(trace, points, state):
            centroid = (points.xs[0], points.ys[0])
            c = list(self.fig.data[0].marker.color)
            o = list(self.fig.data[0].marker.opacity)
            
            if points.point_inds not in self.points_ids: 
                self.points_ids.append(points.point_inds)
                self.selected_cells.append(centroid)
                for i in points.point_inds:
                    c[i] = '#ffff00'
                    o[i] = 1
                    with self.fig.batch_update():
                        self.fig.data[0].marker.color = c
                        self.fig.data[0].marker.opacity = o
            
            else:
                self.points_ids.remove(points.point_inds)
                self.selected_cells.remove(centroid)
                for i in points.point_inds:
                    c[i] = '#36454f'
                    o[i] = 0
                    with self.fig.batch_update():
                        self.fig.data[0].marker.color = c
                        self.fig.data[0].marker.opacity = o

        def selection_fn(trace,points,selector):
            self.selection.append(points.point_inds)

        scatter = self.fig.data[0]
        scatter.on_click(handle_click)
        scatter.on_selection(selection_fn)
        
    def _subset(self, markers, all_cells, expression=True):
        if not markers:
            all_indices = []
            return all_indices

        all_indices = all_cells.index
        for marker in markers:
            cells = all_cells.loc[all_cells[f'{marker}_expression'] == expression]
            indices = cells.index

            all_indices = list(set(all_indices) & set(indices))
            
        return all_indices


    def show_metrics(self, all_cells, pos_cells, neg_cells, marker, pos_markers, neg_markers, roi, has_annotation):
        with self.metrics_output:
            clear_output(wait=True)

            # Grab all pos/neg cells for the choosen marker(s) across the whole tissue
            #pos_indices_global = set()
            #neg_indices_global = set()

            pos_indices_global_list = []
            for marker in pos_markers:
                pos_indices = self.global_markers_data[marker]['pos']
                #pos_indices_global = pos_indices_global | pos_indices
                pos_indices_global_list.append(pos_indices)
            
            if pos_indices_global_list:
                pos_indices_global = pos_indices_global_list[0]

                for s in pos_indices_global_list:
                    pos_indices_global = pos_indices_global.intersection(s)

            else:
                pos_indices_global = set()


            neg_indices_global_list = []
            for marker in neg_markers:
                neg_indices = self.global_markers_data[marker]['neg']
                #neg_indices_global = neg_indices_global | neg_indices
                neg_indices_global_list.append(neg_indices)

            if neg_indices_global_list:
                neg_indices_global = neg_indices_global_list[0]

                for s in neg_indices_global_list:
                    neg_indices_global = neg_indices_global.intersection(s)

            else:
                neg_indices_global = set()

            if pos_indices_global and neg_indices_global:
                final_indices_global = pos_indices_global & neg_indices_global

            else:
                final_indices_global = pos_indices_global if pos_indices_global else neg_indices_global


            if self.technology == 'tma':
                core_indices = set(self.finalcellinfo_df.loc[self.finalcellinfo_df['TMA Core'] == self.core].index)
                final_indices_global = final_indices_global & core_indices


            pos_global = self.global_cells_indices & final_indices_global
            neg_global = self.global_cells_indices - final_indices_global 

            # Grab area of the current roi
            tilestats = pd.read_csv(self.tilestats_file, index_col=['Tile index'])
            roi_area_um = tilestats.loc[roi, 'Tissue area (microns2)'] 
            roi_area_mm = roi_area_um * 10**-6  # converting square um to square mm

            whole_tissue_area_um = np.sum(tilestats.loc[:, 'Tissue area (microns2)'])
            whole_tissue_area_mm = whole_tissue_area_um * 10**-6  # converting square um to square mm

            # print statements TODO: nice table
            pos_cells_global = len(pos_global)
            total_cells_global = len(self.global_cells_indices) if not self.technology == 'tma' else len(core_indices)
            pos_cells_roi = len(pos_cells)
            total_cells_roi = len(all_cells)

            perc_global = (pos_cells_global / total_cells_global) * 100
            perc_roi = (pos_cells_roi / total_cells_roi) * 100
            density_roi = pos_cells_roi / roi_area_mm
            density_global = pos_cells_global / whole_tissue_area_mm 

            html = ''

            html += f'<p> Total cells of interest in the current ROI: {pos_cells_roi} </p>'
            html += f'<p> Total segmented cells in the current ROI: {total_cells_roi} </p>'
            html += f'<p> {format(perc_roi, ".2f")}% of all segmented cells ({format(density_roi, ".2f")} cells/mm^2) are deemed positive in this region of interest. </p>'
            html += f'<p> {format(perc_global, ".2f")}% of all segmented cells ({format(density_global, ".2f")} cells/mm^2) are deemed positive in the whole tissue. </p>'

            # Check if there are tissue annotations and use them also as regions of interest
            if has_annotation:
                path_summary = pd.read_csv(self.pathology_summary, index_col=['tissue_type'])
                tissue_types = path_summary.index

                for tissue in tissue_types:
                    if tissue == 'whole_tissue':
                        continue

                    tissue_indices = self.annotations_indices[tissue]
                    pos_tissue = tissue_indices & final_indices_global
                    neg_tissue = tissue_indices - final_indices_global
                    
                    tissue_area_um = path_summary.loc[tissue, 'micron_area']
                    tissue_area_mm = tissue_area_um * 10**-6  # converting square um to square mm

                    perc = (len(pos_tissue) / len(tissue_indices)) * 100
                    density = len(pos_tissue) / tissue_area_mm

                    html += f'<p> {format(perc, ".2f")}% of all segmented cells ({density} cells/mm^2) are deemed positive in "{tissue}" annotated tissue for the whole slide. </p>'


            display(HTML(html))
            

    def _plot_centroids(self, cells, roi_img, geojson_data, centroid_size=1, color=[0, 0, 0]):
        roi_open_cv = np.array(roi_img)
        final_annotated_roi_open_cv = roi_open_cv
        if cells.empty or geojson_data == []:
            return final_annotated_roi_open_cv

        for _, cell in cells.iterrows():
            centroid = (int(cell['Relative x_coord']), int(cell['Relative y_coord']))
            final_annotated_roi_open_cv = cv2.circle(roi_open_cv, centroid, centroid_size, color, -1)

        return final_annotated_roi_open_cv


    def _plot_boundaries(self, cells, geojson_data, morph_key, fill_cells, roi_img, color=[0, 0, 0]):
        roi_open_cv = np.array(roi_img)
        final_annotated_roi_open_cv = roi_open_cv.copy()

        if cells.empty or geojson_data == []:
            return final_annotated_roi_open_cv

        for idx in cells.index:
            boundary_coords = geojson_data[idx][morph_key]['coordinates'][0]
            pts = np.array(boundary_coords, np.int32)
            annotatedborders_roi_open_cv = cv2.polylines(roi_open_cv, [pts], True, color, 2)

            if fill_cells:
                annotatedfill_roi_open_cv = cv2.fillPoly(roi_open_cv, [pts], color)

        final_annotated_roi_open_cv = cv2.addWeighted(annotatedborders_roi_open_cv, 0.8, np.array(roi_img), 0.2, 0)
        if fill_cells:
            final_annotated_roi_open_cv = cv2.addWeighted(annotatedfill_roi_open_cv, 0.1, final_annotated_roi_open_cv, 0.9, 0)

        return final_annotated_roi_open_cv


    def show_image(self, change):
        with self.roi_output:
            clear_output(wait=True)
            self._interact_visualize(change)
           
            display(self.fig)
        

    def _open(self):
        marker = self.background_channel_widg.value
        roi = self.roi_selected

        # path to roi image (rgb)
        roi_path = os.path.join(self.rgb_dir, f'{self.sample_name}_ROI#{roi}_{marker}.ome.tif')
        roi_img = open_pil(roi_path)
        self.roi_img = roi_img

        # path to roi geojson
        geojson_path = os.path.join(self.geojson_dir, f'{self.sample_name}_ROI#{roi}_{marker}.json')

        if self.technology == 'if':
            geojson_path = os.path.join(self.geojson_dir, f'{self.sample_name}_ROI#{roi}_{self.dna_marker}.json')
            dapi_channel = os.path.join(self.rgb_dir, f'{self.sample_name}_ROI#{roi}_{self.dna_marker}.ome.tif')

            dapi_img = plt.imread(dapi_channel)
            dapi_img = dapi_img.copy()

            dapi_img = dapi_img.astype('float')

            blue = dapi_img[:, :, 2]
            np.multiply(blue, 4.5, out=blue, casting="unsafe")
            blue = np.clip(blue, 0, 255)
            dapi_img[:, :, 2] = blue
            dapi_img = dapi_img.astype('uint8')
            
            roi_img = np.array(roi_img)
            roi_img = roi_img.copy()
            roi_img = roi_img.astype('float')

            red = roi_img[:, :, 0]
            np.multiply(red, 5.25, out=red, casting="unsafe")
            red = np.clip(red, 0, 255)
            roi_img[:, :, 0] = red
            roi_img = roi_img.astype('uint8')

            roi_img = cv2.addWeighted(dapi_img, 1, roi_img, 0.9, 0)
            self.roi_img = roi_img


        if not os.path.exists(geojson_path):
            self.geojson_data = []
            print('There is no cells for the respective ROI. Some reasons for that:\n 1. Too much damage, the cells are not present across stains.')
            return

        with open(geojson_path, 'r') as stream:
            geojson_data = json.load(stream)
            self.geojson_data = geojson_data


    def display_tilemap(self):
        change = None
        self.show_image(change)
        def handle_click(trace, points, state):
            centroid = Point(points.xs[0], points.ys[0])
            self.roi_selected = tilestats_map[centroid]

            c = list(tilemap.data[0].marker.color)
            o = list(tilemap.data[0].marker.opacity)

            c = ['#36454f'] * len(c)
            #o = [0] * len(c)

            last_selection = points.point_inds[-1]
            c[last_selection] = '#1FFF5E'
            o[last_selection] = 1

            with tilemap.batch_update():
                tilemap.data[0].marker.color = c
                tilemap.data[0].marker.opacity = o
    
            self.show_image(change)

        roi_img = np.array(open_pil(self.tilemap_file))

        x = np.array([coord[0] for coord in self.tilestats_map.values()]) + (500 * 1.25 / self.output_resolution)
        y = np.abs(np.array([coord[1] for coord in self.tilestats_map.values()]) - roi_img.shape[0]) - (500 * 1.25 / self.output_resolution)

        tilemap = self._create_figure(roi_img, x, y, scale_factor=0.3, obj='tilemap')
        
        x = np.round((x * 0.3), decimals=2)
        y = np.round((y * 0.3), decimals=2)    
        #tilestats_map = {Point(coord[0], coord[1]): roi_index for roi_index, coord in enumerate(zip(x, y))}
        tilestats_map = {Point(coord[0], coord[1]): int(roi_index) for roi_index, coord in zip(self.tilestats_map.keys(), zip(x, y))}


        scatter = tilemap.data[0]
        scatter.on_click(handle_click)


        return tilemap


    def setup(self, sample_path, user_id):
        self._set_parameters(sample_path, user_id)
        widgs = self._get_initialize_widgets()

        self.pos_markers_widg = widgs['pos_markers']
        self.neg_markers_widg = widgs['neg_markers']
        self.background_channel_widg = widgs['background_channel']
        self.show_all_cells_widg = widgs['show_all_cells']
        self.display_morph_widg = widgs['display_morph']
        self.fill_cells_widg = widgs['fill_cells']

        ui = VBox([HBox([self.pos_markers_widg, self.neg_markers_widg, self.background_channel_widg]), self.show_all_cells_widg, self.display_morph_widg, self.fill_cells_widg])
        if self.technology == 'tma':
            self.select_core_widg = widgs['core']
            self.select_core_widg.observe(self.show_image, names='value')
            ui = VBox([HBox([self.pos_markers_widg, self.neg_markers_widg, self.background_channel_widg]), self.show_all_cells_widg, self.display_morph_widg, self.fill_cells_widg])
            ui = HBox([ui, self.select_core_widg])
            ui = Box(children=[ui], layout=Layout(overflow_y='scroll', overflow_x='scroll', height='500px', display='flex', align_items='flex-start'))


        tilemap = self.display_tilemap()

        self.pos_markers_widg.observe(self.show_image, names='value')
        self.neg_markers_widg.observe(self.show_image, names='value')
        self.display_morph_widg.observe(self.show_image, names='value')
        self.background_channel_widg.observe(self.show_image, names='value')
        self.show_all_cells_widg.observe(self.show_image, names='value')
        self.fill_cells_widg.observe(self.show_image, names='value')

        html = '''<header style="text-align:center">
                      <h1>Tilemap</h1>
                  </header>'''

        box_layout = Layout(overflow_x='scroll',
                            flex_flow='column',
                            display='flex',
                            align_items='center',
                            margin='100 100 100 100')
        box = Box(children=[tilemap], layout=box_layout)

        header1  = HTML(html) 
        display(header1)
        display(box)

        test = GridBox(children=[header1, box], 
                       layout=Layout(
                            margin='100 100 100 100',
                            grid_template_rows='auto auto',
                            grid_template_areas='''
                            "header1"
                            "box"
                    ''')
        )

        test.box_style = 'info'


        #display(HBox([tilemap_widg, ui]))
        test.layout.align_items = 'center'
        #display(test)

        self.roi_output.layout.width = '50%'

        box_layout = Layout(overflow_y='scroll',
                            flex_flow='column',
                            display='flex',
                            height=f'400px',
                            align_items='flex-start')
        metrics_box = Box(children=[self.metrics_output], layout=box_layout)

        
        ui = VBox([ui, metrics_box])
        ui.layout.width = '50%'

        display(HBox([self.roi_output, ui]))
        
        #out = interactive_output(self._interact_visualize, widgs) 
        #out.layout.width = '50%'

        #ui = VBox([HBox([widgs['roi_select'], widgs['pos_markers'], widgs['neg_markers'], widgs['background_channel']]), widgs['show_all_cells'], widgs['display_morph'], widgs['fill_cells']])

        #display(HBox([out, ui]))

        self.edit_output = Output()
            
        self.add_button = edit_button('Add cells')
        self.remove_button = edit_button('Remove cells')

        display(HBox([self.add_button, self.remove_button]))
        display(self.edit_output)     

        self.add_button.on_click(self.edit_cells)
        self.remove_button.on_click(self.edit_cells)


        widgs_ = {'pos_markers': fixed(widgs['pos_markers']),
                  'neg_markers': fixed(widgs['neg_markers']),
                  'background_channel': fixed(widgs['background_channel'])}

        return widgs_ 


    def edit_cells(self, change):
        with self.edit_output:
            clear_output(wait=True)
            if not self.pos_markers_widg.value and not self.neg_markers_widg.value:
                print('Please, select an expression pattern.')
                return

            if not self.selected_cells:
                print('No manually selected cells to edit.')
                return

            option = change.description

            #aux = self.finalcellinfo_df.loc[self.finalcellinfo_df['Tile index'] == self.roi_selected]
            aux = self.rois_data[self.roi_selected]
            x_coords = np.trunc(aux['Relative x_coord'])
            y_coords = np.trunc(aux['Relative y_coord'])
            cell_indexes = aux.index

            cell_map = pd.DataFrame({'Cell index': cell_indexes, 'Relative x_coord': x_coords, 'Relative y_coord': y_coords})
            for cell in self.selected_cells:
                x = np.trunc(cell[0] / self.roi_scale_factor)
                y = np.trunc((1000 - (cell[1] / self.roi_scale_factor)))
                idx = cell_map.loc[(cell_map['Relative x_coord'] == x) & (cell_map['Relative y_coord'] == y), 'Cell index']
                self._update_cell_expression(idx, option)

            #self.finalcellinfo_df.to_csv(self.finalcellinfo_file)
            #self.finalcellinfo_df = self._read()
            self.selected_cells = []

            change = None
            self.show_image(change)

            print('Cell expression manually changed. Please click "Save changes" when changes are done.')

    
    def _update_cell_expression(self, idx, option):
        for pos_marker in self.pos_markers_widg.value:
            self.rois_data[self.roi_selected].loc[idx, f'{pos_marker}_expression'] = True if option == 'Add cells' else False
            #self.finalcellinfo_df.loc[idx, f'{pos_marker}_expression'] = True if option == 'Add cells' else False
    
        for neg_marker in self.neg_markers_widg.value:
            self.rois_data[self.roi_selected].loc[idx, f'{neg_marker}_expression'] = True if option == 'Add cells' else False
            #self.finalcellinfo_df.loc[idx, f'{neg_marker}_expression'] = False if option == 'Add cells' else True


    def execute(self, **kwargs):
        sorted_rois = sorted(map(int, self.rois_data.keys()))
        sorted_cells = [self.rois_data[roi] for roi in sorted_rois]

        finalcellinfo_df = pd.concat(sorted_cells).reset_index(drop=True)
        finalcellinfo_df.to_csv(self.finalcellinfo_file)

        print('Changes saved in *_finalcellinfo.csv.')
