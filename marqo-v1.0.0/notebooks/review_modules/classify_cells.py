import os
import sys
import cv2
import yaml
import json
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import functools
import operator
from matplotlib import colors
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon
import colorsys
from skimage.color import rgb2hed, hed2rgb
import skimage.exposure as exposure

from ipywidgets import Button, HBox, VBox, interact, interactive, fixed, interact_manual, Image, Layout, interactive_output, Output, Box, GridBox, GridspecLayout, ToggleButton, Label, HTML
from IPython.display import display, clear_output, Javascript

from .modulewidgets import *
from PIL.Image import open as open_pil
from PIL.Image import fromarray

sys.path.append("..")
from clustering.kmeans import KMEANS


# evaluate the need to have an ABC to inherit from
class Classify:
    def __init__(self):
        self.button_name = 'Evaluate Marker'

    def _set_parameters(self, sample_path, user_id):
        self.config_file = os.path.join(sample_path, user_id, 'config.yaml')
        with open(self.config_file, 'r') as stream:
            config_data = yaml.safe_load(stream)
            
        try:
            self.sample_name = config_data['sample']['name']
            self.markers = config_data['sample']['markers']
            self.technology = config_data['sample']['technology']
            self.clusters_perc_medians = config_data['k-clusters']
            self.rgb_dir = config_data['directories']['rgb']
            self.geojson_dir = config_data['directories']['geojson'].replace('geojsons_perROI', 'geoJSONS_perROI')
            self.sample_dir = config_data['directories']['marco_output']
            self.output_resolution = config_data['parameters']['image']['output_resolution']
            self.marker_index_maskforclipping = config_data['parameters']['tiling']['marker_index_maskforclipping']
            self.marker_tile = config_data['sample']['markers'][self.marker_index_maskforclipping]
            self.cores_labels = config_data['sample']['cores_labels']

        # TODO: remove this try/except statement. Temp solution only for dealing with old config files
        except:
            self.sample_name = config_data['sample']['sample_name']
            self.markers = config_data['sample']['markers']
            self.technology = 'micsss'
            self.clusters_perc_medians = config_data['k-clusters']
            self.rgb_dir = config_data['directories']['product_directories']['rgb']
            self.geojson_dir = config_data['directories']['product_directories']['geojson'].replace('geojsons_perROI', 'geoJSONS_perROI')
            self.sample_dir = config_data['directories']['sample_directory']
            self.marker_index_maskforclipping = config_data['tiling_parameters']['marker_index_maskforclipping']
            self.marker_tile = config_data['sample']['markers'][self.marker_index_maskforclipping]
            self.output_resolution = config_data['parameters']['output_resolution']

        if self.cores_labels:
            self.cores = config_data['sample']['cores']

        if self.technology == 'if':
            channel = config_data['sample']['dna_marker_channel']
            self.dna_marker = self.markers[channel]
            del self.markers[channel]

        elif self.technology == 'cyif':
            self.cycles_metadata = config_data['cycles_metadata']
            marker_list_across_channels = [v['marker_name_list'] for k, v in self.cycles_metadata.items()]
            marker_list_across_channels = functools.reduce(operator.iconcat, marker_list_across_channels, [])  # flatten the list of lists
            cycles = [f'cycle_{i}' for i, v in self.cycles_metadata.items() for _ in range(len(v['marker_name_list']))]
            self.dna_marker_per_cycle = {f'cycle_{i}': f'cycle_{i}_{v["dna_marker"][0]}' for i, v in self.cycles_metadata.items()}
            self.markers = [f'{cycle}_{marker}' for marker, cycle in zip(marker_list_across_channels, cycles)]

            for _, dna_marker in self.dna_marker_per_cycle.items():
                self.markers.remove(dna_marker)


        self.finalcellinfo_file = os.path.join(sample_path, user_id, f'{self.sample_name}_finalcellinfo.csv')
        self.tilestats_file = os.path.join(sample_path, user_id, f'{self.sample_name}_tilestats.csv')
        self.tilemap_file = os.path.join(sample_path, f'{self.sample_name}_tilemap.ome.tif')
        #self.color_dict = self._get_color_dict()  
        #self.rep_tiles = self._get_rep_tiles()
        self.registered_mask = self.get_registered_mask()
        self.tilestats_map = self.open_tilestats()
        self.roi_selected = 0
        self.roi_output = Output()
        self.dist_output = Output()

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

    def get_registered_mask(self):
        mask_name = f'{self.sample_name}_{self.marker_tile}_thumbnail.png'
        mask_path = os.path.join(self.sample_dir, mask_name)

        image = open_pil(mask_path)
        metadata = image.text


        registered_mask = [float(value) if key != 'linear_shift' else json.loads(value) for key, value in metadata.items()]
        return registered_mask


    def _get_rep_tiles(self):
        #finalcellinfo_df = pd.read_csv(self.finalcellinfo_file)
        tiles = self.finalcellinfo_df['Tile index'].unique().tolist()

        rep_tiles_dict = {}
        for marker in self.markers:
            tiers_dict = {}
            tiers_dict_2 = {}
            for tile in tiles:
                subset = self.finalcellinfo_df.loc[self.finalcellinfo_df['Tile index'] == tile, [f'{marker}_tier']]
                #subset = self.finalcellinfo_df.loc[self.finalcellinfo_df['Tile index'] == tile, ['Cell index', f'{marker}_tier']]
                freq = subset[f'{marker}_tier'].value_counts()
                tiers = freq.index.unique().tolist()
                for tier in tiers:
                    count = freq.loc[freq.index == tier].values[0]

                    if tier not in tiers_dict:
                        tiers_dict[tier] = {tile: count}
                    
                    else:
                        tiers_dict[tier][tile] = count
   
            tiers_dict = {k: v for k, v in sorted(tiers_dict.items(), key=lambda item: item[0])}
            rep_tiles_dict[marker] = self._write_recommendation(tiers_dict, 5)
            
        return rep_tiles_dict


    def create_button(self, tile, count):
        width = '120px'
        height = '30px'
        button = Button(description=f'ROI {tile}', layout=Layout(width=width, height=height))
        button.value = tile

        # Create an HTML widget for styled text
        html = HTML(value=f"<div style='font-size: 10px; color: black; text-align: center; vertical-align: middle;'><b>{count} cells</b></div>", 
                            layout=Layout(width=width, height=height))

        # Function to handle button click
        def on_button_click(button):
            self.roi_selected = button.value

            c = list(self.tilemap.data[0].marker.color)
            o = list(self.tilemap.data[0].marker.opacity)

            c = ['#36454f'] * len(c)

            point_indices = list(self.tilestats_map.keys())
            last_selection = point_indices.index(f'{self.roi_selected}')

            c[last_selection] = '#1FFF5E'
            o[last_selection] = 1

            with self.tilemap.batch_update():
                self.tilemap.data[0].marker.color = c
                self.tilemap.data[0].marker.opacity = o

            change = None
            self._inspect_rois(change)

        # Assign the click event to the button
        button.on_click(on_button_click)

        # Create a VBox to hold the button and HTML widgets
        vbox = VBox([button, html], layout=Layout(width='auto', height=f'60px'))

        return vbox 

    
    def _write_recommendation(self, tiers_dict, n):
        # Create the grid
        rows = []
        for tier, value in tiers_dict.items():
            sorted_counts = {k: v for k, v in sorted(value.items(), key=lambda item: item[1], reverse=True)}
            label = HTML(value=f'<div style="text-align: center; vertical-align: middle; white-space: nowrap; background-color: {self.color_dict[tier]}; padding: 10px;">Tier {tier}</div>',
                         layout=Layout(width='auto', height='auto'))

            cols = [label]  # First column as labels
            i = 0
            for tile, count in sorted_counts.items():
                box = self.create_button(int(tile), count)
                cols.append(box)

                i += 1
                if i == n:
                    break

            rows.append(HBox(cols))

        grid = VBox(rows, layout=Layout(height=f'{len(tiers_dict) * 60}px'))
        #grid = VBox(rows, layout=Layout(height=f'{len(tiers_dict) * 60}px', width='50%', scroll='auto', display='flex'))

        return grid

    def _get_initialize_widgets(self):
        if self.cores_labels:
            widgs = {'marker': marker_channel(self.markers),
                     'core': select_core(self.cores)}
        else:
            widgs = {'marker': marker_channel(self.markers)}

        return widgs


    def _read(self):
        finalcellinfo_df = pd.read_csv(self.finalcellinfo_file, sep=',', header=0, index_col=['Cell index'])

        self.total_cells = len(finalcellinfo_df)

        #if self.cores_labels:
        #    finalcellinfo_df = finalcellinfo_df.loc[finalcellinfo_df['Core Label'] == self.core]

        return finalcellinfo_df


    def _sort_clusters(self, finalcellinfo_df):
        medians = self.clusters_perc_medians[self.marker]

        if self.cores_labels:
            #medians = {k: v for k, v in medians.items() if self.core in k}
            medians = medians[self.core]

        kcluster_dict = {}
        for k, v in medians.items():
            kcluster_dict[k] = v['nuc']['p5']
        
        medians = {float(k.split('-')[-2]): v for k, v in sorted(kcluster_dict.items(), key=lambda item: item[1], reverse=True)}
        
        rank = medians.keys()

        self.parent_clusters = set([int(c) for c in rank])

        return rank


    def _check_preexisting_pos(self, ranked_clusters, finalcellinfo_df):
        preexisting_pos_clusters = []
        for cluster in ranked_clusters:
            rows = finalcellinfo_df.loc[finalcellinfo_df[f'{self.marker}_tier'] == cluster, f'{self.marker}_expression']

            if rows.iloc[0] == True:
                preexisting_pos_clusters.append(cluster)
                
        return preexisting_pos_clusters

    def generate_shades(self, hex_color, num_shades=7):
        # Convert HEX to RGB
        h = hex_color.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

        # Convert RGB to HSL
        hsl = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)

        shades = []
        for i in range(1, num_shades + 1):
            # Vary the lightness
            lightness = hsl[1] * (1 - (0.1 * i))
            new_rgb = colorsys.hsv_to_rgb(hsl[0], hsl[1], lightness)

            # Convert back to HEX
            new_hex = '#%02x%02x%02x' % (int(new_rgb[0]*255), int(new_rgb[1]*255), int(new_rgb[2]*255))
            shades.append(new_hex)

        return shades


    def _get_color_dict(self):
        color_dict = {}

        num_of_tiers = []
        for marker in self.markers:
            n = np.max(self.finalcellinfo_df[f'{marker}_tier'].unique().tolist())
            num_of_tiers.append(n + 1)
    
        n = max(num_of_tiers)
        base_colors = sns.hls_palette(n).as_hex()
        #base_colors = ['#00FF00', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#FF0000', '#800080']
        for idx, color in enumerate(base_colors):
            #group_colors = sns.light_palette(color, 7).as_hex()[:-1]
            group_colors = self.generate_shades(color)
            group_colors.insert(0, color)
            group = [float(f'{idx}.{k}') for k in range(7)]
            color_dict.update({k: c for k, c in zip(group, group_colors)})
            
        return color_dict


    def setup(self, sample_path, user_id):
        self._set_parameters(sample_path, user_id)
       
        widgs = self._get_initialize_widgets()
        return widgs


    def _plot_distribution(self, change):
        with self.dist_output:
            clear_output(wait=True)
            marker = self.marker
            selected = []
            selected = [float(t.split(' ')[1]) for t in self.pos_tiers_widg.value]

            medians = self.clusters_perc_medians[marker]

            if self.cores_labels:
                #medians = {k: v for k, v in medians.items() if self.core in k}
                medians = medians[self.core]

            fig, axs = plt.subplots(1, 3, figsize=(21, 6))

            n_cells_list = []
            clusters_name = []
            for k, v in medians.items():
                n_cells = int(k.split('-')[-1])
                cluster = float(k.split('-')[-2])

                array_nuc = np.array([p for p in v['nuc'].values()])
                array_cyt = np.array([p for p in v['cyt'].values()])

                if selected == []:
                    n_cells_list.append(n_cells)
                    clusters_name.append(str(cluster))

                    sns.kdeplot(array_nuc, label=f'k-{cluster}', ax=axs[0], color=self.color_dict[cluster])
                    sns.kdeplot(array_cyt, label=f'k-{cluster}', ax=axs[1], color=self.color_dict[cluster])


                elif cluster in selected:
                    n_cells_list.append(n_cells)
                    clusters_name.append(str(cluster))

                    sns.kdeplot(array_nuc, label=f'k-{cluster}', ax=axs[0], color=self.color_dict[cluster])
                    sns.kdeplot(array_cyt, label=f'k-{cluster}', ax=axs[1], color=self.color_dict[cluster])


            aux_dict = {pop: name for pop, name in zip(n_cells_list, clusters_name)}
            aux_dict = dict(sorted(aux_dict.items(), reverse=True))
            clusters_name = [name for name in aux_dict.values()]
            n_cells_list = [pop for pop in aux_dict.keys()]

            labels = [f'{n_cell} ({n_cell / self.total_cells:.1%})' for n_cell in n_cells_list]
            bar_container = axs[2].bar(clusters_name, n_cells_list, width=0.8, color=[self.color_dict[float(k)] for k in clusters_name])
            #axs[2].bar_label(bar_container, fmt='{:,.0f}', rotation=45)
            axs[2].bar_label(bar_container, labels=labels, rotation=27, fontsize=8)
            axs[2].set_xlim(-1, len(clusters_name))
            axs[2].set_title(f"Cells per Cluster from a total of {sum(n_cells_list)}")
            axs[2].tick_params('x', labelrotation=27)
            axs[2].set_xlabel("Cluster Label")
            axs[2].set_ylabel("Cell Number")
            axs[0].set_title('Chromogen Nuc Distribution')
            axs[1].set_title('Chromogen Cyto Distribution')
            axs[0].legend()
            axs[1].legend()
            plt.show()



    def _color_if(self, channel, hex_color, brightness=1.0):
        channel = channel.copy()

        if len(channel.shape) == 3 and channel.shape[2] == 3:
            channel = cv2.cvtColor(channel, cv2.COLOR_RGB2GRAY)

        channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        channel = cv2.cvtColor(channel, cv2.COLOR_GRAY2RGB)

        # Convert hex to RGB
        desired_color = tuple(int(hex_color[i: i+2], 16) for i in (1, 3, 5))

        # Calculate scaling factors for each channel
        scaling_factors = [component / 255.0 for component in desired_color]

        colored_channel = channel * np.array(scaling_factors)
        #colored_channel = channel[:, :, np.newaxis] * np.array(scaling_factors)

        # colored_channel = colored_channel * brightness

        brightened_image = exposure.adjust_gamma(colored_channel, brightness)

        colored_channel = np.clip(brightened_image, 0, 255).astype(np.uint8)
        
        return colored_channel

    def _color_if_(self, channel, hex_color, brightness=1.0):
        channel = channel.copy()

        if len(channel.shape) == 3 and channel.shape[2] == 3:
            channel = cv2.cvtColor(channel, cv2.COLOR_RGB2GRAY)

        channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        channel = np.invert(channel)

        channel = cv2.cvtColor(channel, cv2.COLOR_GRAY2RGB)

        # Convert hex to RGB
        desired_color = tuple(int(hex_color[i: i+2], 16) for i in (1, 3, 5))

        intensity = np.dot(channel[..., :3], [0.2989, 0.5870, 0.1140])
        
        # Prepare the new image with a white background
        new_img = np.ones_like(channel) * 255
        
        # Apply blue color to cells based on their intensity
        new_img[..., 0] = (1 - intensity / 255) * desired_color[0] + intensity
        new_img[..., 1] = (1 - intensity / 255) * desired_color[1] + intensity
        new_img[..., 2] = intensity * (desired_color[2] / 255)

        return new_img



    def _inspect_rois(self, change):
        with self.roi_output:
            clear_output(wait=True)

            roi = self.roi_selected
            marker = self.marker
            display_morph = self.display_morph_widg.value
            fill_cells = self.fill_cells_widg.value
            channel = self.channel_radio_buttons.value
            pos_tiers = [float(t.split(' ')[1]) for t in self.pos_tiers_widg.value]

            centroid_size = 10
            fig = plt.figure(figsize=(10, 10), dpi=100)
            graphs = fig.subplots(1, 1)
            graphs.set_title(f'ROI #{roi} for marker {marker}')

            roi_path = os.path.join(self.rgb_dir, f'{self.sample_name}_ROI#{roi}_{marker}.ome.tif')        

            if self.technology != 'if' and self.technology != 'cyif':
                # path to roi image (rgb)
                roi_img = plt.imread(roi_path)

                # path to roi geojson
                geojson_path = os.path.join(self.geojson_dir, f'{self.sample_name}_ROI#{roi}_{marker}.json') 
                if channel != 'RGB':
                    ihc_hed = rgb2hed(roi_img)
                    null = np.zeros_like(ihc_hed[:, :, 0])

                    if channel == 'Hematoxylin':
                        ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
                        ihc_h = (ihc_h * 255).astype(np.uint8)
                        roi_img = ihc_h

                    elif channel == 'AEC':
                        ihc_a = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
                        ihc_a = (ihc_a * 255).astype(np.uint8)
                        roi_img = ihc_a

            elif self.technology == 'if':
                geojson_path = os.path.join(self.geojson_dir, f'{self.sample_name}_ROI#{roi}_{self.dna_marker}.json') 
                dapi_channel = os.path.join(self.rgb_dir, f'{self.sample_name}_ROI#{roi}_{self.dna_marker}.ome.tif')        
    
                dapi_img = plt.imread(dapi_channel)
                dapi_brightness = 1.1
                dapi_colored = self._color_if(dapi_img, '#004EFF', brightness=dapi_brightness)

                chromogen_img = plt.imread(roi_path)
                chromogen_brightness = 1.1
                chromogen_colored = self._color_if(chromogen_img, '#E10600', brightness=chromogen_brightness)

                if channel == 'DNA + chromogen':
                    roi_img = cv2.addWeighted(chromogen_colored, 0.9, dapi_colored, 1, 0)

                elif channel == 'DNA':
                    roi_img = dapi_colored

                elif channel == 'chromogen':
                    roi_img = chromogen_colored

            elif self.technology == 'cyif':
                cycle = '_'.join(marker.split('_')[:2])
                dna_marker = self.dna_marker_per_cycle[cycle]
                geojson_path = os.path.join(self.geojson_dir, f'{self.sample_name}_ROI#{roi}_{marker}.json') 
                dapi_channel = os.path.join(self.rgb_dir, f'{self.sample_name}_ROI#{roi}_{dna_marker}.ome.tif')        
        
    
                dapi_img = plt.imread(dapi_channel)
                dapi_brightness = 1.1
                dapi_colored = self._color_if(dapi_img, '#004EFF', brightness=dapi_brightness)

                chromogen_img = plt.imread(roi_path)
                chromogen_brightness = 1.1
                chromogen_colored = self._color_if(chromogen_img, '#E10600', brightness=chromogen_brightness)

                if channel == 'DNA + chromogen':
                    roi_img = cv2.addWeighted(chromogen_colored, 0.9, dapi_colored, 1, 0)
    
                elif channel == 'DNA':
                    roi_img = dapi_colored

                elif channel == 'chromogen':
                    roi_img = chromogen_colored

            if not os.path.exists(geojson_path):
                graphs.imshow(roi_img)
                print('There is no cells for the respective ROI. Some reasons for that:\n 1. Too much damage, the cells are not present across stains.')
                plt.show()
                return
                
            with open(geojson_path, 'r') as stream:
                geojson_data = json.load(stream)

            if len(geojson_data) == 0:
                graphs.imshow(roi_img)
                print('No cells to display for the respective ROI. Cells pruned due to denoising step (elimination of cells from the edges)')
                plt.show()
                return

            if display_morph == 'None':
                graphs.imshow(roi_img)

            elif display_morph == 'Centroid':
                graphs.imshow(roi_img)

                for pos_tier in pos_tiers:
                    pos_cells = self.finalcellinfo_df.loc[(self.finalcellinfo_df['Tile index'] == roi) & (self.finalcellinfo_df[f'{marker}_tier'] == pos_tier)]
                    y_coord = pos_cells['Relative y_coord']
                    x_coord = pos_cells['Relative x_coord']
                    
                    graphs.scatter(x_coord, y_coord, c=self.color_dict[pos_tier], s=centroid_size)

            else:
                morph_key = {'Nucleus': 'nucleusGeometry', 'Cytoplasm': 'geometry'}[display_morph]
           
                roi_open_cv = roi_img[:, :, :].copy() 
                final_annotated_roi_open_cv = roi_open_cv

                for pos_tier in pos_tiers:
                    all_cells = self.finalcellinfo_df.loc[self.finalcellinfo_df['Tile index'] == roi].reset_index()
                    pos_cells = all_cells.loc[all_cells[f'{marker}_tier'] == pos_tier]
                    cell_indexes = pos_cells.index

                    # check if there is cell
                    if not pos_cells.empty:
                        color = colors.hex2color(self.color_dict[pos_tier])
                        color = [c * 255 for c in color]
                        
                        for idx in cell_indexes:
                            boundary_coords = geojson_data[idx][morph_key]['coordinates'][0]
                            pts = np.array(boundary_coords, np.int32)
                            annotatedborders_roi_open_cv = cv2.polylines(roi_open_cv, [pts], True, color, 2)
                            if fill_cells:
                                annotatedfill_roi_open_cv = cv2.fillPoly(roi_open_cv, [pts], color)

                        final_annotated_roi_open_cv = cv2.addWeighted(annotatedborders_roi_open_cv, 0.8, roi_img, 0.2, 0)
                        if fill_cells:
                            final_annotated_roi_open_cv = cv2.addWeighted(annotatedfill_roi_open_cv, 0.4, final_annotated_roi_open_cv, .6, 0)

                graphs.imshow(final_annotated_roi_open_cv)

            plt.show()


    def _regroup(self, root):
        kmeans = KMEANS()
        kmeans._set_parameters(self.config_file)

        if hasattr(self, 'core'):
            kmeans.regroup(root, self.marker, self.finalcellinfo_file, self.core)

        else:
            kmeans.regroup(root, self.marker, self.finalcellinfo_file)

        with open(self.config_file, 'r') as stream:
            config_data = yaml.safe_load(stream)

        self.clusters_perc_medians = config_data['k-clusters']

        print('Choosen cluster regroup sucessfully.')


    def _recluster(self, **kwargs):
        pos_tiers = [float(t.split(' ')[1]) for t in kwargs['pos_tiers'].value]

        if 0 < len(pos_tiers) < 2:
            tier = pos_tiers[0]
            if tier > int(tier):
                print('Error: You can not recluster a subcluster.') 
                return           

            kmeans = KMEANS()
            kmeans._set_parameters(self.config_file)

            if hasattr(self, 'core'):
                kmeans.reclustering(tier, self.marker, self.finalcellinfo_file, self.core)

            else:
                kmeans.reclustering(tier, self.marker, self.finalcellinfo_file)

            # update clusters_perc_medians attribute with the new data from reclustering
            with open(self.config_file, 'r') as stream:
                config_data = yaml.safe_load(stream)

            self.clusters_perc_medians = config_data['k-clusters']

            print('Reclustering finished.')

        else:
            if len(pos_tiers) == 0:
                print('Kindly select one cluster to recluster.')

            else:
                print('Kindly select only one cluster.')


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

        c = ['#36454f'] * len(x)
        c[self.roi_selected] = '#1FFF5E'
        fig.data[0].marker.color = c

        o = [0] * len(x)
        o[self.roi_selected] = 1
        fig.data[0].marker.opacity = o

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
            hovermode='closest'
        )

        return fig


    def display_tilemap(self):
        change = None
        self._inspect_rois(change)
        self._plot_distribution(change)
        def handle_click(trace, points, state):
            centroid = Point(points.xs[0], points.ys[0])
            self.roi_selected = int(tilestats_map[centroid])

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

            self._inspect_rois(change)

        roi_img = np.array(open_pil(self.tilemap_file))


        # The point indices list has the same order as x,y used to create the figure

        x = np.array([coord[0] for coord in self.tilestats_map.values()]) + (500 * 1.25 / self.output_resolution)
        y = np.abs(np.array([coord[1] for coord in self.tilestats_map.values()]) - roi_img.shape[0]) - (500 * 1.25 / self.output_resolution)

        tilemap = self._create_figure(roi_img, x, y, scale_factor=0.3)

        x = np.round((x * 0.3), decimals=1)
        y = np.round((y * 0.3), decimals=1)
        tilestats_map = {Point(coord[0], coord[1]): roi_index for roi_index, coord in zip(self.tilestats_map.keys(), zip(x, y))}

        scatter = tilemap.data[0]
        scatter.on_click(handle_click)

        return tilemap

    
    def _classify(self, **kwargs):
        pos_tiers = [float(t.split(' ')[1]) for t in kwargs['pos_tiers'].value]
        marker = kwargs['marker']

        finalcellinfo = pd.read_csv(self.finalcellinfo_file, sep=',', header=0, index_col=['Cell index'])
        self.finalcellinfo_df[f'{marker}_expression'] = False

        for pos_tier in pos_tiers:
            self.finalcellinfo_df.loc[self.finalcellinfo_df[f'{marker}_tier'] == pos_tier, f'{marker}_expression'] = True
              
        # lets check all markers evaluated for current sample
        evaluated_markers = []
        not_evaluated_markers = []
        expression_cols = self.finalcellinfo_df.columns[self.finalcellinfo_df.columns.str.contains('expression')]
        for col in expression_cols:
            if not pd.isna(self.finalcellinfo_df[col].iloc[0]):
                evaluated_markers.append(col.split('_')[0])
                
            else:
                not_evaluated_markers.append(col.split('_')[0])
        
        # TODO: updating geojsons, probably on Exporting module
        
        #import and update tilestats.csv
        df_tilestats = pd.read_csv(self.tilestats_file, sep=',', header=0, index_col=['Tile index']) #use ROI column as indices
        df_tilestats[(marker + ' #poscells/microns2')] = 0

        for roi_index in self.finalcellinfo_df['Tile index'].unique().tolist():
            current_roi_area = df_tilestats.loc[roi_index, 'Tissue area (microns2)']
            num_poscells_formarker = np.sum(self.finalcellinfo_df.loc[self.finalcellinfo_df['Tile index'] == roi_index, (marker + '_expression')])
            poscells_perarea_formarker = num_poscells_formarker / current_roi_area
            df_tilestats.loc[roi_index, (marker + ' #poscells/microns2')] = poscells_perarea_formarker       
                
        df_tilestats.to_csv(self.tilestats_file)


        #TODO: need a better way to update, iteratively by index dont change the dtype but is slow
        # the main concern is regarding TMA that we want update only a subset of rows and not the whole column
        indices = self.finalcellinfo_df.index
        for i in indices:
            finalcellinfo.loc[i, f'{marker}_expression'] = self.finalcellinfo_df.loc[i, f'{marker}_expression']

       	#finalcellinfo.update(self.finalcellinfo_df)
        finalcellinfo.to_csv(self.finalcellinfo_file, index=True)

        #provide output notification
        print('\'' + self.sample_name + '_finalcellinfo.csv\' has been updated with ' + marker + ' review.')
        print('')
        print('Marker(s) still not reviewed: ' + str(not_evaluated_markers))
        print('Marker(s) successfully reviewed: ' + str(evaluated_markers))
        print('')


    def on_channel_change(self, change):
        if change['name'] == 'value' and change['new'] is not None:
            print(f'Selected {change["new"]}')


    def _format_table(self, rep_tiles):
        output = Output()
        with output:
            table = f'''
                    <table>
                        <caption>Representative ROIs per Tier</caption>
                        <tbody>
                            {rep_tiles}
                        </tbody>
                    </table>
                    '''
            display(HTML(table))
        
        return output

    def execute(self, **kwargs):
        self.marker = kwargs['marker']
        self.finalcellinfo_df = self._read()

        if self.cores_labels:
            self.core = kwargs['core']
            self.finalcellinfo_df = self.finalcellinfo_df.loc[self.finalcellinfo_df['Core Label'] == self.core]

        self.color_dict = self._get_color_dict()
        self.rep_tiles = self._get_rep_tiles()

        # sorting clusters to display in list widget
        ranked_clusters = self._sort_clusters(self.finalcellinfo_df)

        # identify positive clusters (if already classified)
        preexisting_pos_clusters = self._check_preexisting_pos(ranked_clusters, self.finalcellinfo_df)
   
        # instantiating widgets 
        self.pos_tiers_widg = pos_tiers(ranked_clusters, preexisting_pos_clusters)
        self.display_morph_widg = display_morph()
        self.fill_cells_widg = fill_cells()

        self.channel_radio_buttons = channel_radio_buttons(self.technology)

        self.tilemap = self.display_tilemap()
        box_layout = Layout(overflow_x='scroll',
                    flex_flow='column',
                    display='flex',
                    width='50%',
                    align_items='flex-start')

        tilemap_box = Box(children=[self.tilemap], layout=box_layout)

        rep_table = self.rep_tiles[self.marker]
        table_layout = Layout(scroll='auto',
                            display='flex',
                            align_items='flex-start',
                            width='50%')
        rep_table_box = Box(children=[rep_table], layout=table_layout)


        #rep_table_widget.layout.width = '50%'
        display(HBox([tilemap_box, rep_table_box], layout=Layout(height=f'{self.tilemap.layout.height}px')))
        #display(HBox([self.tilemap, rep_table], layout=Layout(height=f'{self.tilemap.layout.height}px')))

        self.pos_tiers_widg.observe(self._inspect_rois, names='value')
        self.display_morph_widg.observe(self._inspect_rois, names='value')
        self.fill_cells_widg.observe(self._inspect_rois, names='value')
        self.channel_radio_buttons.observe(self._inspect_rois, names='value')

        #out = interactive_output(self._inspect_rois, widgs)
        self.roi_output.layout.width = '50%'
        ui = VBox([HBox([self.pos_tiers_widg]), VBox([self.display_morph_widg]), self.fill_cells_widg, self.channel_radio_buttons])
        display(HBox([self.roi_output, ui]))

        # plotting chromogen distribution for each cluster
        self.pos_tiers_widg.observe(self._plot_distribution, names='value')
        display(HBox([self.dist_output]))


        # display button to perform re-clustering
        # TMP: num of clusters only make sense for normal kmeans
        num_clusters = clusters_number()

        widgs = {'pos_tiers': fixed(self.pos_tiers_widg)}
        #         'num_clusters': num_clusters}

        my_interact_manual = interact_manual.options(manual_name="Recluster")
        my_interact_manual(self._recluster, **widgs)

        
        root = root_cluster(self.parent_clusters)        
            
        my_interact_manual = interact_manual.options(manual_name="Regroup")
        my_interact_manual(self._regroup, root=root)

        # display button to clasify cells (positive tiers selected)
        widgs = {'marker': fixed(self.marker), 'pos_tiers': fixed(self.pos_tiers_widg)}       

        my_interact_manual = interact_manual.options(manual_name="Classify cells")
        my_interact_manual(self._classify, **widgs)

