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
from matplotlib import colors
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import STRtree
from matplotlib.patches import Rectangle
import pandas as pd
import geopandas as gpd
from rtree import Index
from scipy.spatial import cKDTree
from shapely.validation import explain_validity, make_valid

from ipywidgets import Button, HBox, VBox, interact, interactive, fixed, interact_manual, Image, Layout, interactive_output, Output
from .modulewidgets import *
from time import time

# evaluate the need to have an ABC to inherit from
class Reconcile:
    def __init__(self):
        self.button_name = 'Visualize Annotations'
        self.tissue_annotations = []
        self.tissue_coords = []
        self.name_counts = {}

    def _set_parameters(self, sample_path, user_id):
        self.config_file = os.path.join(sample_path, user_id, 'config.yaml')

        with open(self.config_file, 'r') as stream:
            config_data = yaml.safe_load(stream)
       
        try: 
            self.sample_name = config_data['sample']['name']    
            self.markers = config_data['sample']['markers']
            self.output_resolution = config_data['parameters']['image']['output_resolution']
            self.scale_factor = config_data['parameters']['image']['scale_factor']

        except:
            self.sample_name = config_data['sample']['sample_name']    
            self.markers = config_data['sample']['markers']
            self.output_resolution = config_data['parameters']['output_resolution']
            self.scale_factor = config_data['parameters']['SF']


        self.finalcellinfo_file = os.path.join(sample_path, user_id, f'{self.sample_name}_finalcellinfo.csv')
        self.summary_path = os.path.join(sample_path, user_id, f'{self.sample_name}_pathologysummary.csv')
        self.shape_file_path = os.path.join(sample_path, user_id, f'{self.sample_name}_spatial_data.shp')

    def _get_initialize_widgets(self):
        widgs = {'original_image': original_image(),
                 'annotation_geojson': annotation_geojson(),
                 'marker_dropdown': marker_dropdown(self.markers)}

        return widgs


    def _sanity_check(self, **kwargs):
        original_image_path = kwargs['original_image']
        if not os.path.exists(original_image_path):
            print('Please ensure image file path is correct.')
            return

        annotation_geojson_path = kwargs['annotation_geojson']
        if not os.path.exists(annotation_geojson_path):
            print('Please ensure annotation geojson file path is correct.')
            return


    def _open_json(self, json_file):
        with open(json_file, 'r') as stream:
            json_data = json.load(stream)

        return json_data


    def setup(self, sample_path, user_id):
        self._set_parameters(sample_path, user_id)

        widgs = self._get_initialize_widgets()
        return widgs

       
    def _make_valid(self, polygon):
        geometry = make_valid(polygon)
        if geometry.geom_type == 'GeometryCollection':
            collection = [geom for geom in geometry.geoms if geom.geom_type != 'LineString' and geom.geom_type != 'MultiLineString']

        elif geometry.geom_type == 'MultiPolygon':
            collection = [geom for geom in geometry.geoms if geom.geom_type != 'MultiLineString'] 

        else:
            collection = []

        return collection

    def _check_validity(self, poly):
        if poly.is_valid:
            return [poly]

        elif not poly.is_valid:
            collection = self._make_valid(poly)
            return collection

    def _unnest_polygons(self, polygons_list):
        gdf = gpd.GeoDataFrame(geometry=polygons_list)
        strtree = STRtree(gdf.geometry)

        # Set to track inner polygons
        inner_polygons_set = set()

        # Dictionary to track outer polygons and their corresponding inner polygons
        outer_to_inner = {}

        for idx, polygon in enumerate(gdf.geometry):
            possible_outer_indices = strtree.query(polygon)
            for outer_idx in possible_outer_indices:
                other_polygon = gdf.geometry.iloc[outer_idx]
                if idx != outer_idx and other_polygon.contains(polygon):
                    outer_to_inner.setdefault(outer_idx, []).append(idx)
                    inner_polygons_set.add(idx)
                    break

        # Perform the difference operation for outer polygons with inner polygons
        result_polygons = []
        for outer_idx, inner_indices in outer_to_inner.items():
            outer_polygon = gdf.geometry.iloc[outer_idx]
            inner_polygons = [gdf.geometry.iloc[i] for i in inner_indices]
            combined_inner = unary_union(inner_polygons)
            result_polygon = outer_polygon.difference(combined_inner)
            result_polygons.append(result_polygon)

        # Add outer polygons that do not have any inner polygons
        outer_polygons_without_inner = set(range(len(gdf))) - inner_polygons_set
        for idx in outer_polygons_without_inner:
            if idx not in outer_to_inner:  # Only include if not already processed as an outer polygon
                result_polygons.append(gdf.geometry.iloc[idx])

        flatten_polys = []
        for poly in result_polygons:
            if isinstance(poly, MultiPolygon):
                flat_multi_poly = [polygon for polygon in poly.geoms]
                flatten_polys.extend(flat_multi_poly)
            else:
                flatten_polys.append(poly)

        return flatten_polys

    def _generate_unique_names(self, name):
        if name in self.name_counts:
            self.name_counts[name] += 1
            return f'{name}_{self.name_counts[name]}'
        
        else:
            self.name_counts[name] = 0
            return name


    def _get_tissue_types(self, annotation_geojson_file):
        '''
        Function to parse annotation_elements ans extract Tissue type (annotation name) and coords for the respective tissue
        '''
        geojson_data = self._open_json(annotation_geojson_file)

        annotation_elements = geojson_data['features'] if 'features' in geojson_data else geojson_data

        rows = []
        for idx, annotation_item in enumerate(annotation_elements):
            tissue_type = annotation_item['properties']['classification']['name']

            if annotation_item['geometry']['type'] == 'MultiPolygon':

                polygons_list = []
                for poly in annotation_item['geometry']['coordinates']:
                    tissue_type = annotation_item['properties']['classification']['name']

                    if len(poly) == 1:
                        polygon = Polygon(poly[0])
                        collection = self._check_validity(polygon)
                        polygons_list.extend(collection)

                    else:
                        for p in poly:
                            polygon1 = Polygon(p)
                            collection = self._check_validity(polygon1)
                            polygons_list.extend(collection)

                resulting_polygons = self._unnest_polygons(polygons_list)

                multi_polygon = MultiPolygon(resulting_polygons)

                row = [multi_polygon, tissue_type]

            elif annotation_item['geometry']['type'] == 'Polygon':
                coords = annotation_item['geometry']['coordinates'][0]
                poly = Polygon(coords)
                row = [poly, tissue_type]

            rows.append(row)

        df = pd.DataFrame(data=rows, columns=['geometry', 'annotation'])
        df['annotation'] = df['annotation'].apply(self._generate_unique_names)

        self.ann_objects_gdf = gpd.GeoDataFrame(df, geometry='geometry')



    def separate_annotations_(self):
        poly_list = self.ann_objects_gdf['geometry'].tolist()
        names_list = self.ann_objects_gdf['annotation'].tolist()

        centroids = self.ann_objects_gdf['geometry'].apply(lambda g: [g.centroid.x, g.centroid.y]).tolist()

        kdtree = cKDTree(centroids)
        dist, neighborhoods = kdtree.query(centroids, k=len(centroids))

        for idx, polygon in enumerate(poly_list):
            index.insert(idx, polygon.bounds)

        
        rows = []
        for idx, neigh in enumerate(neighborhoods):
            polygon = poly_list[idx]
            tissue_annotation = names_list[idx]
            
            for n in neigh:
                if idx != n:
                    #if not polygon.intersection(poly_list[n]):
                    #    break

                    if polygon.contains(poly_list[n]):   
                        polygon = polygon - poly_list[n]

            row = [polygon, tissue_annotation]
            rows.append(row)
            break

        df = pd.DataFrame(data=rows, columns=['geometry', 'annotation'])
        ann_objects_gdf = gpd.GeoDataFrame(df, geometry='geometry')

        return ann_objects_gdf

    def separate_annotations(self):
        poly_list = self.ann_objects_gdf['geometry'].tolist()
        names_list = self.ann_objects_gdf['annotation'].tolist()

        tree = STRtree(self.ann_objects_gdf['geometry'].buffer(1e-14))  # small buffer to prevent false negatives because of shapely round errors

        combinations  = tree.query(poly_list, predicate='contains')
        outer, inner = combinations

        rows = []

        outer_index = outer[0]
        outer_poly = poly_list[outer_index]
        tissue_annotation = names_list[outer[0]]
        for i, j in zip(outer, inner):
            poly1 = poly_list[i]
            inner_poly = poly_list[j]

            if i != outer_index:
                row = [outer_poly, tissue_annotation]
                rows.append(row)
    
                outer_index = i
                outer_poly = poly1  # change of outer poly
                tissue_annotation = names_list[i]

            if i != j:  # every poly contains itself, so we need to check i and j
                outer_poly = outer_poly - inner_poly  # update of the outer poly

        row = [outer_poly, tissue_annotation]
        rows.append(row)


        df = pd.DataFrame(data=rows, columns=['geometry', 'annotation'])
        ann_objects_gdf = gpd.GeoDataFrame(df, geometry='geometry')

        return ann_objects_gdf


    def separate_annotations_(self):
        """
        If one annotation area resides within another, we need to take the difference. If not, the same that belongs to the 
        small annotation area will also belong to the bigger annotation area
        """
        index = Index()
        poly_list = self.ann_objects_gdf['geometry'].tolist()
        names_list = self.ann_objects_gdf['annotation'].tolist()

        # First loop just to create the RTree
        for idx, polygon in enumerate(poly_list):
            index.insert(idx, polygon.bounds)

        rows = []
        i = 0
        for idx, polygon in enumerate(poly_list):
            tissue_annotation = names_list[idx]
            intersections = index.contains(polygon.bounds)
            for pos in intersections:
                if idx != pos:
                    if polygon.contains(poly_list[pos]):
                        polygon = polygon - poly_list[pos]

            row = [polygon, tissue_annotation]
            rows.append(row)

        df = pd.DataFrame(data=rows, columns=['geometry', 'annotation'])
        ann_objects_gdf = gpd.GeoDataFrame(df, geometry='geometry')

        return ann_objects_gdf


    def overlay_tissue_types(self, img_source, img_metadata):
        OG_width = img_metadata['sizeX']
        OG_height = img_metadata['sizeY']
        base_res = img_metadata['magnification']
        region = img_source.getRegionAtAnotherScale(sourceRegion=dict(left=0, top=0, width=OG_width, height=OG_height, units='base_pixels'), targetScale=dict(magnification=1.25), format=large_image.tilesource.TILE_FORMAT_NUMPY)
        image_array = np.asarray(region[0])
        image_open_cv = image_array[:, :, :].copy()

        SF_to_thumb = 1.25 / base_res

        legend_color = []
        labels = []
        tissue_annotations = self.ann_objects_gdf['annotation'].unique().tolist()
        n_tissue_types = len(tissue_annotations)
        color_palette = sns.color_palette("bright", n_tissue_types)

        color_dict = {annotation: color for annotation, color in zip(tissue_annotations, color_palette)}

        #fig = plt.figure(figsize=(10, 10), dpi=100)
        fig, ax = plt.subplots(frameon=False, figsize=(10, 10))
        ax.axis('off')
        ax.set_title('Continue to next step if visualization looks correct.')
        ax.imshow(image_open_cv)

        for annotation in tissue_annotations:
            color = color_dict[annotation]
            polygons = self.ann_objects_gdf.loc[self.ann_objects_gdf['annotation'] == annotation]
            polygons = polygons.scale(SF_to_thumb, SF_to_thumb, origin=(0, 0))
            polygons.plot(ax=ax, color=color, alpha=0.2)
            legend_color.append(color)
            labels.append(annotation)

        handles = [Rectangle((0,0), 1, 1, color=(color[0], color[1], color[2]), alpha=0.2) for color in legend_color]
        plt.legend(handles, labels)
        plt.show()
 

    def resolver(self):
        all_polys = []
        idx = 0
        for tissue_type, coords in zip(self.tissue_annotations, self.tissue_coords):
            current_polygon = Polygon(coords)
            for _idx, _coords in enumerate(self.tissue_coords):
                if idx != _idx:
                    _p = Polygon(_coords)
                    if current_polygon.contains(_p):
                        current_polygon = current_polygon - _p

            idx += 1
            xx, yy = current_polygon.exterior.coords.xy

            coord = []
            for x, y in zip(xx, yy):
                coord.append((x, y))

            all_polys.append(coord)

        return all_polys

    def reconcile(self, **kwargs):
        finalcellinfo_df = pd.read_csv(self.finalcellinfo_file, sep=',', header=0)
        
        conversion = self.base_res / self.output_resolution


        cell_indexes = finalcellinfo_df['Cell index'].tolist()
        cells_x_coords = finalcellinfo_df[f'{self.marker}_x-coord (pixels)'] * conversion
        cells_y_coords = finalcellinfo_df[f'{self.marker}_y-coord (pixels)'] * conversion

        df = pd.DataFrame()
        df['Cell index'] = cell_indexes
        df['coords'] = list(zip(cells_x_coords, cells_y_coords))
        df['coords'] = df['coords'].apply(Point)

        cells_gdf = gpd.GeoDataFrame(df, geometry='coords')

        points_in_polygons = gpd.tools.sjoin(cells_gdf, self.ann_objects_gdf, predicate='within', how='left') 
        points_in_polygons = points_in_polygons[['Cell index', 'annotation']]

        a = points_in_polygons.groupby('Cell index')['annotation'].apply(lambda x: ','.join([str(i) for i in x])).reset_index()
        a['annotation'] = a['annotation'].replace('nan', 'Undefined')

        annotation_column = a['annotation']


        # This block is to remove duplications -- but we're gonna try a different approach
        #duplicated_cells = points_in_polygons[points_in_polygons.index.duplicated(keep=False)]
        #overlapped_ann = duplicated_cells['annotation'].unique().tolist()        
        #msg = self._get_assignment(duplicated_cells)        
        #print(f'WARNING: {len(duplicated_cells) / 2} cells were detected in two overlapped annotations.\n')
        #print(msg)
        #print('If the automatic assignemnt is wrong, please correct the original annotation.')
        #unique_cells = points_in_polygons.index.drop_duplicates(keep='first')
        #annotation_column = points_in_polygons.iloc[unique_cells]['annotation'].tolist()
        
        print('\nWait few seconds while the reconciliation data is being written ...')
        finalcellinfo_df['Tissue annotation'] = annotation_column
        finalcellinfo_df.to_csv(self.finalcellinfo_file, index=False)

        self.create_pathology_summary() 

    
    def _get_assignment(self, duplicated_cells):

        first = duplicated_cells[duplicated_cells.index.duplicated(keep='last')]  # the first occurrence is True and the last False
        last = duplicated_cells[duplicated_cells.index.duplicated(keep='first')]  # vice-verse


        merged = first[['Cell index', 'annotation']].merge(last[['Cell index', 'annotation']], on='Cell index')
        merged = merged[['annotation_x', 'annotation_y']].value_counts(ascending=True).reset_index(name='count')
        
        assignment = ''
        for idx, row in merged.iterrows():
            first = row['annotation_x']
            last = row['annotation_y']
            count = row['count']

            assignment += f'1. {first} -- 2. {last}  | Cells on both regions: #{count}\n'

        assignment += '\nThe cells were automatically assigned to the first annotation.'
        return assignment

    def create_pathology_summary(self):
        annotations = self.ann_objects_gdf['annotation'].unique().tolist()

        rows = []
        whole_tissue_px = 0
        whole_tissue_um = 0
        for ann in annotations:
            tissue_area_px = self.ann_objects_gdf.loc[self.ann_objects_gdf['annotation'] == ann, 'geometry'].area.sum()
            tissue_area_um = tissue_area_px * (self.pixel_size ** 2)

            row = [ann, tissue_area_px, tissue_area_um]
            rows.append(row)

        whole_tissue_px = unary_union(self.ann_objects_gdf['geometry']).area
        whole_tissue_um = whole_tissue_px * (self.pixel_size ** 2)

        row = ['whole_tissue', whole_tissue_px, whole_tissue_um]
        rows.append(row)

        with open(self.summary_path, 'w+') as csvfile: 
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['tissue_type','pixel_area','micron_area']) 

            for row in rows:  
                csvwriter.writerow(row)

        print('Sample has been reconciled. Pathology summary produced.')
        print('Wait a few seconds while the spatial data is being saved ...')

        self.ann_objects_gdf.to_file(self.shape_file_path, driver='ESRI Shapefile')
        print('Reconciliation module has finished.')


    def execute(self, **kwargs):
        self._sanity_check(**kwargs)

        img_path = kwargs['original_image']
        geojson_file = kwargs['annotation_geojson']
        marker = kwargs['marker_dropdown']
        
        self.marker = marker.split('.')[-1].strip(' ')

        img_source = large_image.getTileSource(img_path)
        img_metadata = img_source.getMetadata()

        self.base_res = img_metadata['magnification']
        self.pixel_size = img_metadata['mm_x'] * 1000 # converting from mm to microns

        self._get_tissue_types(geojson_file)

        self.overlay_tissue_types(img_source, img_metadata)

        params = {k: fixed(v) for k, v in kwargs.items()}

        # triggers the action of assigning each cell to a specific tissue type
        my_interact_manual = interact_manual.options(manual_name="Reconcile annotations")
        my_interact_manual(self.reconcile, **params)
