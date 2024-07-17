
import os
from PIL import Image
import numpy as np
import pandas as pd
import json
import yaml

from scipy import spatial

import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
from typing import List

from utils.image import ImageHandler
from utils.utils import Utils

class WaxmanGraph:
    '''
    Class to define graph object
    '''

    def __init__(self):
        self.adjancy_matrix = ''
        self.alpha = 1
        self.beta = 0.1


    def get_edge_probabilities(self, dist_matrix, L):
        P = self.alpha * np.exp((-dist_matrix) / (self.beta * L))
        return P


class Spatial:
    """
    Class that defines statich methods to deal with spatial analysis (coordinates, polygons, points, set operations)
    """

    def __init__(self):
        pass


    @staticmethod
    def get_hull(x_coords: list, y_coords: list, segmentation_buffer: int):
        """
        Hull coodinates

        Inputs:
            ->

        Output:
            <-
        """
        xy = np.array((x_coords, y_coords))
        xy = xy.T

        if len(xy) >= 3:
            hull = spatial.ConvexHull(xy, incremental=False, qhull_options='Qt')
            hull_indices = hull.vertices

        else:
            return xy.tolist()

        hull_coordinates = []

        for i in range(len(hull_indices)):
            index = hull_indices[i]
            x = xy[index, 0].astype('float64') - segmentation_buffer 
            # simplify float number to reduce space on disk
            #x = np.round(x)

            y = xy[index, 1].astype('float64') - segmentation_buffer
            #y = np.round(y)

            hull_coordinates.append([x, y])

        hull_coordinates.append([hull_coordinates[0][0], hull_coordinates[0][1]])  # close the polygon -- first and last points must be equals

        return hull_coordinates

    @staticmethod
    def get_core_label(mother_coordinates, config_file):
        with open(config_file, 'r+') as stream:
            config_data = yaml.safe_load(stream)
            marco_output_path = config_data['directories']['marco_output']
            sample_name = config_data['sample']['name']
            tile_dimension = config_data['parameters']['image']['tile_dimension']
            
            cores_map_file = os.path.join(marco_output_path, f'{sample_name}_cores_map.json')

        with open(cores_map_file, 'r') as stream:
            cores_map = json.load(stream)

        tile_centroid = Point((mother_coordinates[1] + tile_dimension / 2), (mother_coordinates[0] + tile_dimension / 2))
        for label, rects in cores_map.items():
            for rect in rects:
                y, x, width, height = rect
                corners = [
                    (x, y),                    # Top-left
                    (x + width, y),            # Top-right
                    (x + width, y + height),   # Bottom-right
                    (x, y + height)            # Bottom-left
                ]
                
                # Create a Polygon
                rect_polygon = Polygon(corners)
                
                if tile_centroid.within(rect_polygon):
                    return label    

        return 'Undefined'
        
    
    @staticmethod
    def tma_core_assignment(data_for_tile, tma_map_files, tma_map_marker, output_resolution):
        # For each marker
        #    Create polygons for each core (Poly1 -> core A1, Poly2 -> core A2 ...)
        #    Iterate over the cells for specific marker and check if point is within polygon (for each marker is there is the need to check the base resolution of the image the map was built)
        #    Assign to new column the TMA core ID
        # Return the assigned data

        # Grab the TMA maps from config file -- same order as the markers, using Utils
        tma_cores, base_res = Utils.get_tma_cores(tma_map_files) 
	
        rawcellmetrics_fortile_df = data_for_tile[0]
        
        tma_column = []
        for idx, row in rawcellmetrics_fortile_df.iterrows():
            y_coord = row[f'{tma_map_marker} y_coor (pixels)'] * (base_res / output_resolution)
            x_coord = row[f'{tma_map_marker} x_coor (pixels)'] * (base_res / output_resolution)
            
            #y_coord = (mother_coordinates[0] + row[f'{marker_name} y_coor (pixels)']) * (base_res / output_resolution)
            #x_coord = (mother_coordinates[1] + row[f'{marker_name} x_coor (pixels)']) * (base_res / output_resolution)

            centroid = Point(x_coord, y_coord)
            for core in tma_cores:
                poly = core[0]
                name = core[1]
               
                found = False
                if centroid.within(poly):
                    tma_column.append(name)
                    found = True
                    break
            
            if not found:
                tma_column.append('Unknown')

        rawcellmetrics_fortile_df['TMA Core'] = tma_column
        return rawcellmetrics_fortile_df, data_for_tile[1], data_for_tile[2]
    

    def add_nodes(self, graph, coords):
        for node, coo in enumerate(coords, start=0):
            graph.add_node(node, pos=coo)

        return graph


    def generate_graph_for_specific_marker(self, finalcellinfo_data, marker):
        # Subsetting the marker of interest
        finalcellinfo_data = finalcellinfo_data[['Tile index', 'Cell index', 'Relative x_coord', 'Relative y_coord', f'{marker}_expression']]

        # Filtering
        finalcellinfo_data = finalcellinfo_data[finalcellinfo_data['Tile index'] == 11.0]
        finalcellinfo_data = finalcellinfo_data[finalcellinfo_data[f'{marker}_expression'] == True]

        # my_nodes = finalcellinfo_data['Cell index'].tolist()
        X = list(zip(finalcellinfo_data['Relative x_coord'], finalcellinfo_data['Relative y_coord']))

        dist_matrix = euclidean_distances(X, X)
        # print(dist_matrix)
        L = np.amax(dist_matrix)
        # print(f'Max dist: {L}')

        waxman = WaxmanGraph()
        out = waxman.get_edge_probabilities(dist_matrix, L)
        np.fill_diagonal(out, np.nan)

        # mask = (out > 0.2) & (out < 1.0)
        rows, cols = np.where(out > 0.9)
        edges = zip(rows.tolist(), cols.tolist())


        G = nx.Graph()
        G = self.add_nodes(graph=G, coords=X)
        pos = nx.get_node_attributes(G, 'pos')
        G.add_edges_from(edges)

        return G, pos


    def build_waxman_graph(self):
        '''
        TODO: when stitching ROIs, create a finalcellinfo with absolute coodinates and use it to generate the graph
        It reads the finalcellinfo.csv (to grab the data about expression 'positive/negative');
        Using the cell index it matches with the geojson (with absoluto coordinates for the ROI - or stitched ROIs)
        With the group of cells of interest (usually positive ones) we calcualte the Euclidean distance for every pair
        With the distance we assign a probability (edge) between the cells (nodes)
        Store that information in the adjancy matrix.
        The adjancy matrix can be one of the followings configurations:
            i.  N x N: square matrix for a single marker. 
            ii. M x N: graph for two markers.
        '''
        img = Image.open('/Users/igorsouza/Workdir/MARCO/2023/v1_branch/tmp/RGBimages_perROI/sample_10_2_ROI#11_CD3.ome.tif')

        finalcellinfo_file = '/Users/igorsouza/Workdir/MARCO/2023/v1_branch/tmp/sample_10_2_finalcellinfo.csv'
        finalcellinfo_data = pd.read_csv(finalcellinfo_file)
        
        G, pos = self.generate_graph_for_specific_marker(finalcellinfo_data, marker='CD3')
        H, pos2 = self.generate_graph_for_specific_marker(finalcellinfo_data, marker='CD8')

        plt.imshow(img)

        nx.draw(G, node_size=5, pos=pos, node_color='r', with_labels=False)
        nx.draw(H, node_size=5, pos=pos2, node_color='g', with_labels=False)
        plt.show()
        return
    

    def get_absolute_coords(self, cells_of_interest, tilestats):
        """
        Function to change the coords form relative (x,y) to absolute
        """
        def _shift(row):
            tile_index = row['Tile index']
            abs_x = row['Relative x_coord'] + tilestats[tile_index][1]
            abs_y = row['Relative y_coord'] + tilestats[tile_index][0]
            abs_xy = (abs_x, abs_y)
            
            return abs_xy


        cells_of_interest['Abs(x,y)'] = cells_of_interest.apply(_shift, axis=1)
            
        return cells_of_interest


    def get_cells_of_interest(self, finalcellinfo, markers, tilestats):
        """
        Function to get the cell indexes with the desired coexpression pattern 

        Inputs:

        Outputs:
        """

        marker_columns = [f'{marker}_expression' for marker in markers]
        marker_columns.extend(['Tile index', 'Relative y_coord', 'Relative x_coord'])
        finalcellinfo = finalcellinfo[marker_columns]

        cells_indices = finalcellinfo.all(axis='columns')
        cells_of_interest = finalcellinfo[cells_indices]

        cells_of_interest = self.get_absolute_coords(cells_of_interest, tilestats)

        return cells_of_interest


    def _convert_str_to_int(self, tilestats):
        for key in tilestats:
            value_as_string = tilestats[key]
            value_as_string = value_as_string.replace('[','')
            value_as_string = value_as_string.replace(']','')
            ROI_ycoord = int(str.split(value_as_string,',')[0])
            ROI_xcoord = int(str.split(value_as_string,',')[1])
            tilestats[key] = [ROI_ycoord, ROI_xcoord]

        return tilestats


    def plot_coexp(self, markers, tiles, finalcellinfo_file, tilestats_file):
        """
        Function to visualize the coexpression. As input we need the list of the images of interest (different markers),
        the tiles (used to build the image of interest - stitched image), the tilestats.csv and the markers of interest.
        
        Inputs:
            ->
        
        Outputs:
            <-

        """
        
        # Get the coordinates information for every tile of interest
        tilestats_df = pd.read_csv(tilestats_file, sep=',', header=0, index_col=[0])
        tilestats = dict(tilestats_df.loc[tiles, 'Tile top left pixel coordinates (Y,X)'])
        tilestats = self._convert_str_to_int(tilestats)

        # Open finalcellinfo.csv as dataframe. Indexing the cell index column
        finalcellinfo = pd.read_csv(finalcellinfo_file, sep=',', header=0, index_col=[1])

        cells_of_interest = self.get_cells_of_interest(finalcellinfo, markers, tilestats)


