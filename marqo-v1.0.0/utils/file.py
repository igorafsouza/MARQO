
import os
import yaml
import json
import pandas as pd
import functools
import operator
import traceback

class FileHandler:
    """
    Class that defined a bunch of methods to deal with writing and reading text files (.csv, .jsons, .geojsons, tmp files ...)
    """


    def __init__(self):
        pass


    def set_parameters(self, config_file: str):
        with open(config_file) as stream:
            config_data = yaml.safe_load(stream)


        try:
            self.sample_name = config_data['sample']['name']
            self.marker_name_list = config_data['sample']['markers']

            self.marco_output_path = config_data['directories']['marco_output']
            self.geojson_path = config_data['directories']['geojson'].replace('geojsons_perROI', 'geoJSONS_perROI')


            self.tile_dimension = config_data['parameters']['image']['tile_dimension']
            self.technology = config_data['sample']['technology']

            if self.technology == 'if':
                self.dna_marker = config_data['sample']['dna_marker_channel']

            elif self.technology == 'cyif':
                self.cycles_metadata = config_data['cycles_metadata']
                marker_list_across_channels = [v['marker_name_list'] for k, v in self.cycles_metadata.items()]
                marker_list_across_channels = functools.reduce(operator.iconcat, marker_list_across_channels, [])  # flatten the list of lists
                cycles = [f'cycle_{i}' for i, v in self.cycles_metadata.items() for _ in range(len(v['marker_name_list']))]
                self.marker_name_list = [f'{cycle}_{marker}' for marker, cycle in zip(marker_list_across_channels, cycles)]


        except:
            self.sample_name = config_data['sample']['sample_name']
            self.marker_name_list = config_data['sample']['markers']

            self.marco_output_path = config_data['directories']['sample_directory']
            self.geojson_path = config_data['directories']['product_directories']['geojson'].replace('geojsons_perROI', 'geoJSONS_perROI')

            self.tile_dimension = config_data['parameters']['tile_dimension']
            self.technology = 'micsss'


    @staticmethod
    def open_json(file_path):
        """
        Args:
            file_path: json file path

        returns:
            dictionary containing the data from the jsonå
        """
        with open(file_path, 'r') as stream:
            json_data = json.load(stream) 

        return json_data

    def _stage_tmp_dir(self):
        self.tmp_directory = os.path.join(self.marco_output_path, '.tmp')
        if not os.path.exists(self.tmp_directory):
            os.mkdir(self.tmp_directory)


    def write_marco_output(self, roi_index, rawcellmetrics_fortile_df, ROI_pixel_metrics, mother_coordinates, core_label=None):
        self._stage_tmp_dir()

        #rawcellmetrics_fortile_df, _, ROI_pixel_metrics = data_for_tile
        if not rawcellmetrics_fortile_df.empty:
            # Raw cell metrics output (cell resolution)
            cellres_sample_outputpath = os.path.join(self.tmp_directory, f'{roi_index}_rawcellmetrics.csv')
            rawcellmetrics_fortile_df.to_csv(cellres_sample_outputpath, index=False)

            # Tile data ouput (tile resolution)
            tileres_sample_outputpath = os.path.join(self.tmp_directory, f'{roi_index}_tilestats.csv')
            tilestats = {'Tile index': roi_index,
                '% tissue coverage': ROI_pixel_metrics[0][0],
                'Tissue area (microns2)': ROI_pixel_metrics[0][1],
                'Tile top left pixel coordinates (Y,X)': mother_coordinates,
                'Tile height by width (pixels)': [self.tile_dimension, self.tile_dimension]
            }

            if core_label:
                tilestats['Core Label'] = core_label

            tilestats_df = pd.DataFrame([tilestats])
            tilestats_df.to_csv(tileres_sample_outputpath, index=False)

        else:
            # TODO: improve track, for now its gonna be only a print statement
            print(f'[X] Raw cell metrics empty for ROI #{roi_index}, no .csv produced.')


    def write_geojsons(self, roi_index, allmarkers_geojson_list):

        for i, geojson in enumerate(allmarkers_geojson_list):
            if self.technology == 'if':
                i = self.dna_marker

            marker_name = self.marker_name_list[i]
            geojson_filename = f'{self.sample_name}_ROI#{roi_index}_{marker_name}.json'
            save_path = os.path.join(self.geojson_path, geojson_filename)

            with open(save_path, 'w+') as fout:
                json.dump([cell.jsonfy() if cell.__class__.__name__ == 'Cell' else cell for cell in geojson], fout)



    def consolidate_files(self):
        pass


    def delete_tmp(self):
        pass


    
