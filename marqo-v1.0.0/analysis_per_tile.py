#!/usr/bin/env python
import os
import sys
import yaml
import pandas as pd

from pipelines.micsss import MICSSS
from pipelines.singleplex import SinglePlex
from pipelines.fluorescence import IF
from pipelines.tma import TMA
from pipelines.cyclic_immunofluorescence import CyCIF

from utils.image import ImageHandler
from utils.file import FileHandler
from utils.utils import Utils

import multiprocessing as mp
import tifffile
from time import time, sleep
from tifffile import TiffFile, TiffWriter, TiffPage
from tqdm.notebook import tqdm
import ipywidgets as widgets
from IPython.display import display
import queue
import threading
import random

import ipywidgets as widgets
from IPython.display import display

class Pipeline():
    """
    The Pipeline class controls the flow and orchestrate the data between the different modules of the pipeline.
    """
    def __init__(self, sample_directory: str, technology: str):
        self.sample_directory = sample_directory
        self.config_file = os.path.join(sample_directory, 'config.yaml')

        self.technology = technology
        self.instance = self.grab_instance(technology)
        
        if not os.path.exists(self.config_file):
            exit()

        self.get_parameters()


    def get_parameters(self):
        with open(self.config_file, 'r') as stream:
            self.config_data = yaml.safe_load(stream)

        try:
            self.marker_name_list = self.config_data['sample']['markers']
            self.sample_name = self.config_data['sample']['name']
            self.sample_directory = self.config_data['directories']['marco_output']
            self.segmentation_model = self.config_data['parameters']['segmentation']['model']

        except:
            self.marker_name_list = self.config_data['sample']['markers']
            self.sample_name = self.config_data['sample']['sample_name']
            self.sample_directory = self.config_data['directories']['sample_directory']
            self.segmentation_model = self.config_data['parameters']['segmentation']['model']

        self.registered_masks = ImageHandler.get_registered_masks(self.marker_name_list, self.sample_name, self.sample_directory, self.technology)

        tilestats_file_path = os.path.join(self.sample_directory, f'{self.sample_name}_tilemap.json')
        self.allsamples_tilestats = FileHandler.open_json(tilestats_file_path)


    def cellmetrics_reconciliation(self, n_tiles):
        tmp_files_path = os.path.join(self.sample_directory, '.tmp')

        final_cellmetrics_path = os.path.join(self.sample_directory, f'{self.sample_name}_rawcellmetrics.csv')
        final_cellmetrics = open(final_cellmetrics_path, 'w+')

        is_first = True
        for i in range(n_tiles):
            roi_index = i
            _file = os.path.join(tmp_files_path, f'{i}_rawcellmetrics.csv')

            if os.path.exists(_file):
                df_tmp = pd.read_csv(_file)

                if is_first:
                    df_tmp.to_csv(final_cellmetrics, index=False)
                    os.remove(_file)
                    is_first = False

                else:
                    df_tmp.to_csv(final_cellmetrics, mode='a', header=False, index=False)
                    os.remove(_file)

            else:
                print(f'ROI {i} didnt produce rawcellmetrics')

        # Close file
        final_cellmetrics.close()
            

    def tilestats_reconciliation(self, n_tiles):
        '''
        Reconcile the .csv files for each tile
        '''
        tmp_files_path = os.path.join(self.sample_directory, '.tmp')
        
        final_tilestats_path = os.path.join(self.sample_directory, f'{self.sample_name}_tilestats.csv') 
        final_tilestats = open(final_tilestats_path, 'w+') 
        
        csv_list = []
        for i in range(n_tiles):
            _file = os.path.join(tmp_files_path, f'{i}_tilestats.csv')

            if os.path.exists(_file):
                csv_list.append(pd.read_csv(_file))

                os.remove(_file)

            else:
                print(f'No tile stats for ROI {i}')

        csv_merged = pd.concat(csv_list, ignore_index=True)
        csv_merged.to_csv(final_tilestats, mode='w', index=False)
            
        final_tilestats.close()

        Utils.check_cores_annotation(csv_merged.columns, self.config_file)


    def run(self, parameters_array):
        try:
            roi_index = parameters_array[0]
            # Simulate processing with sleep and a random delay
            #sleep(random.randint(1, 5))
            self.instance.analysis(parameters_array)

            # Return a message indicating completion
            return f"Processed ROI {roi_index}"

        except Exception as e:
            return f"Error processing ROI {roi_index}: {e}"


    def grab_instance(self, technology):

        instances_dict = {
            'micsss': MICSSS(self.config_file),
            'singleplex': SinglePlex(self.config_file),
            'if': IF(self.config_file),
            'tma': TMA(self.config_file),
            'cycif': CyCIF(self.config_file)
        }

        return instances_dict[technology]


    def __getattr__(self, name):
        return self.instance.__getattribute__(name)
