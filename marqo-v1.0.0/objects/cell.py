import os
import json
import numpy as np

class Cell:
    """
    Class that defines the Cell objects that populate the geojsons
    """
    __slots__ = ('type', 'geometry', 'properties', 'nucleusGeometry')

    def __init__(self, roi_index, cell_index, cyto_hull_coordinates, nuc_hull_coordinates):
        self.type = 'Feature'
        self.geometry = {'type': 'Polygon', 'coordinates': [cyto_hull_coordinates]}
        self.properties = {"object_type": "annotation", 
                   "isLocked": "false", 
                   "roi_index": roi_index, 
                   "cell_index": cell_index, 
                   "measurements": {},
                   "classification": {
                       "name": "Negative",
                       "colorRGB": -9408287
                   }}

        self.nucleusGeometry = {'type': 'Polygon', 'coordinates': [nuc_hull_coordinates]}

    def calculate_measurements(self, nuc_hema_values, cyto_hema_values, mem_hema_values, 
                               nuc_AEC_values, cyto_AEC_values, mem_AEC_values, 
                               nuc_res_values, cyto_res_values, mem_res_values):
        
        self.properties['measurements'] = {
                        'Hematoxylin: Nucleus: Mean': np.mean(nuc_hema_values) / 255,
                        'Hematoxylin: Nucleus: Median': np.median(nuc_hema_values) / 255,
                        'Hematoxylin: Nucleus: Min': np.min(nuc_hema_values) / 255,
                        'Hematoxylin: Nucleus: Max': np.max(nuc_hema_values) / 255,
                        'Hematoxylin: Nucleus: Std.Dev.': np.std(nuc_hema_values) / 255,
                        'Hematoxylin: Nucleus: Variance': np.var(nuc_hema_values) / 255,
                        'Hematoxylin: Cytoplasm: Mean': np.mean(cyto_hema_values) / 255,
                        'Hematoxylin: Cytoplasm: Median': np.median(cyto_hema_values) / 255,
                        'Hematoxylin: Cytoplasm: Min': np.min(cyto_hema_values) / 255,
                        'Hematoxylin: Cytoplasm: Max': np.max(cyto_hema_values) / 255,
                        'Hematoxylin: Cytoplasm: Std.Dev.': np.std(cyto_hema_values) / 255,
                        'Hematoxylin: Cytoplasm: Variance': np.var(cyto_hema_values) / 255,
                        'Hematoxylin: Membrane: Mean': np.mean(mem_hema_values) / 255,
                        'Hematoxylin: Membrane: Median': np.median(mem_hema_values) / 255,
                        'Hematoxylin: Membrane: Min': np.min(mem_hema_values) / 255,
                        'Hematoxylin: Membrane: Max': np.max(mem_hema_values) / 255,
                        'Hematoxylin: Membrane: Std.Dev.': np.std(mem_hema_values) / 255,
                        'Hematoxylin: Membrane: Variance': np.var(mem_hema_values) / 255,
                        'Hematoxylin: Cell: Mean': np.mean(np.concatenate([cyto_hema_values, nuc_hema_values, mem_hema_values])) / 255,
                        'Hematoxylin: Cell: Median': np.median(np.concatenate([cyto_hema_values, nuc_hema_values, mem_hema_values])) / 255,
                        'Hematoxylin: Cell: Min': np.min(np.concatenate([cyto_hema_values, nuc_hema_values, mem_hema_values])) / 255,
                        'Hematoxylin: Cell: Max': np.max(np.concatenate([cyto_hema_values, nuc_hema_values, mem_hema_values])) / 255,
                        'Hematoxylin: Cell: Std.Dev.': np.std(np.concatenate([cyto_hema_values, nuc_hema_values, mem_hema_values])) / 255,
                        'Hematoxylin: Cell: Variance': np.var(np.concatenate([cyto_hema_values, nuc_hema_values, mem_hema_values])) / 255,
                        'DAB: Nucleus: Mean': np.mean(nuc_AEC_values) / 255,
                        'DAB: Nucleus: Median': np.median(nuc_AEC_values) / 255,
                        'DAB: Nucleus: Min': np.min(nuc_AEC_values) / 255,
                        'DAB: Nucleus: Max': np.max(nuc_AEC_values) / 255,
                        'DAB: Nucleus: Std.Dev.': np.std(nuc_AEC_values) / 255,
                        'DAB: Nucleus: Variance': np.var(nuc_AEC_values) / 255,
                        'DAB: Cytoplasm: Mean': np.mean(cyto_AEC_values) / 255,
                        'DAB: Cytoplasm: Median': np.median(cyto_AEC_values) / 255,
                        'DAB: Cytoplasm: Min': np.min(cyto_AEC_values) / 255,
                        'DAB: Cytoplasm: Max': np.max(cyto_AEC_values) / 255,
                        'DAB: Cytoplasm: Std.Dev.': np.std(cyto_AEC_values) / 255,
                        'DAB: Cytoplasm: Variance': np.var(cyto_AEC_values) / 255,
                        "DAB: Membrane: Mean": np.mean(mem_AEC_values) / 255,
                        "DAB: Membrane: Median": np.median(mem_AEC_values) / 255,
                        "DAB: Membrane: Min": np.min(mem_AEC_values) / 255,
                        "DAB: Membrane: Max": np.max(mem_AEC_values) / 255,
                        "DAB: Membrane: Std.Dev.": np.std(mem_AEC_values) / 255,
                        "DAB: Membrane: Variance": np.var(mem_AEC_values) / 255,
                        'DAB: Cell: Mean': np.mean(np.concatenate([cyto_AEC_values, nuc_AEC_values, mem_AEC_values])) / 255,
                        'DAB: Cell: Median': np.median(np.concatenate([cyto_AEC_values, nuc_AEC_values, mem_AEC_values])) / 255,
                        'DAB: Cell: Min': np.min(np.concatenate([cyto_AEC_values, nuc_AEC_values, mem_AEC_values])) / 255,
                        'DAB: Cell: Max': np.max(np.concatenate([cyto_AEC_values, nuc_AEC_values, mem_AEC_values])) / 255,
                        'DAB: Cell: Std.Dev.': np.std(np.concatenate([cyto_AEC_values, nuc_AEC_values, mem_AEC_values])) / 255,
                        'DAB: Cell: Variance': np.var(np.concatenate([cyto_AEC_values, nuc_AEC_values, mem_AEC_values])) / 255,
                        "Residual: Nucleus: Mean": np.mean(nuc_res_values) / 255,
                        "Residual: Nucleus: Median": np.median(nuc_res_values) / 255,
                        "Residual: Nucleus: Min": np.min(nuc_res_values) / 255,
                        "Residual: Nucleus: Max": np.max(nuc_res_values) / 255,
                        "Residual: Nucleus: Std.Dev.": np.std(nuc_res_values) / 255,
                        "Residual: Nucleus: Variance": np.var(nuc_res_values) / 255,
                        "Residual: Cytoplasm: Mean": np.mean(cyto_res_values) / 255,
                        "Residual: Cytoplasm: Median": np.median(cyto_res_values) / 255,
                        "Residual: Cytoplasm: Min": np.min(cyto_res_values) / 255,
                        "Residual: Cytoplasm: Max": np.max(cyto_res_values) / 255,
                        "Residual: Cytoplasm: Std.Dev.": np.std(cyto_res_values) / 255,
                        "Residual: Cytoplasm: Variance": np.var(cyto_res_values) / 255,
                        "Residual: Membrane: Mean": np.mean(mem_res_values) / 255,
                        "Residual: Membrane: Median": np.median(mem_res_values) / 255,
                        "Residual: Membrane: Min": np.min(mem_res_values) / 255,
                        "Residual: Membrane: Max": np.max(mem_res_values) / 255,
                        "Residual: Membrane: Std.Dev.": np.std(mem_res_values) / 255,
                        "Residual: Membrane: Variance": np.var(mem_res_values) / 255,
                        "Residual: Cell: Mean": np.mean(np.concatenate([cyto_res_values, nuc_res_values, mem_res_values])) / 255,
                        "Residual: Cell: Median": np.median(np.concatenate([cyto_res_values, nuc_res_values, mem_res_values])) / 255,
                        "Residual: Cell: Min": np.min(np.concatenate([cyto_res_values, nuc_res_values, mem_res_values])) / 255,
                        "Residual: Cell: Max": np.max(np.concatenate([cyto_res_values, nuc_res_values, mem_res_values])) / 255,
                        "Residual: Cell: Std.Dev.": np.std(np.concatenate([cyto_res_values, nuc_res_values, mem_res_values])) / 255,
                        "Residual: Cell: Variance": np.var(np.concatenate([cyto_res_values, nuc_res_values, mem_res_values])) / 255
                } 

    def calculate_measurements_if(self, nuc_chromogen_values, cyto_chromogen_values, mem_chromogen_values, nuc_dapi_values, mem_dapi_values, cyto_dapi_values, max_pixel_value):

        self.properties['measurements'] = {
                        'DAPI: Nucleus: Mean': np.mean(nuc_dapi_values) / max_pixel_value,
                        'DAPI: Nucleus: Median': np.median(nuc_dapi_values) / max_pixel_value,
                        'DAPI: Nucleus: Min': np.min(nuc_dapi_values) / max_pixel_value,
                        'DAPI: Nucleus: Max': np.max(nuc_dapi_values) / max_pixel_value,
                        'DAPI: Nucleus: Std.Dev.': np.std(nuc_dapi_values) / max_pixel_value,
                        'DAPI: Nucleus: Variance': np.var(nuc_dapi_values) / max_pixel_value,
                        'DAPI: Cytoplasm: Mean': np.mean(cyto_dapi_values) / max_pixel_value,
                        'DAPI: Cytoplasm: Median': np.median(cyto_dapi_values) / max_pixel_value,
                        'DAPI: Cytoplasm: Min': np.min(cyto_dapi_values) / max_pixel_value,
                        'DAPI: Cytoplasm: Max': np.max(cyto_dapi_values) / max_pixel_value,
                        'DAPI: Cytoplasm: Std.Dev.': np.std(cyto_dapi_values) / max_pixel_value,
                        'DAPI: Cytoplasm: Variance': np.var(cyto_dapi_values) / max_pixel_value,
                        'DAPI: Membrane: Mean': np.mean(mem_dapi_values) / max_pixel_value,
                        'DAPI: Membrane: Median': np.median(mem_dapi_values) / max_pixel_value,
                        'DAPI: Membrane: Min': np.min(mem_dapi_values) / max_pixel_value,
                        'DAPI: Membrane: Max': np.max(mem_dapi_values) / max_pixel_value,
                        'DAPI: Membrane: Std.Dev.': np.std(mem_dapi_values) / max_pixel_value,
                        'DAPI: Membrane: Variance': np.var(mem_dapi_values) / max_pixel_value,
                        'DAPI: Cell: Mean': np.mean(np.concatenate([cyto_dapi_values, nuc_dapi_values, mem_dapi_values])) / max_pixel_value,
                        'DAPI: Cell: Median': np.median(np.concatenate([cyto_dapi_values, nuc_dapi_values, mem_dapi_values])) / max_pixel_value,
                        'DAPI: Cell: Min': np.min(np.concatenate([cyto_dapi_values, nuc_dapi_values, mem_dapi_values])) / max_pixel_value,
                        'DAPI: Cell: Max': np.max(np.concatenate([cyto_dapi_values, nuc_dapi_values, mem_dapi_values])) / max_pixel_value,
                        'DAPI: Cell: Std.Dev.': np.std(np.concatenate([cyto_dapi_values, nuc_dapi_values, mem_dapi_values])) / max_pixel_value,
                        'DAPI: Cell: Variance': np.var(np.concatenate([cyto_dapi_values, nuc_dapi_values, mem_dapi_values])) / max_pixel_value,
                        'Chromogen: Nucleus: Mean': np.mean(nuc_chromogen_values) / max_pixel_value,
                        'Chromogen: Nucleus: Median': np.median(nuc_chromogen_values) / max_pixel_value,
                        'Chromogen: Nucleus: Min': np.min(nuc_chromogen_values) / max_pixel_value,
                        'Chromogen: Nucleus: Max': np.max(nuc_chromogen_values) / max_pixel_value,
                        'Chromogen: Nucleus: Std.Dev.': np.std(nuc_chromogen_values) / max_pixel_value,
                        'Chromogen: Nucleus: Variance': np.var(nuc_chromogen_values) / max_pixel_value,
                        'Chromogen: Cytoplasm: Mean': np.mean(cyto_chromogen_values) / max_pixel_value,
                        'Chromogen: Cytoplasm: Median': np.median(cyto_chromogen_values) / max_pixel_value,
                        'Chromogen: Cytoplasm: Min': np.min(cyto_chromogen_values) / max_pixel_value,
                        'Chromogen: Cytoplasm: Max': np.max(cyto_chromogen_values) / max_pixel_value,
                        'Chromogen: Cytoplasm: Std.Dev.': np.std(cyto_chromogen_values) / max_pixel_value,
                        'Chromogen: Cytoplasm: Variance': np.var(cyto_chromogen_values) / max_pixel_value,
                        'Chromogen: Membrane: Mean': np.mean(mem_chromogen_values) / max_pixel_value,
                        'Chromogen: Membrane: Median': np.median(mem_chromogen_values) / max_pixel_value,
                        'Chromogen: Membrane: Min': np.min(mem_chromogen_values) / max_pixel_value,
                        'Chromogen: Membrane: Max': np.max(mem_chromogen_values) / max_pixel_value,
                        'Chromogen: Membrane: Std.Dev.': np.std(mem_chromogen_values) / max_pixel_value,
                        'Chromogen: Membrane: Variance': np.var(mem_chromogen_values) / max_pixel_value,
                        'Chromogen: Cell: Mean': np.mean(np.concatenate([cyto_chromogen_values, nuc_chromogen_values, mem_chromogen_values])) / max_pixel_value,
                        'Chromogen: Cell: Median': np.median(np.concatenate([cyto_chromogen_values, nuc_chromogen_values, mem_chromogen_values])) / max_pixel_value,
                        'Chromogen: Cell: Min': np.min(np.concatenate([cyto_chromogen_values, nuc_chromogen_values, mem_chromogen_values])) / max_pixel_value,
                        'Chromogen: Cell: Max': np.max(np.concatenate([cyto_chromogen_values, nuc_chromogen_values, mem_chromogen_values])) / max_pixel_value,
                        'Chromogen: Cell: Std.Dev.': np.std(np.concatenate([cyto_chromogen_values, nuc_chromogen_values, mem_chromogen_values])) / max_pixel_value,
                        'Chromogen: Cell: Variance': np.var(np.concatenate([cyto_chromogen_values, nuc_chromogen_values, mem_chromogen_values])) / max_pixel_value,
                }



    def jsonfy(self):
        return {slot: getattr(self, slot) for slot in self.__slots__}



'''
def timeit(method):
    def timed(*args, **kw):
        global RESULT
        s = datetime.now()
        RESULT = method(*args, **kw)
        e = datetime.now()

        sizeMb = process.memory_info().rss / 1024 / 1024
        sizeMbStr = "{0:,}".format(round(sizeMb, 2))

        print('Time Taken = %s, \t%s, \tSize = %s' % (e - s, method.__name__, sizeMbStr))

    return timed
'''

