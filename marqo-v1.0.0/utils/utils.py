import os
import yaml
import json
import numpy as np
import large_image
from shapely.geometry import Polygon


from typing import List
from utils.image import ImageHandler

class Utils:
    """
    Util Class define a number of static methods to handle the config file (staging and updating), create and check existence of directories,
    iterate over those directories, and (TODO) to call the logger
    """

    @staticmethod
    def get_scale_factor(config_file: str):

        with open(config_file) as stream:
            config_data = yaml.safe_load(stream)

            try:
                output_resolution = config_data['parameters']['image']['output_resolution']
                image_type = config_data['parameters']['image']['image_type']

            except:
                output_resolution = config_data['parameters']['output_resolution']
                image_type = config_data['parameters']['image_type']

            #convert pixels to microns
            svs_20x = 0.5031 
            svs_40x = 0.2521
            ndpi_20x = 0.4414
            ndpi_40x = 0.2207

            if image_type=='svs' and output_resolution == 20:
                SF = svs_20x

            if image_type=='svs' and output_resolution == 40 :
                SF = svs_40x

            if image_type == 'ndpi' and output_resolution == 20:
                SF = ndpi_20x

            if image_type == 'ndpi' and output_resolution == 40:
                SF = ndpi_40x

            config_data['parameters']['image']['scale_factor'] = SF

        with open(config_file, 'w+') as fout:
            yaml.dump(config_data, fout)

    @staticmethod
    def remove_special_characters(list_of_names: List[str]):

        formated_names = []
        for name in list_of_names:
            name = name.replace('.', '')
            name = name.replace("'", "")
            name = name.replace('"', '')
            name = name.replace(' ', '')
            name = name.replace('-', '')

            formated_names.append(name)

        return formated_names


    @staticmethod
    def get_slides_full_path(images_names: List[str], config_file: str):

        with open(config_file, 'r+') as stream:
            config_data = yaml.safe_load(stream)

        raw_images_dir = config_data['directories']['raw_images']

        raw_images_full_path = []
        for image_name in images_names:
            full_path = os.path.join(raw_images_dir, image_name)
            raw_images_full_path.append(full_path)

        config_data['sample']['raw_images_full_path'] = raw_images_full_path

        with open(config_file, 'w+') as fout:
            yaml.dump(config_data, fout)


    @staticmethod
    def parse_tma_map(tma_core_data):
        tma_cores = []
        for core in tma_core_data['features']:
            if core['properties']['isMissing'] is not True:
                name = core['properties']['name']
                coordinates = core['geometry']['coordinates'][0]

                tma_cores.append((Polygon(coordinates), name))

        return tma_cores


    @staticmethod
    def get_tma_cores(tma_map_files):
        """
        
        out:
            <- tuple(coordinates: List, base resolution: int)
        """
        image_path = tma_map_files[0]
        tma_map = tma_map_files[1]
        base_res = ImageHandler.get_base_res(image_path)

        with open(tma_map, 'r') as stream:
            tma_core_data = json.load(stream)

        tma_cores = Utils.parse_tma_map(tma_core_data)

        return tma_cores, base_res

    @staticmethod
    def check_cores_annotation(column_names, config_file):
        if 'Core Label' in column_names:
            with open(config_file, 'r+') as stream:
                config_data = yaml.safe_load(stream)
                config_data['sample']['cores_labels'] = True

            with open(config_file, 'w+') as fout:
                yaml.dump(config_data, fout)


    @staticmethod
    def exist_cores(config_file):

        with open(config_file, 'r+') as stream:
            config_data = yaml.safe_load(stream)
            try:
                marco_output_path = config_data['directories']['marco_output']
                sample_name = config_data['sample']['name']
            except:
                marco_output_path = config_data['directories']['sample_directory']
                sample_name = config_data['sample']['sample_name']

           
        cores_map_file = os.path.join(marco_output_path, f'{sample_name}_cores_map.json')
        if os.path.exists(cores_map_file):
            return True

        else:
            return False


    @staticmethod
    def get_tma_core_map_info(tma_map_marker, images_names, config_file):

        with open(config_file, 'r+') as stream:
            config_data = yaml.safe_load(stream)

            raw_images_dir = config_data['directories']['raw_images']
            raw_images_full_path = config_data['sample']['raw_images_full_path']
            markers_name_list = config_data['sample']['markers']
           
        root, _, fs = next(os.walk(os.path.join(raw_images_dir, 'tma_map')))

        tma_image = raw_images_full_path[markers_name_list.index(tma_map_marker)]
        tma_map = os.path.join(root, fs[0])
        tma_cores, _ = Utils.get_tma_cores([tma_image, tma_map])
        
        config_data['sample']['tma_map_marker'] = tma_map_marker
        config_data['sample']['tma_map_files'] = [tma_image, tma_map]
        config_data['sample']['tma_cores'] = [t[1] for t in tma_cores]  # grab only the names of the cores

        with open(config_file, 'w+') as fout:
            yaml.dump(config_data, fout)



    @staticmethod
    def get_metadata_from_if(images_names: List[str], raw_images_path):
        if_image = images_names[0]
        full_path = os.path.join(raw_images_path, if_image)

        source = large_image.open(full_path)
        metadata = source.getMetadata()
        width = metadata['sizeX']
        height = metadata['sizeY']

        dimensions = {'width' : width, 'height': height}

        return dimensions

    
    @staticmethod
    def get_pixelsize(images_names: List[str], raw_images_path, config_file):
        with open(config_file) as stream:
            config_data = yaml.safe_load(stream)

        # TODO: i'm assuming all images were taken using same resolution, it is not ideal.
        if_image = images_names[0]
        full_path = os.path.join(raw_images_path, if_image)


        source = large_image.open(full_path)
        metadata = source.getMetadata()

        SF = metadata['mm_x'] * 1000  # converting mm to um
        config_data['parameters']['image']['scale_factor'] = SF

        with open(config_file, 'w+') as fout:
            yaml.dump(config_data, fout)



    @staticmethod
    def add_dimensions_as_parameters(dimensions, config_file):
        with open(config_file, 'r') as stream:
            config_data = yaml.safe_load(stream)

        height = dimensions['height']
        width = dimensions['width']

        config_data['parameters']['image']['height'] = height
        config_data['parameters']['image']['width'] = width

        with open(config_file, 'w+') as fout:
            yaml.dump(config_data, fout)

    @staticmethod
    def add_dna_marker(marker_name, marker_name_list, config_file):
        with open(config_file, 'r') as stream:
            config_data = yaml.safe_load(stream)

        config_data['sample']['dna_marker_channel'] = marker_name_list.index(marker_name)

        with open(config_file, 'w+') as fout:
            yaml.dump(config_data, fout)

    @staticmethod
    def add_ref_channels(af_channels, marker_name_list, config_file):
        with open(config_file, 'r') as stream:
            config_data = yaml.safe_load(stream)

        af_indices = [i for i, marker in enumerate(marker_name_list) if marker in af_channels]

        config_data['sample']['af_channels'] = af_indices

        with open(config_file, 'w+') as fout:
            yaml.dump(config_data, fout)

    @staticmethod
    def stage_directories(config_file: str):
        
        with open(config_file, 'r+') as stream:
            config_data = yaml.safe_load(stream)   
        
            marco_output_path = config_data['directories']['marco_output']
            
            multitiff_directory = os.path.join(marco_output_path, 'multitiffs_perROI')
            config_data['directories']['tiff'] = multitiff_directory

            RGBimages_directory = os.path.join(marco_output_path, 'RGBimages_perROI')
            config_data['directories']['rgb'] = RGBimages_directory

            geoJSONs_directory = os.path.join(marco_output_path, 'geoJSONS_perROI')
            config_data['directories']['geojson'] = geoJSONs_directory

            if not os.path.exists(multitiff_directory):
                os.mkdir(multitiff_directory)

            if not os.path.exists(RGBimages_directory):
                os.mkdir(RGBimages_directory)

            if not os.path.exists(geoJSONs_directory):
                os.mkdir(geoJSONs_directory)

            # Check if stardist2D model exist in the pipeline directory tree
            exists, base_dir = Utils._segmentation_model_folder()
            if exists:
                config_data['directories']['stardist_models'] = base_dir

            else:
                print('Stardist model directory do not exist within segmentation/ folder. Please check it.')

        with open(config_file, 'w+') as fout:
            yaml.dump(config_data, fout)

    @staticmethod
    def _segmentation_model_folder():
        utils_dir = os.path.dirname(os.path.realpath(__file__))
        root = '/'.join(utils_dir.split('/')[:-1])

        model_dir = os.path.join(root, 'segmentation', 'models')
        if os.path.exists(model_dir):
            base_dir = os.path.join(root, 'segmentation')
            return True, base_dir

        else:
            return False, False
        

    @staticmethod
    def stage_config_file(sample_name: str, raw_images_path: str, output_path: str, **kwargs):
        
        marco_output_path = os.path.join(output_path, sample_name)

        if not os.path.exists(marco_output_path):
            os.mkdir(marco_output_path)


        with open('../pipelines/parameters.yaml') as stream:
            parameters = yaml.safe_load(stream)

        config_data = {
            'parameters': {
                'image': {
                    'tile_dimension': parameters['IMAGE']['TILE_DIMENSION'],
                    'visual_per_tile': parameters['IMAGE']['VISUAL_PER_TILE'],
                    'mag_to_analyze_for_matrix': parameters['IMAGE']['MAG_TO_ANALYZE_FOR_MATRIX'],
                    'output_resolution': kwargs['output_resolution'],  # user defined values
                    'image_type': 'svs',  # TODO: new widget on notebook so user can define it
                    'scale_factor': 0  # user defined values
                },
                'registration': {
                    'registration_buffer': parameters['REGISTRATION']['BUFFER'],
                    'iterations': parameters['REGISTRATION']['ITERATIONS'],
                    'resolutions': parameters['REGISTRATION']['RESOLUTIONS'],
                    'spatial_samples': parameters['REGISTRATION']['SPATIAL_SAMPLES']
                },
                'segmentation': {
                    'segmentation_buffer': parameters['SEGMENTATION']['SEGMENTATION_BUFFER'],
                    'noise_threshold': parameters['SEGMENTATION']['NOISE_THRESHOLD'],
                    'perc_consensus': parameters['SEGMENTATION']['PERC_CONSENSUS'],
                    'cyto_distances': kwargs['cyto_distances'] if 'cyto_distances' in kwargs else ''  # user defined values
                },
                'quantification': {
                    'num_pixels_thresh_for_peak': parameters['QUANTIFICATION']['NUM_PIXELS_THRES_FOR_PEAK'],
                    'hema_thresh_deemedRBC': parameters['QUANTIFICATION']['HEME_THRESH_DEEMEDRBC'],

                },
                'classification': {
                    'bw': parameters['CLASSIFICATION']['BW']
                },
                'masking': {},
                'tiling': {}
            },
            'directories': {
                'raw_images': raw_images_path,
                'marco_output': marco_output_path,
                'rgb': '',
                'tiff': '',
                'geojson': '',
                'logs': '',
                'stardist_models': '',
            },
            'sample': {
                'technology': kwargs['technology'],
                'cores_labels': False,
                'name': sample_name,
                'markers': kwargs['marker_name_list'],
                'raw_images_full_path': []
            }
        }
        
        if kwargs['segmentation_model']['model'] == 'stardist':
            config_data['parameters']['segmentation']['model'] = 'stardist'
            config_data['parameters']['segmentation']['prob'] = kwargs['segmentation_model']['stardist_prob']
            config_data['parameters']['segmentation']['nms'] = kwargs['segmentation_model']['stardist_nms']
            config_data['parameters']['segmentation']['pixel_size'] = kwargs['segmentation_model']['stardist_pixel_size']

        elif kwargs['segmentation_model']['model'] == 'cellpose':
            config_data['parameters']['segmentation']['model'] = 'cellpose'
            config_data['parameters']['segmentation']['flow'] = kwargs['segmentation_model']['cellpose_flow']
            config_data['parameters']['segmentation']['cell_diameter'] = kwargs['segmentation_model']['cellpose_diameter']

        config_file = os.path.join(marco_output_path, 'config.yaml')

        with open(config_file, 'w+') as fout:
            yaml.dump(config_data, fout)

        return config_file

    @staticmethod
    def add_cycles_metadata(cycles_metadata, config_file):
        with open(config_file) as stream:
            config_data = yaml.safe_load(stream)

        
        config_data['cycles_metadata'] = cycles_metadata

        with open(config_file, 'w+') as fout:
            yaml.dump(config_data, fout)


    @staticmethod
    def add_parameters(config_file, step_name, index, **kwargs):
        """
        Function to add parameters to the config file
        """
        with open(config_file, 'r') as stream:
            config_data = yaml.safe_load(stream)

        if 'cycles_metadata' in config_data:
            marker = f'cycle_{index}'

        else:
            marker = config_data['sample']['markers'][index]       

        if step_name == 'masking':
            config_data['parameters'][step_name][marker] = {}

            for name, value in kwargs.items():
                config_data['parameters'][step_name][marker][name] = value

        elif step_name == 'tiling':
            config_data['parameters'][step_name] = {}

            for name, value in kwargs.items():
                config_data['parameters'][step_name][name] = value


        with open(config_file, 'w+') as fout:
            yaml.dump(config_data, fout)


    @staticmethod
    def fetch_parameters(config_file):
        with open(config_file, 'r') as stream:
            config_data = yaml.safe_load(stream)

        return config_data
    
    @staticmethod
    def fetch_one_parameter(name, config_file):
        with open(config_file, 'r') as stream:
            config_data = yaml.safe_load(stream)

        for k, v in config_data['parameters'].items():
            for parameter, value in v.items():
                if parameter == name:
                    return value


    @staticmethod
    def get_decon_matrix(config_file):
        with open(config_file, 'r') as stream:
            config_data = yaml.safe_load(stream)
        
        try:
            sample_name = config_data['sample']['name']
            marker_name_list = config_data['sample']['markers']
            marco_output_path = config_data['directories']['marco_output']        

        except:
            sample_name = config_data['sample']['sample_name']
            marker_name_list = config_data['sample']['markers']
            marco_output_path = config_data['directories']['sample_directory']

        root, _, files = next(os.walk(marco_output_path))

        decon_matrices_dict = []
        for _, marker_name in enumerate(marker_name_list):
            matrix_name = f'{sample_name}_{marker_name}.npy'
            matrix_path = os.path.join(root, matrix_name)
            matrix = np.load(matrix_path)
            decon_matrices_dict.append(matrix)

        return decon_matrices_dict
