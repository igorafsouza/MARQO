import warnings
warnings.filterwarnings('ignore')

import os
import yaml
import shutil
import matplotlib
from IPython.display import display, clear_output, Markdown
from ipywidgets import Button, HBox, VBox, interact, interactive, fixed, interact_manual, Image, Layout, interactive_output, Output, HTML

from review_modules.modulewidgets import module, sample_path, user_id 
from review_modules.classify_cells import Classify
from review_modules.reconcile_annotations import Reconcile
from review_modules.visualize_populations import VisualizePopulations
from review_modules.stitch_rois import RGBStitcher
from review_modules.multidimensional_images import MultTiffStitcher

class Notebook:
    pass

    def _grab_module(self, module):
        # equivalent to a match/case statement
        self.module = {'Classify cell types': Classify(),
                       'Reconcile pathology tissue annotations': Reconcile(),
                       'Visualize cell populations': VisualizePopulations(),
                       'Create larger RGB tiles': RGBStitcher(),
                       'Create larger multi-dimensional images': MultTiffStitcher()}[module]

    def _copy(self, sample_path, user_directory):       
        sample = sample_path.strip('/').split('/')[-1]
        
        src = os.path.join(sample_path, 'config.yaml')
        target = os.path.join(user_directory, 'config.yaml')
        shutil.copyfile(src, target)

        src = os.path.join(sample_path, f'{sample}_tilestats.csv')
        target = os.path.join(user_directory, f'{sample}_tilestats.csv')
        shutil.copyfile(src, target)
        
        src = os.path.join(sample_path, f'{sample}_finalcellinfo.csv')        
        target = os.path.join(user_directory, f'{sample}_finalcellinfo.csv')
        shutil.copyfile(src, target)
                        
        path_summary = os.path.join(sample_path, f'{sample}_pathologysummary.csv')
        if os.path.exists(path_summary):
            src = path_summary
            target = os.path.join(user_directory, f'{sample}_pathologysummary.csv')
            shutil.copyfile(src, target)
            
        
    def _sanity_check(self, sample_path, user_id):
        '''
        Function to check wether the sample directory and user directory exist
        '''
        sample = sample_path.strip('/').split('/')[-1]
        status = True
        
        if not os.path.exists(sample_path):
            print('Kindly ensure path to sample directory is correct.')
            status = False
        
        if user_id == '':
            print('Kindly provide a Reviwer name.')
            status = False
            
        finalcellinfo_file = os.path.join(sample_path, f'{sample}_finalcellinfo.csv')
        if not os.path.exists(finalcellinfo_file):
            print('No *_finalcellinfo file found. Please make sure to run the clustering step.')
            status = False
            
        tilestats_file = os.path.join(sample_path, f'{sample}_tilestats.csv')
        if not os.path.exists(tilestats_file):
            print('No *_tilestats file found.')
            status = False
        
        user_directory = os.path.join(sample_path, user_id)
        if not os.path.exists(user_directory) and status:
            os.mkdir(user_directory)
            self._copy(sample_path, user_directory)
        
        return status
    
    def traverse_project(self, project_path):
        root, dirs, _ = next(os.walk(project_path))

        samples_in_project = {}
        for d in dirs:
            sample_directory = os.path.join(root, d)
            samples_in_project[d] = sample_directory

        return samples_in_project

    
    def initialize(self, module, sample_path, user_id):                
        # Create a new flux for module ClassifyProject
        if module == 'Classify project cell types':
            samples_in_project = self.traverse_project(sample_path)
            for sample, s_path in samples_in_project.items():
                if not self._sanity_check(s_path, user_id):
                    return

        else:
            if not self._sanity_check(sample_path, user_id):
                return
        
        self._grab_module(module)
        
        widgs = self.module.setup(sample_path, user_id)
        
        my_interact_manual = interact_manual.options(manual_name=self.module.button_name)
        my_interact_manual(self.execute, **widgs)
        
    def execute(self, **kwargs):
        self.module.execute(**kwargs)
        

class Workflow:

    def initialization_markdown(self):
        markdown_content = '''# <center> <I> REVIEW APPLICATION

<center><b>Classify cell types</b> module — User can classify the identified clusters per sample as positive or negative per marker. The user can also "Re-cluster" a cluster into sub-clusters.
<center><b>Classify project cell types</b> module — User can classify the identified clusters per batch of samples as positive or negative per marker.
<center><b>Reconcile pathology tissue annotations</b> module — User can import a .geojson file with tissue annotation metadata and overlay this information onto the corresponding sample's MARQO outputs. These annotations can usually be performed on QuPath.
<center><b>Create larger multi-dimensional images</b> module — User can choose multiple tiles per sample to create a multi-dimensional .ome.tiff image, with chromagen staining extracted for each Z-axis channel (length by width by number of markers).
<center><b>Create larger RGB tiles</b> module — User can choose multiple tiles per sample to create a multi-dimensional .ome.tiff image, with red-green-blue channels stacked into the Z-axis (length by width by 3*number of markers).
<center><b>Visualize cell populations</b> module — User can select a tile and view cell populations of interest by selecting combinations of positive-staining and negative-staining markers.


'''

        return markdown_content

    def start(self):
        notebook = Notebook()

        display(Markdown(self.initialization_markdown()))

        my_interact_manual = interact_manual.options(manual_name="Review")
        my_interact_manual(notebook.initialize, module = module, sample_path = sample_path, user_id = user_id);
