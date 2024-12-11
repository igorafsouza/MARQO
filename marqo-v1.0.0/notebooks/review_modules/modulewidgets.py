from ipywidgets import RadioButtons, IntSlider, SelectionSlider, SelectMultiple, ToggleButtons, Checkbox, Text, Dropdown, Layout, IntProgress, Button
# Widgets definition for intialization
user_id = Text(
    value='',
    placeholder='Type something',
    style = {'description_width': 'initial'},
    description='Reviewer name:',
    disabled=False,
    layout = Layout(width='500px'))


sample_path = Text(
    value='/sc/arion/projects/path_to_sample_folder',
    placeholder='Type something',
    style = {'description_width': 'initial'},
    description='Path to sample folder:',
    disabled=False,
    layout = Layout(width='800px'))

module = Dropdown(
        options = ['Classify cell types', 'Classify project cell types', 'Reconcile pathology tissue annotations', 'Create larger multi-dimensional images', 'Create larger RGB tiles','Visualize cell populations'],
        value = 'Classify cell types',
        description = 'Select module:',
        disabled = False,
        style = {'description_width': 'initial'},
        readout=True,
        readout_format='d',
        layout={'width': 'max-content'})


def marker_channel(markers):
    background_channel = RadioButtons(
        options=markers,
        value=markers[0],
        description='Choose marker to review:',
        style = {'description_width': 'initial'},
        layout={'width': 'max-content'}, # If the items' names are long
        disabled=False)

    return background_channel


def select_core(cores):
    core = RadioButtons(
        options=cores,
        value=cores[0],
        description='Choose Core to review:',
        style = {'description_width': 'initial'},
        layout={'width': 'max-content'}, # If the items' names are long
        disabled=False)

    return core


def roi_select(min_roi_index, max_roi_index):
    roi_select = IntSlider(
        value=min_roi_index,
        min=min_roi_index,
        max=max_roi_index,
        step=1,
        description='ROI:',
        disabled=False,
        continuous_update=False,
        orientation='vertical',
        readout=True,
        readout_format='d')  

    return roi_select

def roi_select_tma(rois):
    roi_select = SelectionSlider(
        options=rois,
        value=rois[0],
        description='ROI:',
        disabled=False,
        continuous_update=False,
        orientation='vertical',
        readout=True,
        readout_format='d')
 
    return roi_select

def pos_tiers(ranked_clusters, preexisting_pos_clusters):
    pos_tiers = SelectMultiple(
        options=[('Tier ' + str(tier)) for i, tier in enumerate(ranked_clusters)],
        value = [('Tier ' + str(tier)) for i, tier in enumerate(preexisting_pos_clusters)], 
        style = {'description_width': 'initial'},
        rows= len(ranked_clusters),
        description='Select positive tiers (can select multiple):',
        layout={'width': 'max-content'},
        disabled=False)

    return pos_tiers

def display_morph():
    display_morph = ToggleButtons(
        options=['Cytoplasm', 'Nucleus', 'Centroid', 'None'],
        description='Display cells by:',
        style = {'description_width': 'initial'},
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Cytoplasm', 'Nucleus', 'Centroid', 'None'],
        layout = Layout(width='300px'),
        orientation='vertical')

    return display_morph

def fill_cells():
    fill_cells = Checkbox(
        value=False,
        description='Fill in cell boundaries',
        disabled=False,
        indent=False)

    return fill_cells


def output_name(sample):
    output_name = Text(
        value=(sample + '_stitchedtile_UNIQUE-IDENTIFIER'),
        placeholder='Type something',
        style = {'description_width': 'initial'},
        description='Provide uniuque name for tile:',
        disabled=False,
        layout = Layout(width='800px'))

    return output_name

def annotation_geojson():
    annotation_geojson = Text(
        value='/sc/arion/projects/path_to_JSON_file',
        placeholder='Type something',
        style = {'description_width': 'initial'},
        description='Path to JSON or geoJSON file, including extension:',
        disabled=False,
        layout = Layout(width='1000px'))

    return annotation_geojson


def original_image():
    original_image = Text(
        value='/sc/arion/projects/path_to_image',
        placeholder='Type something',
        style = {'description_width': 'initial'},
        description='Path to image that annotations were performed on, including extension (ex. .svs or .ndpi):',
        disabled=False,
        layout = Layout(width='1000px'))

    return original_image

def marker_dropdown(markers): 
    marker_dropdown = Dropdown(
            options = [(str(index) + '. ' + str(marker_name)) for index, marker_name in enumerate(markers)],
            value = [(str(index) + '. ' + str(marker_name)) for index, marker_name in enumerate(markers)][0],
            description = 'Select which marker was annotated by the pathologist:',
            disabled = False,
            style = {'description_width': 'initial'},
            readout=True,
            readout_format='d',
            layout={'width': 'max-content'})

    return marker_dropdown


def progress_bar(description='', _max=10):
    progress_bar = IntProgress(
            value=0,
            min=0,
            max=_max,
            description=f'{description}:',
            bar_style='success', # 'success', 'info', 'warning', 'danger' or ''
            style={'bar_color': 'maroon'},
            orientation='horizontal'
        )

    return progress_bar

def pos_markers(markers):
    pos_markers = SelectMultiple(
        options=markers,
        value=[],
        rows=len(markers),
        description='+markers:',
        disabled=False)

    return pos_markers


def neg_markers(markers):
    neg_markers = SelectMultiple(
        options=markers,
        value=[],
        rows=len(markers),
        description='-markers:',
        disabled=False)

    return neg_markers


def background_channel(markers):
    background_channel = RadioButtons(
        options=markers,
        value=markers[0],
        description='Backdrop:',
        disabled=False)

    return background_channel

def show_all_cells():
    show_all_cells = Checkbox(
        value=True,
        description='Display all segmented cells',
        disabled=False,
        indent=False)

    return show_all_cells

#def edit_cells():
#    edit_cells = ToggleButtons(
#        options=['Add cells', 'Remove cells'],
#        value=None,
#        description='Select to remove or add the manually selected cells to the current expression pattern.',
#        style = {'description_width': 'initial'},
#        disabled=False,
#        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
#        layout = Layout(width='1000px'))

#    return edit_cells

def edit_button(description):
    button = Button(
        description=description,
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='',
    )

    return button


def save_editions():
    button = Button(
        description='Save Editions',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='',
    )
    return button

def annotation_list(annotations):
    annotation_list = SelectMultiple(
        options=annotations,
        value=[],
        rows=len(annotations),
        description='Annotations:',
        disabled=False)

    return annotation_list


def clusters_number():
    kclusters = RadioButtons(
        options=[5, 7],
        value=5,
        description='Choose the number of clusters:',
        style = {'description_width': 'initial'},
        disabled=False)

    return kclusters

def root_cluster(clusters):
    root = RadioButtons(
        options=clusters,
        value=0,
        description='Choose the root cluster to re-group sub-clusters:',
        style = {'description_width': 'initial'},
        disabled=False)

    return root

def channel_radio_buttons(technology):
    radio_buttons = RadioButtons(
        options=['RGB', 'Hematoxylin', 'AEC'] if technology not in ['if', 'cycif'] else ['DNA + chromogen', 'DNA', 'chromogen'],
        value='RGB' if technology not in ['if', 'cycif'] else 'DNA + chromogen',
        description='Select the channel to display:',
        orientation='vertical')

    return radio_buttons
