
import yaml
import numpy as np
import functools
import operator


# classes from namespaces
from skimage.measure import regionprops, label
from sklearn.neighbors import KNeighborsClassifier

# functions from namaspeaces
from skimage.draw import polygon


from utils.image import ImageHandler
from objects.cell import Cell
from utils.spatial import Spatial


class CompositeSegmentation:
    """
    Class to define methods to perform the composite segmentation, using StarDist2D model.
    """

    def __init__(self):
        pass


    def set_parameters(self, config_file: str):
        with open(config_file) as stream:
            config_data = yaml.safe_load(stream)

            self.noise_threshold = config_data['parameters']['segmentation']['noise_threshold']
            self.perc_consensus = config_data['parameters']['segmentation']['perc_consensus']

            self.segmentation_buffer = config_data['parameters']['segmentation']['segmentation_buffer']
            self.cyto_distances = config_data['parameters']['segmentation']['cyto_distances']

            self.image_scale_factor = config_data['parameters']['image']['scale_factor']

            self.markers_list = config_data['sample']['markers']
            self.technology = config_data['sample']['technology']

            if self.technology == 'cycif':
                self.cycles_metadata = config_data['cycles_metadata']
                


    def apply_model(self, roi, segmentation_model):
        if segmentation_model.name == 'stardist':
            scale = self.image_scale_factor / segmentation_model.pixel_size

            img_transformed = segmentation_model.transform_image(roi)
            labels, details = segmentation_model.model.predict_instances(img_transformed, scale=scale)
            centroids = details['points'] #outputs (y,x) format
            coordinates = details['coord']

        elif segmentation_model.name == 'cellpose':
            img_transformed = segmentation_model.transform_image(roi)

            if segmentation_model.cell_diameter != 0.0:
                masks, flows, styles, diams = segmentation_model.model.eval(img_transformed, diameter=segmentation_model.cell_diameter, flow_threshold=segmentation_model.flow, normalize=True, channels=[0, 0])

            else:
                masks, flows, styles, diams = segmentation_model.model.eval(img_transformed, diameter=None, flow_threshold=segmentation_model.flow, normalize=True, channels=[0, 0])
                
            # Find centroids and coordinates of nuclei
            props = regionprops(masks)
            centroids = [np.array(prop.centroid) for prop in props]  # Centroids in (y, x) format
            coordinates = []
            for prop in props:
                # Get the (row, column) coordinates of the pixels in this region
                coords = prop.coords
                yn, xn = coords[:, 0], coords[:, 1]
                coordinates.append([yn, xn])
 
        return centroids, coordinates


    def predict(self, registered_rois, roi_index, visualization_figs, segmentation_model):
        initial_centroids_across_images = []
        initial_centroids_across_images_combinedcoor = []
        initial_coords_across_images = []
        initial_centroids_across_images_boolean = []

        final_centroids = []
        final_nuc_edgecoords = []
        final_nuc_allcoords = []
        final_cyto_edgecoords = []
        final_cyto_allcoords = []

        training_labels = []
        training_points_y = []
        training_points_x = []

        allmarkers_geojson_list = []

        official_cell_count = 0
        num_stains = len(registered_rois)
        cell_count_thresh = int(self.perc_consensus * num_stains/100)

        height, width, null = registered_rois[0].shape  # grabbing the dimensions of the tiles
        nuclei_mask=np.zeros((height,width))
        final_cyto_mask = np.zeros((height,width))
        final_cytoedge_mask = np.zeros((height,width))

        for roi in registered_rois:

            centroids, coords = self.apply_model(roi, segmentation_model)
            print(f'Celss segmented: {len(coords)}')

            initial_centroids_across_images_combinedcoor.append(centroids) #appends (y,x) coordinates while combined
            centroids = np.array(list((zip(*centroids)))) #separate y and x coordinates
            initial_centroids_across_images.append(centroids) #appends (y,x) coordinates while separated
            initial_coords_across_images.append(coords) #appends coords, aka coordinate edges (y,x) of cell nuclei
            initial_centroids_across_images_boolean.append(np.array([0]*len(coords))) #ID sparse array for valid cells

        for tile_index, centroids_in_tile in enumerate(initial_centroids_across_images_combinedcoor):
            for possible_cell_index, possible_cell in enumerate(centroids_in_tile):
                if initial_centroids_across_images_boolean[tile_index][possible_cell_index] == 0:
                    bottom_thresh_x = possible_cell[1] - self.noise_threshold
                    bottom_thresh_y = possible_cell[0] - self.noise_threshold
                    top_thresh_x = possible_cell[1] + self.noise_threshold
                    top_thresh_y = possible_cell[0] + self.noise_threshold
                    cell_count = 0 #count # of times cell is repeated across stains
                    
                    for remaining_stain_index in range(len(initial_centroids_across_images)): 
                        # compare to only other stains and if there are cells for the respective stain  
                        if (remaining_stain_index != tile_index) and len(initial_centroids_across_images[remaining_stain_index]) > 0:
                            flag1 = np.where(initial_centroids_across_images[remaining_stain_index][0] > bottom_thresh_y)
                            flag2 = np.where(initial_centroids_across_images[remaining_stain_index][1] > bottom_thresh_x)
                            flag3 = np.where(initial_centroids_across_images[remaining_stain_index][0] < top_thresh_y)
                            flag4 = np.where(initial_centroids_across_images[remaining_stain_index][1] < top_thresh_x)
                            flag5 = np.where(np.array(initial_centroids_across_images_boolean[remaining_stain_index]) == 0)
                            final_index = list(set(flag1[0]) & set(flag2[0]) & set(flag3[0]) & set(flag4[0]) & set(flag5[0])) #find all cells that meet 5 flagged requirements
                            
                            if final_index:
                                initial_centroids_across_images_boolean[remaining_stain_index][final_index] =1 #do not iterate over the same same across multiple stains
                                cell_count = cell_count + 1

                    condition1 = cell_count > cell_count_thresh #keep cell if it repeats over X% of stains
                    condition2 = (possible_cell > self.segmentation_buffer).all() and (possible_cell < (height - self.segmentation_buffer)).all()  #keep cell if its within cropped bounds after subtracting segmentation buffer
                    if (condition1 and condition2):
                        #save centroids
                        final_centroids.append(list(possible_cell))
                        #save nuclear edge coordinates
                        yn, xn = initial_coords_across_images[tile_index][possible_cell_index]

                        yn[np.where(yn >= height)] = height - 1
                        xn[np.where(xn >= height)] = width - 1

                        final_nuc_edgecoords.append([yn, xn])
                        current_nuc_allcoords_y, current_nuc_allcoords_x = polygon(yn,xn)  #polygon fxn outputs y,x
                        #save nuclear all coordinates
                        current_nuc_allcoords_y[current_nuc_allcoords_y >= height] = height - 1  #ensure nuclei coords are within bounds
                        current_nuc_allcoords_x[current_nuc_allcoords_x >= width] = width - 1  #ensure nuclei coords are within bounds
                        final_nuc_allcoords.append([current_nuc_allcoords_y, current_nuc_allcoords_x])
                        nuclei_mask[current_nuc_allcoords_y, current_nuc_allcoords_x] = 1
                        #init values for k-clustering when artifically expanding overlapping cytosplasms later.
                        training_labels = training_labels + [official_cell_count]*len(current_nuc_allcoords_y)
                        training_points_x = training_points_x + list(current_nuc_allcoords_x)
                        training_points_y = training_points_y + list(current_nuc_allcoords_y)
                        official_cell_count = official_cell_count+1
        
        nuclei_mask.astype(bool) #fianl nuclei mask
        training_points = np.column_stack((training_points_y, training_points_x))

        # if no cells were detected in the previous block, we dont execute the following one 
        if len(training_points) > 0:
            #determine nuclei and cytoplasm edge and total coordinates
            neigh = KNeighborsClassifier(n_neighbors=1) 
            neigh.fit(training_points, training_labels)

            for stain_index, stain in enumerate(registered_rois):
                final_cyto_mask = np.zeros((height, width))
                final_cyto_allcoords.append([])
                final_cyto_edgecoords.append([])
                cyto_expansion = self.cyto_distances[stain_index]
                geojson_list = []

                if cyto_expansion >= self.segmentation_buffer: #ensure cytoplasm expansion still resides inside tile bounds
                    cyto_expansion = self.segmentation_buffer - 1

                for cell_index, cell_coords in enumerate(final_nuc_edgecoords): #iterate across cells per stain since cyto_expansion is diff per stain
                    #artificially expand cytoplasm based on nuclear edge coordinates
                    xcenter = np.mean(cell_coords[1])
                    ycenter = np.mean(cell_coords[0])
                    xnucedges = cell_coords[1]
                    ynucedges = cell_coords[0]
                    a=(xnucedges-xcenter)**2
                    b=(ynucedges-ycenter)**2
                    lengths = np.sqrt(a + b)
                    unitSlopeX = (xnucedges - xcenter) / lengths
                    unitSlopeY = (ynucedges - ycenter) / lengths
                    xcytedges = np.round(xcenter + unitSlopeX * (lengths + cyto_expansion))
                    ycytedges = np.round(ycenter + unitSlopeY * (lengths + cyto_expansion))
                    cell_allcoords_y, cell_allcoords_x = polygon(ycytedges, xcytedges)
                    #fill in cyto edge indices
                    try:
                        pred = neigh.predict(list(zip(np.asarray(cell_allcoords_y),np.asarray(cell_allcoords_x))))
                        cell_allcoords_x_updated = cell_allcoords_x[(pred==cell_index)]
                        cell_allcoords_y_updated = cell_allcoords_y[(pred==cell_index)]
                    
                    except: #if k-clustering fails, default to cytoplasm edges
                        cell_allcoords_x_updated = xcytedges.astype(int)
                        cell_allcoords_y_updated = ycytedges.astype(int)
                    
                    #ensure all coords are within bounds of tile -- cyto edges
                    xcytedges_updated = xcytedges.astype(int)
                    ycytedges_updated = ycytedges.astype(int)

                    x_inidices_to_keep_forcytoedges = np.where((0 < xcytedges_updated) & (xcytedges_updated < width))
                    xcytedges_updated = xcytedges_updated[x_inidices_to_keep_forcytoedges]
                    ycytedges_updated = ycytedges_updated[x_inidices_to_keep_forcytoedges]

                    y_inidices_to_keep_forcytoedges = np.where((0 < ycytedges_updated) & (ycytedges_updated < height))
                    xcytedges_updated = xcytedges_updated[y_inidices_to_keep_forcytoedges]
                    ycytedges_updated = ycytedges_updated[y_inidices_to_keep_forcytoedges]

                    #ensure all coords are within bounds of tile -- all cyto coords
                    x_inidices_to_keep = np.where((0 < cell_allcoords_x_updated) & (cell_allcoords_x_updated < width))
                    cell_allcoords_x_updated = cell_allcoords_x_updated[x_inidices_to_keep]
                    cell_allcoords_y_updated = cell_allcoords_y_updated[x_inidices_to_keep]

                    y_inidices_to_keep = np.where((0 < cell_allcoords_y_updated) & (cell_allcoords_y_updated < height))
                    cell_allcoords_x_updated = cell_allcoords_x_updated[y_inidices_to_keep]
                    cell_allcoords_y_updated = cell_allcoords_y_updated[y_inidices_to_keep]

                    #save outputs for cyto edges, aka membrane
                    cytoedge_mask = np.zeros((height,width))
                    cytoedge_mask.astype(bool)
                    cytoedge_mask[ycytedges_updated, xcytedges_updated] = True
                    cytoedge_mask[nuclei_mask.astype(bool)] = False #ensure predictions didnt overlap with any existing nuceli coordinates
                    cyto_edgecoords_y, cyto_edgecoords_x = np.where(cytoedge_mask)
                    
                    if len(cyto_edgecoords_y)==0: #in dense regions, if no cytoplasm expansion region exists, default to nuclear boundaries
                        cyto_edgecoords_y = final_nuc_edgecoords[cell_index][0].astype(int)
                        cyto_edgecoords_x = final_nuc_edgecoords[cell_index][1].astype(int)
                    
                    final_cyto_edgecoords[stain_index].append([cyto_edgecoords_y, cyto_edgecoords_x])
                    final_cytoedge_mask[cyto_edgecoords_y, cyto_edgecoords_x] = True
                    #save outputs for all cyto coords
                    cyto_mask = np.zeros((height,width))
                    cyto_mask.astype(bool)
                    cyto_mask[cell_allcoords_y_updated, cell_allcoords_x_updated] = True
                    cyto_mask[nuclei_mask.astype(bool)] = False #ensure predictions didnt overlap with any existing nuceli coordinates
                    cyto_allcoords_y, cyto_allcoords_x = np.where(cyto_mask)
                    
                    if len(cyto_allcoords_y)==0: #in dense regions, if no cytoplasm expansion region exists, default to nuclear boundaries
                        cyto_allcoords_y = final_nuc_edgecoords[cell_index][0].astype(int)
                        cyto_allcoords_x = final_nuc_edgecoords[cell_index][1].astype(int)

                    try:
                        # this if is to prevent colinearity, which rises an exception in get_hull
                        if not np.all(cell_allcoords_x_updated == cell_allcoords_x_updated[0]) and not np.all(cell_allcoords_y_updated == cell_allcoords_y_updated[0]):

                            cyto_hull_coordinates = Spatial.get_hull(cell_allcoords_x_updated, cell_allcoords_y_updated, self.segmentation_buffer)
                            nuc_hull_coordinates = Spatial.get_hull(xnucedges, ynucedges, self.segmentation_buffer)

                        else:
                            cyto_hull_coordinates = list(zip(cell_allcoords_x_updated, cell_allcoords_y_updated))
                            cyto_hull_coordinates = [[float(pair[0]), float(pair[1])] for pair in cyto_hull_coordinates]
                            cyto_hull_coordinates.append(cyto_hull_coordinates[0])

                            nuc_hull_coordinates = list(zip(xnucedges, ynucedges))
                            nuc_hull_coordinates = [[float(pair[0]), float(pair[1])] for pair in nuc_hull_coordinates]
                            nuc_hull_coordinates.append(nuc_hull_coordinates[0])

                        cell = Cell(roi_index, cell_index, cyto_hull_coordinates, nuc_hull_coordinates)
                        geojson_list.append(cell)

                        final_cyto_allcoords[stain_index].append([cyto_allcoords_y, cyto_allcoords_x])
                        final_cyto_mask[cell_allcoords_y_updated, cell_allcoords_x_updated] = True

                    except Exception as err:
                        print(f'ROI index: {roi_index}')
                        print(f'X Coords: {cell_allcoords_x_updated}\nY Coords: {cell_allcoords_y_updated}')
                        print(f'[X] Cell index #{cell_index} failed. Exceptions:\n{err}')


                #for visualization
                seg_mask_cytoandnuc = np.zeros((height,width,3)).astype(np.uint8)
                seg_mask_cytoandnuc[np.where(final_cyto_mask)] = stain[np.where(final_cyto_mask)]
                visualization_figs['segmentation'].append([stain, initial_centroids_across_images[stain_index], final_centroids, seg_mask_cytoandnuc])

                #begin writing geojson
                allmarkers_geojson_list.append(geojson_list)


        results = {
            'final_centroids': final_centroids, 
            'final_nuc_edgecoords': final_nuc_edgecoords, 
            'final_nuc_allcoords': final_nuc_allcoords, 
            'final_cyto_edgecoords': final_cyto_edgecoords, 
            'final_cyto_allcoords': final_cyto_allcoords, 
            'allmarkers_geojson_list': allmarkers_geojson_list
        }

        return results


    def predict_dapi(self, registered_rois, roi_index, visualization_figs, segmentation_model):
        initial_centroids_across_images = []
        initial_centroids_across_images_combinedcoor = []
        initial_coords_across_images = []
        initial_centroids_across_images_boolean = []

        final_centroids = []
        final_nuc_edgecoords = []
        final_nuc_allcoords = []
        final_cyto_edgecoords = []
        final_cyto_allcoords = []

        training_labels = []
        training_points_y = []
        training_points_x = []

        allmarkers_geojson_list = []

        official_cell_count = 0
        num_stains = len(registered_rois)
        cell_count_thresh = int(self.perc_consensus * num_stains/100)

        height = registered_rois[0].shape[0]  # grabbing the dimensions of the tiles
        width = registered_rois[0].shape[1]  # grabbing the dimensions of the tiles

        print(f'Dimensions: ({height}, {width})')

        nuclei_mask=np.zeros((height,width))
        final_cyto_mask = np.zeros((height,width))
        final_cytoedge_mask = np.zeros((height,width))

        cyto_distances_across_channels = [v['cyto_distances'] for k, v in self.cycles_metadata.items()]  
        print(f'Cyto expansions:\n{cyto_distances_across_channels}')
        cyto_distances_across_channels = functools.reduce(operator.iconcat, cyto_distances_across_channels, [])  # flatten the list of lists

        for roi in registered_rois:

            centroids, coords = self.apply_model(roi, segmentation_model)

            print(f'Segmented celss: {len(coords)}')

            initial_centroids_across_images_combinedcoor.append(centroids) #appends (y,x) coordinates while combined
            centroids = np.array(list((zip(*centroids)))) #separate y and x coordinates
            initial_centroids_across_images.append(centroids) #appends (y,x) coordinates while separated
            initial_coords_across_images.append(coords) #appends coords, aka coordinate edges (y,x) of cell nuclei
            initial_centroids_across_images_boolean.append(np.array([0]*len(coords))) #ID sparse array for valid cells

        for tile_index, centroids_in_tile in enumerate(initial_centroids_across_images_combinedcoor):
            for possible_cell_index, possible_cell in enumerate(centroids_in_tile):
                if initial_centroids_across_images_boolean[tile_index][possible_cell_index] == 0:
                    bottom_thresh_x = possible_cell[1] - self.noise_threshold
                    bottom_thresh_y = possible_cell[0] - self.noise_threshold
                    top_thresh_x = possible_cell[1] + self.noise_threshold
                    top_thresh_y = possible_cell[0] + self.noise_threshold
                    cell_count = 0 #count # of times cell is repeated across stains
                    
                    for remaining_stain_index in range(len(initial_centroids_across_images)): 
                        # compare to only other stains and if there are cells for the respective stain  
                        if (remaining_stain_index != tile_index) and len(initial_centroids_across_images[remaining_stain_index]) > 0:
                            flag1 = np.where(initial_centroids_across_images[remaining_stain_index][0] > bottom_thresh_y)
                            flag2 = np.where(initial_centroids_across_images[remaining_stain_index][1] > bottom_thresh_x)
                            flag3 = np.where(initial_centroids_across_images[remaining_stain_index][0] < top_thresh_y)
                            flag4 = np.where(initial_centroids_across_images[remaining_stain_index][1] < top_thresh_x)
                            flag5 = np.where(np.array(initial_centroids_across_images_boolean[remaining_stain_index]) == 0)
                            final_index = list(set(flag1[0]) & set(flag2[0]) & set(flag3[0]) & set(flag4[0]) & set(flag5[0])) #find all cells that meet 5 flagged requirements
                            
                            if final_index:
                                initial_centroids_across_images_boolean[remaining_stain_index][final_index] =1 #do not iterate over the same same across multiple stains
                                cell_count = cell_count + 1

                    condition1 = cell_count > cell_count_thresh #keep cell if it repeats over X% of stains
                    condition2 = (possible_cell > self.segmentation_buffer).all() and (possible_cell < (height - self.segmentation_buffer)).all()  #keep cell if its within cropped bounds after subtracting segmentation buffer
                    if (condition1 and condition2):
                        #save centroids
                        final_centroids.append(list(possible_cell))
                        #save nuclear edge coordinates
                        yn, xn = initial_coords_across_images[tile_index][possible_cell_index]

                        yn[np.where(yn >= height)] = height - 1
                        xn[np.where(xn >= height)] = width - 1

                        final_nuc_edgecoords.append([yn, xn])
                        current_nuc_allcoords_y, current_nuc_allcoords_x = polygon(yn,xn)  #polygon fxn outputs y,x
                        #save nuclear all coordinates
                        current_nuc_allcoords_y[current_nuc_allcoords_y >= height] = height - 1  #ensure nuclei coords are within bounds
                        current_nuc_allcoords_x[current_nuc_allcoords_x >= width] = width - 1  #ensure nuclei coords are within bounds
                        final_nuc_allcoords.append([current_nuc_allcoords_y, current_nuc_allcoords_x])
                        nuclei_mask[current_nuc_allcoords_y, current_nuc_allcoords_x] = 1
                        #init values for k-clustering when artifically expanding overlapping cytosplasms later.
                        training_labels = training_labels + [official_cell_count]*len(current_nuc_allcoords_y)
                        training_points_x = training_points_x + list(current_nuc_allcoords_x)
                        training_points_y = training_points_y + list(current_nuc_allcoords_y)
                        official_cell_count = official_cell_count+1
        
        nuclei_mask.astype(bool) #final nuclei mask
        training_points = np.column_stack((training_points_y, training_points_x))

        # if no cells were detected in the previous block, we dont execute the following one 
        if len(training_points) > 0:
            #determine nuclei and cytoplasm edge and total coordinates
            neigh = KNeighborsClassifier(n_neighbors=1) 
            neigh.fit(training_points, training_labels)

            print(f'Cyto expansions:\n{cyto_distances_across_channels}')
            for stain_index, _ in enumerate(cyto_distances_across_channels):
                final_cyto_mask = np.zeros((height, width))
                final_cyto_allcoords.append([])
                final_cyto_edgecoords.append([])
                cyto_expansion = cyto_distances_across_channels[stain_index]
                geojson_list = []

                if cyto_expansion >= self.segmentation_buffer: #ensure cytoplasm expansion still resides inside tile bounds
                    cyto_expansion = self.segmentation_buffer - 1

                for cell_index, cell_coords in enumerate(final_nuc_edgecoords): #iterate across cells per stain since cyto_expansion is diff per stain
                    #artificially expand cytoplasm based on nuclear edge coordinates
                    xcenter = np.mean(cell_coords[1])
                    ycenter = np.mean(cell_coords[0])
                    xnucedges = cell_coords[1]
                    ynucedges = cell_coords[0]
                    a=(xnucedges-xcenter)**2
                    b=(ynucedges-ycenter)**2
                    lengths = np.sqrt(a + b)
                    unitSlopeX = (xnucedges - xcenter) / lengths
                    unitSlopeY = (ynucedges - ycenter) / lengths
                    xcytedges = np.round(xcenter + unitSlopeX * (lengths + cyto_expansion))
                    ycytedges = np.round(ycenter + unitSlopeY * (lengths + cyto_expansion))
                    cell_allcoords_y, cell_allcoords_x = polygon(ycytedges, xcytedges)
                    #fill in cyto edge indices
                    try:
                        pred = neigh.predict(list(zip(np.asarray(cell_allcoords_y),np.asarray(cell_allcoords_x))))
                        cell_allcoords_x_updated = cell_allcoords_x[(pred==cell_index)]
                        cell_allcoords_y_updated = cell_allcoords_y[(pred==cell_index)]
                    
                    except: #if k-clustering fails, default to cytoplasm edges
                        cell_allcoords_x_updated = xcytedges.astype(int)
                        cell_allcoords_y_updated = ycytedges.astype(int)
                    
                    #ensure all coords are within bounds of tile -- cyto edges
                    xcytedges_updated = xcytedges.astype(int)
                    ycytedges_updated = ycytedges.astype(int)

                    x_inidices_to_keep_forcytoedges = np.where((0 < xcytedges_updated) & (xcytedges_updated < width))
                    xcytedges_updated = xcytedges_updated[x_inidices_to_keep_forcytoedges]
                    ycytedges_updated = ycytedges_updated[x_inidices_to_keep_forcytoedges]

                    y_inidices_to_keep_forcytoedges = np.where((0 < ycytedges_updated) & (ycytedges_updated < height))
                    xcytedges_updated = xcytedges_updated[y_inidices_to_keep_forcytoedges]
                    ycytedges_updated = ycytedges_updated[y_inidices_to_keep_forcytoedges]

                    #ensure all coords are within bounds of tile -- all cyto coords
                    x_inidices_to_keep = np.where((0 < cell_allcoords_x_updated) & (cell_allcoords_x_updated < width))
                    cell_allcoords_x_updated = cell_allcoords_x_updated[x_inidices_to_keep]
                    cell_allcoords_y_updated = cell_allcoords_y_updated[x_inidices_to_keep]

                    y_inidices_to_keep = np.where((0 < cell_allcoords_y_updated) & (cell_allcoords_y_updated < height))
                    cell_allcoords_x_updated = cell_allcoords_x_updated[y_inidices_to_keep]
                    cell_allcoords_y_updated = cell_allcoords_y_updated[y_inidices_to_keep]

                    #save outputs for cyto edges, aka membrane
                    cytoedge_mask = np.zeros((height,width))
                    cytoedge_mask.astype(bool)
                    cytoedge_mask[ycytedges_updated, xcytedges_updated] = True
                    cytoedge_mask[nuclei_mask.astype(bool)] = False #ensure predictions didnt overlap with any existing nuceli coordinates
                    cyto_edgecoords_y, cyto_edgecoords_x = np.where(cytoedge_mask)
                    
                    if len(cyto_edgecoords_y)==0: #in dense regions, if no cytoplasm expansion region exists, default to nuclear boundaries
                        cyto_edgecoords_y = final_nuc_edgecoords[cell_index][0].astype(int)
                        cyto_edgecoords_x = final_nuc_edgecoords[cell_index][1].astype(int)
                    
                    final_cyto_edgecoords[stain_index].append([cyto_edgecoords_y, cyto_edgecoords_x])
                    final_cytoedge_mask[cyto_edgecoords_y, cyto_edgecoords_x] = True
                    #save outputs for all cyto coords
                    cyto_mask = np.zeros((height,width))
                    cyto_mask.astype(bool)
                    cyto_mask[cell_allcoords_y_updated, cell_allcoords_x_updated] = True
                    cyto_mask[nuclei_mask.astype(bool)] = False #ensure predictions didnt overlap with any existing nuceli coordinates
                    cyto_allcoords_y, cyto_allcoords_x = np.where(cyto_mask)
                    
                    if len(cyto_allcoords_y)==0: #in dense regions, if no cytoplasm expansion region exists, default to nuclear boundaries
                        cyto_allcoords_y = final_nuc_edgecoords[cell_index][0].astype(int)
                        cyto_allcoords_x = final_nuc_edgecoords[cell_index][1].astype(int)

                    try:
                        # this if is to prevent colinearity, which rises an exception in get_hull
                        if not np.all(cell_allcoords_x_updated == cell_allcoords_x_updated[0]) and not np.all(cell_allcoords_y_updated == cell_allcoords_y_updated[0]):

                            cyto_hull_coordinates = Spatial.get_hull(cell_allcoords_x_updated, cell_allcoords_y_updated, self.segmentation_buffer)
                            nuc_hull_coordinates = Spatial.get_hull(xnucedges, ynucedges, self.segmentation_buffer)

                        else:
                            cyto_hull_coordinates = list(zip(cell_allcoords_x_updated, cell_allcoords_y_updated))
                            cyto_hull_coordinates = [[float(pair[0]), float(pair[1])] for pair in cyto_hull_coordinates]
                            cyto_hull_coordinates.append(cyto_hull_coordinates[0])

                            nuc_hull_coordinates = list(zip(xnucedges, ynucedges))
                            nuc_hull_coordinates = [[float(pair[0]), float(pair[1])] for pair in nuc_hull_coordinates]
                            nuc_hull_coordinates.append(nuc_hull_coordinates[0])


                        cell = Cell(roi_index, cell_index, cyto_hull_coordinates, nuc_hull_coordinates)
                        geojson_list.append(cell)

                        final_cyto_allcoords[stain_index].append([cyto_allcoords_y, cyto_allcoords_x])
                        final_cyto_mask[cell_allcoords_y_updated, cell_allcoords_x_updated] = True

                    except Exception as err:
                        print(f'ROI index: {roi_index}')
                        print(f'X Coords: {cell_allcoords_x_updated}\nY Coords: {cell_allcoords_y_updated}')
                        print(f'[X] Cell index #{cell_index} failed. Exceptions:\n{err}')

                #begin writing geojson
                print(f'Current marker has #{len(geojson_list)} cells')
                allmarkers_geojson_list.append(geojson_list)


        results = {
            'final_centroids': final_centroids, 
            'final_nuc_edgecoords': final_nuc_edgecoords, 
            'final_nuc_allcoords': final_nuc_allcoords, 
            'final_cyto_edgecoords': final_cyto_edgecoords, 
            'final_cyto_allcoords': final_cyto_allcoords, 
            'allmarkers_geojson_list': allmarkers_geojson_list
        }

        return results
