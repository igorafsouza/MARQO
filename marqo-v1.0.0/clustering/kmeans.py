
import os
import yaml
import traceback
import numpy as np
import pandas as pd
import functools
import operator
from datetime import datetime
from time import time

from sklearn import preprocessing
from sklearn import decomposition
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering

class KMEANS:
    """
    """

    def __init__(self, parent_class=object):
        self.k_clusters = 7
        self.var_expl = 0.90
        self.parent_class = parent_class


    def _set_parameters(self, config_file):
        with open(config_file, 'r') as stream:
            self.config_data = yaml.safe_load(stream)

        try:
            self.marker_name_list = self.config_data['sample']['markers']
            self.sample_name = self.config_data['sample']['name']
            self.marco_output_path = self.config_data['directories']['marco_output']
            self.technology = self.config_data['sample']['technology']
            self.cores_labels = self.config_data['sample']['cores_labels']

            if self.technology == 'if':
                self.dna_marker_channel = self.config_data['sample']['dna_marker_channel']

            elif self.technology == 'cyif':
                self.cycles_metadata = self.config_data['cycles_metadata']
                marker_list_across_channels = [v['marker_name_list'] for k, v in self.cycles_metadata.items()]
                marker_list_across_channels = functools.reduce(operator.iconcat, marker_list_across_channels, [])  # flatten the list of lists
                cycles = [f'cycle_{i}' for i, v in self.cycles_metadata.items() for _ in range(len(v['marker_name_list']))]
                self.marker_name_list = [f'{cycle}_{marker}' for marker, cycle in zip(marker_list_across_channels, cycles)]

        except:
            self.marker_name_list = self.config_data['sample']['markers']
            self.sample_name = self.config_data['sample']['sample_name']
            self.marco_output_path = self.config_data['directories']['sample_directory']
            self.technology = 'micsss'

        self.config_file = config_file
        self.tmp_directory = os.path.join(self.marco_output_path, '.tmp')

    def cluster_files_reconciliation(self):
        tmp_files_path = os.path.join(self.marco_output_path, '.tmp')

        # Open the file and keep it opened
        n_stains = len(self.marker_name_list)
        final_clusters_path = os.path.join(self.marco_output_path, f'{self.sample_name}_finalcellinfo.csv')

        is_first = True
        for i in range(n_stains):
            marker = self.marker_name_list[i]
            _file = os.path.join(tmp_files_path, f'{marker}_finalcellinfo.csv')

            if os.path.exists(_file):
                df_tmp = pd.read_csv(_file)
                #print(f'Grabing {_file} | {len(df_tmp)} lines')

                if is_first:
                    df_tmp[f'{marker}_expression'] = ''
                    final_clusters_data = df_tmp
                    is_first = False

                else:
                    tier_column = df_tmp[f'{marker}_tier'].values
                    x_coor = df_tmp[f'{marker}_x-coord (pixels)'].values
                    y_coor = df_tmp[f'{marker}_y-coord (pixels)'].values

                    final_clusters_data[f'{marker}_x-coord (pixels)'] = x_coor
                    final_clusters_data[f'{marker}_y-coord (pixels)'] = y_coor
                    final_clusters_data[f'{marker}_tier'] = tier_column
                    final_clusters_data[f'{marker}_expression'] = ''

            else:
                print(f'Marker {i} didnt produce clusters data.')

        final_clusters_data['Tissue annotation'] = ''
        final_clusters_data.to_csv(final_clusters_path, index=False)

        print('Clustering step finished.')


    def _check_for_cores(self):
        self.tilestats_file = os.path.join(self.marco_output_path, f'{self.sample_name}_tilestats.csv')
        tilestats_df = pd.read_csv(self.tilestats_file)

        if self.cores_labels:
            self.tile_map = {tile: label for tile, label in zip(tilestats_df['Tile index'], tilestats_df['Core Label'])}
            self.cores_map = {}
            for key, value in self.tile_map.items():
                if value not in self.cores_map:
                    self.cores_map[value] = []
                self.cores_map[value].append(key)


    def _read_rawcellmetrics(self, marker):
        
        rawcellmetrics_file = os.path.join(self.marco_output_path, f'{self.sample_name}_rawcellmetrics.csv')
        raw_df = pd.read_csv(rawcellmetrics_file, sep=',', header=0)
        raw_df.reset_index(inplace=True, drop=True)
        raw_df.index.rename('Cell index', inplace=True)

        raw_df.rename(columns={'ROI-relative cell y_coord (pixels)': 'Relative y_coord', 'ROI-relative cell x_coord (pixels)': 'Relative x_coord', f'{marker} y_coor (pixels)': f'{marker}_y-coord (pixels)', f'{marker} x_coor (pixels)': f'{marker}_x-coord (pixels)'}, inplace=True)
        
        cols_to_subset = ['Tile index', 'Relative y_coord', 'Relative x_coord', f'{marker}_y-coord (pixels)', f'{marker}_x-coord (pixels)']
        final_marker_info = raw_df[raw_df.columns.intersection(cols_to_subset)]

        return raw_df, final_marker_info


    def create_project_rawcellmetrics(self, marker):
        raw_df_list = []
        final_df_list = []

        for sample_name, rawcellmetrics_file in zip(self.sample_names, self.rawcellmetrics_list):
            raw_df = pd.read_csv(rawcellmetrics_file, sep=',', header=0)
            raw_df.reset_index(inplace=True, drop=True)
            raw_df.index.rename('Cell index', inplace=True)

            raw_df.rename(columns={'ROI-relative cell y_coord (pixels)': 'Relative y_coord', 'ROI-relative cell x_coord (pixels)': 'Relative x_coord', f'{marker} y_coor (pixels)': f'{marker}_y-coord (pixels)', f'{marker} x_coor (pixels)': f'{marker}_x-coord (pixels)'}, inplace=True)

            cols_to_subset = ['Tile index', 'Relative y_coord', 'Relative x_coord', f'{marker}_y-coord (pixels)', f'{marker}_x-coord (pixels)']
            final_marker_info = raw_df[raw_df.columns.intersection(cols_to_subset)].copy()
            final_marker_info.loc[:, 'sample_name'] = sample_name

            #raw_df_scaled = pd.DataFrame(preprocessing.scale(raw_df), columns=raw_df.columns)

            raw_df_list.append(raw_df) 
            final_df_list.append(final_marker_info) 

        project_raw_df = pd.concat(raw_df_list)
        project_raw_df.reset_index(inplace=True, drop=True)
        project_raw_df.index.rename('Cell index', inplace=True)

        project_final_marker_info = pd.concat(final_df_list)
        project_final_marker_info.reset_index(inplace=True, drop=True)
        project_final_marker_info.index.rename('Cell index', inplace=True)

        return project_raw_df, project_final_marker_info
 
        
    def calculate_perc_medians(self, marker_clusters, marker):        
        medians = {}

        #marker_data = marker_data.merge(clusters, on='Cell index')
        clusters = marker_clusters[f'{marker}_tier'].unique().tolist()
        for cluster in clusters:
            count = len(marker_clusters.loc[marker_clusters[f'{marker}_tier'] == cluster].index)
            medians[f'{cluster}-{count}'] = {'nuc': {}, 'cyt': {}}
            
            # iterate over all percentiles calculating the respective medians
            for i in range(1, 10):
                # For the love of god, change it for a smarter way to apply this block to different technologies (if and ihc)
                try:
                    aec_nuc_values = marker_clusters.loc[marker_clusters[f'{marker}_tier'] == cluster, f'{marker} nuc AEC p{i}']
                    aec_cyt_values = marker_clusters.loc[marker_clusters[f'{marker}_tier'] == cluster, f'{marker} cyto AEC p{i}']
                
                except:
                    aec_nuc_values = marker_clusters.loc[marker_clusters[f'{marker}_tier'] == cluster, f'{marker} nuc chromogen p{i}']
                    aec_cyt_values = marker_clusters.loc[marker_clusters[f'{marker}_tier'] == cluster, f'{marker} cyto chromogen p{i}']

                median_nuc = np.median(aec_nuc_values)
                median_cyt = np.median(aec_cyt_values)
                medians[f'{cluster}-{count}']['nuc'][f'p{i}'] = median_nuc.tolist()
                medians[f'{cluster}-{count}']['cyt'][f'p{i}'] = median_cyt.tolist()

        return medians


    def write_perc_medians(self, perc_medians):
        with open(self.config_file) as stream:
            config_data = yaml.safe_load(stream)

        data = {}
        #for results in perc_medians:
        #    print(results)
        for marker in perc_medians:
            data.update(marker)

        config_data['k-clusters'] = data
        if self.cores_labels:
            config_data['sample']['cores'] = list(set(self.cores_map.keys()))
        
        with open(self.config_file, 'w+') as fout:
            yaml.dump(config_data, fout)
        
    
    def write_perc_medians_project(self, sample_name, perc_medians, marco_output_path):
        config_file = os.path.join(marco_output_path, 'config.yaml')
        with open(config_file) as stream:
            config_data = yaml.safe_load(stream)

        data = {}
        for results in perc_medians:
            for marker in results:
                data.update(marker[sample_name])

        config_data['k-clusters'] = data

        with open(config_file, 'w+') as fout:
            yaml.dump(config_data, fout)

    def _update_perc_medians_regroup(self, medians, root_cluster, marker):

        with open(self.config_file) as stream:
            config_data = yaml.safe_load(stream)

        keys_to_remove = []

        if self.cores_labels:
            for k, v in config_data['k-clusters'][marker][self.core].items():
                if float(k.split('-')[-2]) >= float(root_cluster) and float(k.split('-')[-2]) < float(root_cluster + 1):
                    keys_to_remove.append(k)

            for k in keys_to_remove:
                config_data['k-clusters'][marker][self.core].pop(k, None)

            for k, v in medians.items():
                config_data['k-clusters'][marker][self.core][k] = v

        else:
            for k, v in config_data['k-clusters'][marker].items():
                if float(k.split('-')[-2]) >= float(root_cluster) and float(k.split('-')[-2]) < float(root_cluster + 1):
                    keys_to_remove.append(k)

            for k in keys_to_remove:
                config_data['k-clusters'][marker].pop(k, None)

            for k, v in medians.items():
                config_data['k-clusters'][marker][k] = v

        with open(self.config_file, 'w+') as fout:
            yaml.dump(config_data, fout)


    def update_perc_medians(self, medians, target_cluster, marker):
        # TODO: rewrite this uglyness

        with open(self.config_file) as stream:
            config_data = yaml.safe_load(stream)
        
        if self.cores_labels:
            kclusters = config_data['k-clusters'][marker][self.core]
            for k, v in kclusters.items():
                if float(k.split('-')[-2]) == float(target_cluster):
                    config_data['k-clusters'][marker][self.core].pop(k, None)
                    break

            for k, v in medians.items():
                config_data['k-clusters'][marker][self.core][k] = v

        else:
            kclusters = config_data['k-clusters'][marker]
            for k, v in kclusters.items():
                if float(k.split('-')[-2]) == float(target_cluster):
                    config_data['k-clusters'][marker].pop(k, None)
                    break

            for k, v in medians.items():
                config_data['k-clusters'][marker][k] = v
                
        
        with open(self.config_file, 'w+') as fout:
            yaml.dump(config_data, fout)


    def kmeans_clustering(self, marker_data, marker):
        marker_scaled_data = pd.DataFrame(preprocessing.scale(marker_data), columns=marker_data.columns)
        n_comp = len(marker_scaled_data.columns)

        if len(marker_scaled_data) < n_comp:
            n_comp = len(marker_scaled_data)

        first_pca = decomposition.PCA(
                    n_components=n_comp)

        marker_df_all_pcs = first_pca.fit_transform(marker_scaled_data)

        var_cum_explained = first_pca.explained_variance_ratio_.cumsum()
        for n, pc in enumerate(var_cum_explained):
            if pc >= self.var_expl:
                n_pcs = n + 1
                break

        #print(f'Number of PCs selected to retain {self.var_expl} of variance: {n_pcs}')
        reduced_pca = decomposition.PCA(
                            n_components = n_pcs)


        marker_pca = reduced_pca.fit_transform(marker_scaled_data)
        start = time()
        kmeans = MiniBatchKMeans(n_clusters=self.k_clusters, random_state=0, max_iter=300, n_init=3).fit(marker_pca)
        end = time()

        marker_data[f'{marker}_tier'] = [float(k) for k in kmeans.labels_.tolist()]

        #clusters_column = marker_data[[f'{marker}_tier']]

        return marker_data


    def get_marker_data(self, marker, raw_df):
        """
        Filter out non-informative variables
        """
        marker_raw_df = raw_df.loc[:, raw_df.columns.str.contains(marker)]
        marker_raw_df = marker_raw_df.loc[:, ~marker_raw_df.columns.str.contains('y.coor')]
        marker_raw_df = marker_raw_df.loc[:, ~marker_raw_df.columns.str.contains('x.coor')]
        marker_raw_df = marker_raw_df.loc[:, ~marker_raw_df.columns.str.contains('cell')]
        marker_raw_df = marker_raw_df.loc[:, ~marker_raw_df.columns.str.contains('peri')]
        marker_raw_df = marker_raw_df.loc[:, ~marker_raw_df.columns.str.contains('area')]
        marker_raw_df = marker_raw_df.loc[:, ~marker_raw_df.columns.str.contains('circularity')]
        marker_raw_df = marker_raw_df.loc[:, ~marker_raw_df.columns.str.contains('major')]
        marker_raw_df = marker_raw_df.loc[:, ~marker_raw_df.columns.str.contains('minor')]
        marker_raw_df = marker_raw_df.loc[:, ~marker_raw_df.columns.str.contains('peaks')]
        marker_raw_df = marker_raw_df.loc[:, ~marker_raw_df.columns.str.contains('IQR')]
        marker_raw_df = marker_raw_df.loc[:, ~marker_raw_df.columns.str.contains('hema')]
        marker_raw_df = marker_raw_df.loc[:, ~marker_raw_df.columns.str.contains('dapi')]

        if self.technology == 'if':
            dna_marker = self.marker_name_list[self.dna_marker_channel]
            if dna_marker != marker:
                marker_raw_df = marker_raw_df.loc[:, ~marker_raw_df.columns.str.contains(dna_marker)]

        return marker_raw_df


    def clustering(self, marker, raw_df):

        marker_raw_data = self.get_marker_data(marker, raw_df)

        marker_clusters = self.kmeans_clustering(marker_raw_data, marker)

        medians = self.calculate_perc_medians(marker_clusters, marker)

        clusters = marker_clusters[[f'{marker}_tier']]

        data = {marker: medians}

        return clusters, data
    

    def tma_clustering(self, marker, raw_df):
        pd.options.mode.chained_assignment = None
 
        cores_subsets = []
        medians = {}
        for core, tiles in self.cores_map.items():
            medians[core] = {}
            subset = raw_df.loc[raw_df[f'Tile index'].isin(tiles)]
            marker_data = self.get_marker_data(marker, subset)

            try:
                marker_clusters = self.kmeans_clustering(marker_data, marker)
                clusters = marker_clusters[[f'{marker}_tier']]

                aux = self.calculate_perc_medians(marker_clusters, marker)
                medians[core].update(aux)

                cores_subsets.append(clusters)
                
            except:
                print(f'[X] Error | Tile #{tile}')
                print(traceback.format_exc())

        clusters = pd.concat(cores_subsets)
        clusters.sort_index(inplace=True)

        data = {marker: medians}

        return clusters, data


    def start(self, stain_index):

        marker = self.marker_name_list[stain_index]

        raw_df, final_marker_info = self._read_rawcellmetrics(marker)
        self._check_for_cores()
        if self.cores_labels:
            final_marker_info['Core Label'] = final_marker_info['Tile index'].map(self.tile_map)
            clusters, medians = self.tma_clustering(marker, raw_df)

        else:
            clusters, medians = self.clustering(marker, raw_df)

        final_marker_info = pd.merge(final_marker_info, clusters, on='Cell index')

        save_path = os.path.join(self.tmp_directory, f'{marker}_finalcellinfo.csv')

        final_marker_info.to_csv(save_path)

        return medians


    def project_clustering(self, stain_index):

        marker = self.markers[0][stain_index]

        raw_df, final_marker_info = self.create_project_rawcellmetrics(marker)

        marker_raw_data = self.get_marker_data(marker, raw_df)

        marker_clusters = self.kmeans_clustering(marker_raw_data, marker)

        clusters = marker_clusters[[f'{marker}_tier']]

        final_marker_info = pd.merge(final_marker_info, clusters, on='Cell index')        

        # Lets add sample_name column in marker_raw_data
        marker_sample_labeled = marker_raw_data.copy()
        marker_sample_labeled.loc[:, 'sample_name'] = final_marker_info['sample_name']

        medians = {}
        for sample, tmp_directory in zip(self.sample_names, self.tmp_directories):
            # Calculating medians per sample
            medians[sample] = {}
            marker_clusters_sample = marker_sample_labeled.loc[marker_sample_labeled['sample_name'] == sample]

            medians_per_sample = self.calculate_perc_medians(marker_clusters_sample, marker)

            medians[sample][marker] = medians_per_sample

            # Saving finalcellinfo per sample
            save_path = os.path.join(tmp_directory, f'{marker}_finalcellinfo.csv')

            sample_final_marker_info = final_marker_info.loc[final_marker_info['sample_name'] == sample]
            sample_final_marker_info.reset_index(inplace=True, drop=True)
            sample_final_marker_info.index.rename('Cell index', inplace=True)

            sample_final_marker_info = sample_final_marker_info.drop('sample_name', axis=1)
            sample_final_marker_info.to_csv(save_path)

        
        return medians    
    

    def regroup(self, *args):
        root_cluster = int(args[0])
        marker = args[1]
        finalcellinfo_file = args[2]

        if len(args) == 4:
            core = args[3]
            self.core = core

        finalcellinfo_df = pd.read_csv(finalcellinfo_file, index_col=['Cell index'])
        # Assign root_cluster value to all cells belonging to the subcluster
        if self.cores_labels:
            finalcellinfo_df.loc[(finalcellinfo_df[f'{marker}_tier'] >= root_cluster) &
                              (finalcellinfo_df[f'{marker}_tier'] < (root_cluster + 1)) &
                              (finalcellinfo_df['TMA Core'] == core), [f'{marker}_tier']] = root_cluster

        else:
            finalcellinfo_df.loc[(finalcellinfo_df[f'{marker}_tier'] >= root_cluster) &
                              (finalcellinfo_df[f'{marker}_tier'] < (root_cluster + 1)), [f'{marker}_tier']] = root_cluster


        # Saving new finalcellinfo with root cluster "regrouped"
        finalcellinfo_df.to_csv(finalcellinfo_file)

        raw_df, _ = self._read_rawcellmetrics(marker)
        marker_clusters = raw_df.copy()
        marker_clusters.loc[:, f'{marker}_tier'] = finalcellinfo_df[f'{marker}_tier']

        medians = self.calculate_perc_medians(marker_clusters, marker)

        self._update_perc_medians_regroup(medians, root_cluster, marker)


    def reclustering_project(self, *args):
        target = int(args[0])
        marker = args[1]
        finalcellinfo_combined_df = args[2]
        samples_map = args[3]

        finalcellinfo_subset = finalcellinfo_combined_df.loc[finalcellinfo_combined_df[f'{marker}_tier'] == target]
        
        raw_df, _ = self.create_project_rawcellmetrics(marker)

        # Merge the datasets to subset only the targeted cells
        target_cells = raw_df.iloc[finalcellinfo_subset.index]
        marker_raw_data = self.get_marker_data(marker, target_cells)
        marker_clusters = self.kmeans_clustering(marker_raw_data, marker)

        marker_clusters[f'{marker}_tier'] = marker_clusters[f'{marker}_tier'].apply(lambda x: f'{target}.{int(x)}')
        marker_sample_labeled = marker_clusters.copy()
        marker_sample_labeled.loc[:, 'sample_name'] = finalcellinfo_combined_df['sample_name']

        # Join and update finallcellinfo and saves the old one (maybe *bak)
        subclusters = marker_clusters[[f'{marker}_tier']]

        final_marker_info = subclusters.combine_first(finalcellinfo_combined_df)

        sample_names = finalcellinfo_combined_df['sample_name'].unique().tolist()
        for sample_name in sample_names:
            sample_finalcellinfo_df = final_marker_info.loc[final_marker_info['sample_name'] == sample_name] 
            marker_sample_data = marker_sample_labeled.loc[marker_sample_labeled['sample_name'] == sample_name]

            sample_finalcellinfo_df.reset_index(inplace=True, drop=True)
            sample_finalcellinfo_df.index.rename('Cell index', inplace=True)

            final_df = sample_finalcellinfo_df.drop('sample_name', axis=1)

            final_df.to_csv(samples_map[sample_name][0])

            self.config_file = samples_map[sample_name][1]
            medians = self.calculate_perc_medians(marker_sample_data, marker)

            self.update_perc_medians(medians, target, marker)


    def reclustering(self, *args):
        target = int(args[0])
        marker = args[1]
        finalcellinfo_file = args[2]

        if len(args) == 4:
            core = args[3]
            self.core = core

        # Extract the cells belongging to the target cluster
        finalcellinfo = pd.read_csv(finalcellinfo_file, index_col=['Cell index'])
        finalcellinfo_subset = finalcellinfo.loc[finalcellinfo[f'{marker}_tier'] == target]

        if self.cores_labels:
            finalcellinfo_subset = finalcellinfo.loc[(finalcellinfo[f'{marker}_tier'] == target) & (finalcellinfo['TMA Core'] == core)]

        # Open the rawcellmetrics and extract the metadata associated with the cells
        raw_df, _ = self._read_rawcellmetrics(marker)

        # Merge the datasets to subset only the targeted cells
        target_cells = raw_df.iloc[finalcellinfo_subset.index]
        marker_raw_data = self.get_marker_data(marker, target_cells)
        marker_clusters = self.kmeans_clustering(marker_raw_data, marker)
        
        # Add prefix to subclusters "target_cluster.new_number"
        pd.options.mode.chained_assignment = None
        marker_clusters[f'{marker}_tier'] = marker_clusters[f'{marker}_tier'].apply(lambda x: f'{target}.{int(x)}')

        # Join and update finallcellinfo and saves the old one (maybe *bak)
        subclusters = marker_clusters[[f'{marker}_tier']]

        final_marker_info = subclusters.combine_first(finalcellinfo)
        final_marker_info.to_csv(finalcellinfo_file)

        medians = self.calculate_perc_medians(marker_clusters, marker)
        
        # Write new values to config.yaml
        self.update_perc_medians(medians, target, marker)


    def __getattr__(self, name):
        return self.parent_class.__getattribute__(name)

