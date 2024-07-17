
import os
import gc
import yaml
import umap
import numpy as np
import pandas as pd


from tqdm import tqdm
import seaborn as sns
from functools import partial
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


from sklearn.preprocessing import StandardScaler

class GMM:
    """
    
    """
    def _init__(self):
        pass


    def PCA_UMAP_GMM_learning(self, df_input, marker, sample, counter, output_path):

        marker_df = df_input.iloc[:, (df_input.columns.get_level_values(0)==marker)]
        marker_df_shape = df_input.iloc[:, (df_input.columns.get_level_values(0) == 'Shape')]
        marker_df = pd.concat([marker_df,marker_df_shape],axis=1)
        marker_df = marker_df.dropna(axis=1, how='all')
        neighbors_size = int(len(marker_df.index) / 45) 
        
        if neighbors_size < 1000:
                neighbors_size = 1000
        
        if neighbors_size > 1500:
                neighbors_size = 1500

        reducer = umap.UMAP(densmap=False, 
                                    min_dist = 0.1,
                                    n_neighbors = 1000,       ### neighbors_size
                                    n_components = 3,
                                    metric='euclidean')
                        
        pca = decomposition.PCA(
                    n_components = int(0.9*len(marker_df.columns)), 
                    svd_solver = 'arpack')
        marker_df_pca_coords = pca.fit(marker_df).transform(marker_df)
        marker_umap_input = reducer.fit(marker_df_pca_coords)  

        #GMM APPROACH
        gmm = GaussianMixture(n_components = 60,
                                n_init=200,
                                max_iter=500,
                                covariance_type='full',
                                init_params='k-means++').fit(marker_umap_input.embedding_) 
        gmm_labels = gmm.predict(marker_umap_input.embedding_)
        marker_df['gmm'] = gmm_labels.tolist() 

        ##DBSCAN APPROACH:
        #    db = DBSCAN(eps=0.12, min_samples=5).fit(marker_umap_input.embedding_)
        #    db_labels = db.labels_ 
        #    marker_df['gmm'] = db_labels.tolist()

        sns.set(font_scale=6)
        clustermap = sns.clustermap(marker_df.groupby('gmm').mean(),
                        metric = 'euclidean',
                        method = 'ward',   # weighted, complete or centroid or average
                        z_score = None,
                        standard_scale = None,
                        row_cluster = True,
                        col_cluster = True,
                        yticklabels = True,
                        xticklabels = True,
                        cmap="coolwarm",
                        figsize=(70,50),
                        dendrogram_ratio=(0.1),
                        colors_ratio=(0.02),
                        cbar_kws={'label': 'Z-score, mean', 'orientation': 'vertical'},
                        center=0, 
                        linewidths=0.01
                        )
        clustermap.ax_col_dendrogram.set_visible(False)
        clustermap.ax_row_dendrogram.set_visible(False)
        clustermap.fig.suptitle('Z-scores per Gaussian mixed models '+marker)
        clustermap.savefig(output_path+'/'+sample+'_'+marker+'_'+str(counter)+'.CLUSTERMAP.pdf',
                facecolor = 'white',
                edgecolor='white', 
                dpi=200)
        
        return marker_df


    def classification_worker(self, marker, df_input, input_sample, counter, output_path):
        
        df_output = pd.DataFrame(index=df_input.index)
        df_marker_gmm  = self.PCA_UMAP_GMM_learning(df_input,
                                                        marker,
                                                        input_sample,
                                                        counter,
                                                        output_path)
        df_marker_gmm_means = df_marker_gmm.groupby('gmm').mean()
        df_marker_gmm_stdev = df_marker_gmm.groupby('gmm').std()
        df_marker_gmm_stdev.columns = pd.MultiIndex.from_tuples([(x[0], 'std_'+ x[1], x[2]) for x in df_marker_gmm_stdev.columns])
        df_marker_gmm_labels = df_marker_gmm_means.index.to_list()
        df_marker_gmm_labels = [str(marker) + '_' + str(s) + '_'+ str(counter) for s in df_marker_gmm_labels]
        df_marker_gmm_means.to_csv(output_path + '/' + input_sample + '_' + marker +'_' + str(counter) + '.STATS.csv', sep=',', header=True, index=True, compression=None)
        header = marker + '_gmm'
        df_output[header] = df_marker_gmm['gmm'].astype(str) + '_' + str(counter)

        return (df_output, df_marker_gmm_means, df_marker_gmm_labels)



    def classification(self): 
        #read rawcellmetrics.csv file per sample 
        cellres_sample_outputpath = os.path.join(self.marco_output_path, self.sample_name + '_rawcellmetrics.csv')
        df = pd.read_csv(cellres_sample_outputpath, sep=',', header=0, index_col=[0])
        df.reset_index(inplace=True, drop=True)
        df.index.rename('Cell index', inplace=True)
        df.set_index(['Tile index', 'ROI-relative cell y_coord (pixels)', 'ROI-relative cell x_coord (pixels)'], inplace=True, append=True)

        plot_output_path = os.path.join(self.marco_output_path, 'classification_plots')
        os.makedirs(plot_output_path)

        #reformat dataframe
        df.columns = [x.replace(' ','.') for x in df.columns]
        df.columns = [x.replace(self.sample_name,'') for x in df.columns]
        df.columns = [x.replace('_','') for x in df.columns]
        df.columns = [x.strip('.') for x in df.columns]
        df.columns = [x.replace('-','') for x in df.columns]
        df.columns = [x.replace('.(pixels)','') for x in df.columns]
        df.columns = [x.replace('.(microns)','') for x in df.columns]
        df.columns = [x.replace('.nuc.hema.','.hema.AEC.') for x in df.columns]
        df.index.names = [x.replace(' ','.') for x in df.index.names]
        df.index.names = [x.replace('.index','') for x in df.index.names]
        df.index.names = [x.replace('.(pixels)','') for x in df.index.names]
        df.index.names = [x.replace('ROI-relative.cell.','') for x in df.index.names]
        df = df.loc[:,~df.columns.str.contains('ycoor')]
        df = df.loc[:,~df.columns.str.contains('xcoor')]
        df = df.loc[:,~df.columns.str.contains('cell')]
        df = df.loc[:,~df.columns.str.contains('cyto')]
        df = df.loc[:,~df.columns.str.contains('.circularity')]
        df = df.loc[:,~df.columns.str.contains('.p1')]
        df = df.loc[:,~df.columns.str.contains('.p3')]
        df = df.loc[:,~df.columns.str.contains('\.p4')]
        df = df.loc[:,~df.columns.str.contains('.p6')]
        df = df.loc[:,~df.columns.str.contains('.p7')]
        df = df.loc[:,~df.columns.str.contains('.p9')]
        df = df.loc[:,~df.columns.str.endswith('.min')]
        df = df.loc[:,~df.columns.str.endswith('.max')]
        df = df.loc[:,~df.columns.str.contains('.mean')]

        #### uncomment if numpeaks and hema should NOT be used:
        # df = df.loc[:,~df.columns.str.endswith('.numpeaks')]
        # df = df.loc[:,~df.columns.str.contains('.hema.')]

        #reshape df
        df.rename(columns = {"Nuc.peri": "Shape.nucleus.peri", 
                             "Nuc.area": "Shape.nucleus.surf", 
                            "Nuc.major.axis": "Shape.nucleus.major",
                            "Nuc.minor.axis": "Shape.nucleus.minor"}, inplace = True)
        df_distr = df.loc[:,df.columns.str.contains('.numpeaks|.SD|.IQR')]
        df_shape = df.loc[:,df.columns.str.contains('Shape.')]
        df = df.loc[:,~df.columns.str.contains('Shape.')]
        df = df.loc[:,~df.columns.str.contains('.numpeaks')]
        df = df.loc[:,~df.columns.str.contains('.SD')]
        df = df.loc[:,~df.columns.str.contains('.IQR')]
        df_shape.columns = [ tuple(x.split('.')) for x in df_shape.columns ] 
        df.columns = [ tuple(x.split('.')) for x in df.columns ] 
        df.columns = [ tuple(x for x in y if x != 'AEC') for y in df.columns ]
        df_distr.columns = [ tuple(x.split('.')) for x in df_distr.columns ] 
        df_distr.columns = [ tuple(x for x in y if x != 'AEC') for y in df_distr.columns ]
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['stain','area','type'])
        df_shape.columns = pd.MultiIndex.from_tuples(df_shape.columns, names=['stain','area','type'])
        df_distr.columns = pd.MultiIndex.from_tuples(df_distr.columns, names=['stain','area','type'])
        df_diff = pd.DataFrame()
        df_diff.columns = pd.MultiIndex(levels=[[],[],[]], codes=[[],[],[]], names=['stain','area','type'])
        for marker in df.columns.get_level_values(0):
            marker_df = df.iloc[:, (df.columns.get_level_values(0) == marker)]
            for quantile in marker_df.columns.get_level_values(2):
                df_diff[marker, "diff", quantile] = df[marker,"nuc",quantile] - df[marker,"hema",quantile]
        df = df.iloc[:,~(df.columns.get_level_values(1)=='hema')]

        #scale df
        std_scaler = StandardScaler()
        df = df.stack(level=['area'])
        df_aec_scaled = std_scaler.fit_transform(df)
        df_shape_scaled = std_scaler.fit_transform(df_shape)
        df_distr_scaled = std_scaler.fit_transform(df_distr)
        df_diff_scaled = std_scaler.fit_transform(df_diff)
        df_aec_scaled = pd.DataFrame(df_aec_scaled, 
                columns = df.columns, 
                index = df.index)
        df_shape_scaled = pd.DataFrame(df_shape_scaled, 
                columns = df_shape.columns, 
                index = df_shape.index)
        df_distr_scaled = pd.DataFrame(df_distr_scaled, 
                columns = df_distr.columns, 
                index = df_distr.index)
        df_diff_scaled = pd.DataFrame(df_diff_scaled, 
                columns = df_diff.columns, 
                index = df_diff.index)
        df = df.unstack(level=['area'])
        df_aec_scaled = df_aec_scaled.unstack(level='area')
        df_aec_scaled.columns = df_aec_scaled.columns.reorder_levels(["stain", "area","type"])
        df_shape_scaled.columns = df_shape_scaled.columns.reorder_levels(["stain", "area","type"])
        df_distr_scaled.columns = df_distr_scaled.columns.reorder_levels(["stain", "area","type"])
        df_diff_scaled.columns = df_diff_scaled.columns.reorder_levels(["stain", "area","type"])
        del df
        del df_shape
        del df_distr
        del df_diff
        gc.collect()

        df_aec_scaled = pd.concat([df_aec_scaled,df_shape_scaled], axis=1)
        df_aec_scaled = pd.concat([df_aec_scaled,df_distr_scaled], axis=1)
        df_aec_scaled = pd.concat([df_aec_scaled,df_diff_scaled], axis=1)

        #Determine num of subsamples
        filesize = len(df_aec_scaled.index)
        splits = filesize // 45000
        if splits == 0:
            splits = 1
        df_shuffled = df_aec_scaled.sample(frac=1)
        df_shuffled = df_shuffled.sample(frac=1)
        df_splits = np.array_split(df_shuffled, splits)
        del df_shuffled
        del df_aec_scaled
        gc.collect()
        df = pd.DataFrame()
        df_shuffled = pd.DataFrame()
        df_aec_scaled = pd.DataFrame()

        #learning per subsample

        output = []
        for count, idf in enumerate(tqdm(df_splits)):
            classification_worker_partial=partial(self.classification_worker,
                                                  df_input = idf,
                                                  input_sample = self.sample_name,
                                                  counter = count,
                                                  output_path = plot_output_path)
            markers = idf.columns.levels[0].to_list()
            markers.remove('Shape')
            #backend paramter --> can use 'multithreaded' or 'loky'
            dfs = Parallel(n_jobs = 4, backend='loky', temp_folder=None)(delayed(classification_worker_partial)(marker) for marker in markers)
            output.extend(dfs)

        #determine # tiers per marker 
        marker_centroids_list = []
        annotated_markers = pd.DataFrame()
        number_of_classes = []
        for marker in markers:
            marker_output = [table for table in output if marker in table[1]]
            marker_centroids = map(lambda x: x[0], marker_output)
            marker_stats = map(lambda x: x[1], marker_output)
            marker_labels = map(lambda x: x[2], marker_output)
            marker_centroids = pd.concat(marker_centroids,axis=0, ignore_index=False)
            marker_stats = pd.concat(marker_stats,axis=0, ignore_index=True)
            marker_labels = [item for sublist in marker_labels for item in sublist]
            silhouette_avg = []
            calinski_harabasz_avg=[]
            inertia = []
            max_clusters = 50
            for cluster in range(5,max_clusters):
                y_pred = KMeans(n_clusters=cluster,
                        n_init=1000,
                        init = 'k-means++',
                        max_iter=1000,
                       algorithm='elkan').fit(marker_stats)
                silhouette_avg.append(silhouette_score(marker_stats, y_pred.labels_))
                inertia.append(y_pred.inertia_)
            index_max = max(range(len(silhouette_avg)), key=silhouette_avg.__getitem__) + 5
            index_elbow = KneeLocator(range(5,max_clusters), inertia, curve='convex', direction='decreasing').knee
            cluster = min(index_max,index_elbow)
            if cluster < 15 or cluster > 30:
                cluster = 15

            #export figure per marker
            fig, ax1 = plt.subplots(figsize=(10,10),dpi=100)
            ax2 = ax1.twinx()
            ax1.plot(range(5,max_clusters),silhouette_avg, 'g-')
            ax2.plot(range(5,max_clusters),inertia, 'b-')
            ax2.axvline(x=cluster, color='b', label='axvline - full height')
            ax1.set_xlabel('Values of K')
            ax1.set_ylabel('Silhouette score', color='g')
            ax2.set_ylabel('Sum_of_squared_distances', color='b')
            plt.title('Silhouette and Elbow analysis For Optimal k for marker : '+str(marker))
            plt.savefig(plot_output_path + '/scores_Kmeans_' + str(marker)+'.png')
            plt.close()

            #finalize data per marker
            y_pred = KMeans(n_clusters=cluster,
                        n_init=1000,
                        init = 'k-means++',
                        max_iter=1000,
                       algorithm='elkan').fit_predict(marker_stats).tolist()
            list_zip = zip(marker_labels, y_pred)
            list_zip = list(list_zip)
            marker_centroids[marker] = 0
            for i in list_zip:
                gmm=i[0].split('_')[1]
                counter=i[0].split('_')[2]
                flag=gmm+'_'+counter
                marker_centroids[marker] = np.where(marker_centroids[marker+'_gmm'] == flag,i[1], marker_centroids[marker])
            marker_centroids_list.append(marker_centroids)
            number_of_classes.append(cluster)

        #read and initialize dataframes
        rawcellmetrics_df = pd.read_csv(cellres_sample_outputpath, sep=',', header=0, index_col=[0])
        rawcellmetrics_df.reset_index(inplace=True, drop=True)
        rawcellmetrics_df.index.rename('Cell index', inplace=True)

        # update rawcellmetrics file with increasing order indexes
        rawcellmetrics_df.to_csv(cellres_sample_outputpath)

        annotated_markers = pd.concat(marker_centroids_list,axis=1,ignore_index=False)
        annotated_markers = annotated_markers.sort_values(by=['Cell'])
        annotated_markers = annotated_markers.rename_axis(index={'Cell': 'Cell index', 'Tile': 'Tile index', 'y_coord':'Relative y_coord','x_coord':'Relative x_coord'})
        annotated_markers = annotated_markers.reset_index()
        annotated_markers.to_csv(os.path.join(self.marco_output_path, self.sample_name + '_checkpoint.csv'), index=False)

        df_classification_key = pd.DataFrame()

        #sort gmms
        marker_name_list_df = []
        presorted_to_sorted_dict_allmarkers = []
        for marker_name_to_eval_raw in self.marker_name_list:

            #sort gmms by AEC median of medians' intensities
            marker_name_to_eval_df = marker_name_to_eval_raw.replace(' ','')
            marker_name_to_eval_df = marker_name_to_eval_df.replace('_','')
            marker_name_to_eval_df = marker_name_to_eval_df.replace('-','')
            marker_name_to_eval_df = marker_name_to_eval_df.replace('.','')
            marker_name_list_df.append(marker_name_to_eval_df)
            AEC_median_dict_for_marker = {}
            total_GMMs_for_marker = np.max(annotated_markers[marker_name_to_eval_df])
            for GMM_index_to_eval in range(total_GMMs_for_marker+1): #add fencepost index
                current_indices = annotated_markers[annotated_markers[marker_name_to_eval_df]==GMM_index_to_eval].index.get_level_values(0)
                median_of_medians_AEC = np.median(rawcellmetrics_df.loc[current_indices, marker_name_to_eval_raw + ' nuc AEC p5'])
                AEC_median_dict_for_marker[GMM_index_to_eval] = median_of_medians_AEC    
            AEC_median_dict_for_marker_sorted = {k: v for k, v in sorted(AEC_median_dict_for_marker.items(), reverse=True, key=lambda item: item[1])}
            presorted_to_sorted_dict_permarker = {}
            for index, key in enumerate(AEC_median_dict_for_marker_sorted):
                presorted_to_sorted_dict_permarker[key] = index
            presorted_to_sorted_dict_allmarkers.append(presorted_to_sorted_dict_permarker)

            #create key df for original, presorted, and sorted gmms
            og_gmm_list = []
            presorted_gmm_list = []
            sorted_gmm_list = []
            for og_gmm in sorted(np.unique(annotated_markers[marker_name_to_eval_df + '_gmm'])):
                presorted_gmm = list(annotated_markers.loc[annotated_markers[marker_name_to_eval_df + '_gmm'] == og_gmm ,marker_name_to_eval_df])[0]
                sorted_gmm = list(AEC_median_dict_for_marker_sorted.keys()).index(presorted_gmm)
                og_gmm_list.append(og_gmm)
                presorted_gmm_list.append(presorted_gmm)
                sorted_gmm_list.append(sorted_gmm)
            df_classification_key[marker_name_to_eval_raw + ' Original GMM'] = og_gmm_list 
            df_classification_key[marker_name_to_eval_raw + ' Presorted GMM'] = presorted_gmm_list 
            df_classification_key[marker_name_to_eval_raw + ' Sorted GMM'] = sorted_gmm_list 

        #export classification df key
        df_classification_key.to_csv((plot_output_path + '/' + self.sample_name + '_classificationkey.csv'), index=False)

        #init and populate finalcellinfo.csv
        finalcellinfo_df = annotated_markers.loc[:,['Cell index','Tile index', 'Relative y_coord', 'Relative x_coord']]
        finalcellinfo_df['Tissue annotation'] = np.NaN
        for marker_index, marker_name_original in enumerate(self.marker_name_list):
            marker_name_df = marker_name_list_df[marker_index]
            finalcellinfo_df[marker_name_original + '_y-coord (pixels)'] = rawcellmetrics_df.loc[:, marker_name_original + ' y_coor (pixels)']
            finalcellinfo_df[marker_name_original + '_x-coord (pixels)'] = rawcellmetrics_df.loc[:, marker_name_original + ' x_coor (pixels)']    
            #update gmm indicies to sorted gmms
            sorted_annotated_markers = annotated_markers[marker_name_df].copy()
            for cell_index, presorted_value in enumerate(annotated_markers[marker_name_df]):
                sorted_value = presorted_to_sorted_dict_allmarkers[marker_index][presorted_value]
                sorted_annotated_markers[cell_index] = sorted_value        
            finalcellinfo_df[marker_name_original + '_tier'] = sorted_annotated_markers
            finalcellinfo_df[marker_name_original + '_expression'] = np.NaN

        #export finalcellinfo.csv
        finalcellinfo_df.to_csv(os.path.join(self.marco_output_path, self.sample_name + '_finalcellinfo.csv'), index=False)

        #output status
        print('Classification completed for ' + self.sample_name + '.')

