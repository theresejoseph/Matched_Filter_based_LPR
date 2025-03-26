import os
import json
import logging
import pickle
import time
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

import cupy as cp
from scipy import stats, signal, interpolate
from sklearn.preprocessing import StandardScaler
import skimage

# Import custom helper functions (assuming they exist in helperFunctions.py)
from helper import (
    LiDAR_PlaceRec, 
    find_closest_point
)

class PlaceRecognitionVisualizer:
    """
    A comprehensive visualization tool for LiDAR-based place recognition
    """
    def __init__(
        self, 
        config_path: str = './scripts/config.json', 
        dataset_name: str = 'WildPlaces_Karawatha',
        ref_num: int = 11, 
        query_num: int = 2,
        err_tolerance: float = 3.0
    ):
        """
        Initialize the place recognition visualizer
        
        Args:
            config_path (str): Path to configuration JSON file
            dataset_name (str): Name of the dataset to analyze
            ref_num (int): Reference dataset number
            query_num (int): Query dataset number
            err_tolerance (float): Error tolerance for matching
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Load configuration
        with open(config_path) as f:
            self.config = json.load(f).get(dataset_name, {})
        
        # Set parameters
        self.dataset_name = dataset_name
        self.ref_num = ref_num
        self.query_num = query_num
        # self.err_tolerance = err_tolerance

        # Initialize dataset-specific parameters
        self._initialize_parameters()
        
        # Initialize LiDAR Place Recognition
        self._initialize_lidar_place_recognition()

    def _initialize_parameters(self):
        """Extract and set parameters from configuration"""
        params = self.config.get('parameters', {})
        
        # Extract key parameters
        self.ref_incr = self.config.get('refIncr', 1)
        self.query_incr = self.config.get('queryIncr', 1)
        self.err_tolerance = self.config.get('errTolerance',1)
        
        # Scan and mapping parameters
        self.scan_dim_x = params.get('scanDimX', 120)
        self.scan_dim_y = params.get('scanDimY', 120)
        self.map_res = params.get('mapRes', 0.3)
        self.block_size = params.get('blkSize', 2)
        
        # Radius and filtering parameters 
        self.ref_radius = params.get('refRadius', 2)
        self.query_radius = params.get('queryRadius', 2)
        self.num_matches = params.get('topN', 2)
        
        # Background and thresholding
        self.background_weight = params.get('unoccWeight', -0.15)
        self.ref_thresh = params.get('refThresh', 2)
        self.query_thresh = params.get('queryThresh', 2)

    def _initialize_lidar_place_recognition(self):
        """
        Initialize LiDAR Place Recognition with dataset-specific configurations
        """
        try:
            # Handle different dataset types
            if self.dataset_name.startswith('WildPlaces'):
                # # Load evaluation info for WildPlaces datasets
                with open(self.config.get('evalInfoFile'), 'rb') as f:
                    evalInfo = pickle.load(f)[self.query_num-1]
                
                all_files = os.listdir(self.config.get('details', {}).get(str(self.query_num), [])[0])
                frames_query = []
                for k in range(len(evalInfo)):
                    try:
                        if evalInfo[k][self.ref_num-1] != []:
                            index = all_files.index(evalInfo[k]['query'].split('/')[-1])
                            frames_query.append(index)
                    except ValueError:
                        pass  
                

                # Reference grid file path
                ref_grid_name = f"_refGrid_dim{self.scan_dim_x}_mapRes{self.map_res}_refIncr{self.ref_incr}_refRad{self.ref_radius}_rThresh{self.ref_thresh}_nptsMult{self.num_matches}_AvgDownsamp.npy"
                ref_grid_path = f'./data_processed_files/{self.dataset_name}{self.ref_num}'+ ref_grid_name
                # Initialize LiDAR Place Recognition
                self.lpr = LiDAR_PlaceRec(
                    self.dataset_name, 
                    self.config, 
                    self.ref_num, 
                    self.query_num, 
                    framesQuery=frames_query,
                    refGridFilePath=ref_grid_path, 
                    n= self.num_matches,
                    background=self.background_weight
                )
            else:
                # For other datasets like NCLT or OxfordRadar
                frames_query = list(np.load(
                    f'./results/LPR_{self.dataset_name}/FramesFiles/Q:{self.query_num}_framesQuery_qInc:{self.query_incr}_qRad:{self.query_radius}_everyN.npy'
                ))
                frames_ref=list(np.load(
                    f'./results/LPR_{self.dataset_name}/FramesFiles/R:{self.ref_num}_framesRef_rInc:{self.ref_incr}_rRad:{self.ref_radius}_Napart.npy'
                ))

                ref_grid_name = f"_refGrid_dim{self.scan_dim_x}_mapRes{self.map_res}_refIncr{self.ref_incr}_refRad{self.ref_radius}_rThresh{self.ref_thresh}_nptsMult{self.num_matches}_AvgDownsamp.npy"
                ref_grid_path = f'./data_processed_files/{self.dataset_name}{self.ref_num}'+ ref_grid_name
            
                # Initialize LiDAR Place Recognition
                self.lpr = LiDAR_PlaceRec(
                    self.dataset_name, 
                    self.config, 
                    self.ref_num, 
                    self.query_num, 
                    framesQuery=frames_query, 
                    framesRef= frames_ref,
                    refGridFilePath=ref_grid_path, 
                    n= self.num_matches,
                    background=self.background_weight
                )
            print(f"Total ref frames: {self.lpr.numRefScans()}")
            # Extract scan centers and poses
            self.center_xs, self.center_ys = zip(*self.lpr.scanCenter())
            self.xq, self.yq, _, _, self.yaw_q = self.lpr.scanPoses('QUERY')
            self.xr, self.yr, _, _, self.yaw_r = self.lpr.scanPoses('REF')

            # Find closest points and minimum distances
            self.close_ids = [
                find_closest_point(zip(self.xr, self.yr), self.xq[i], self.yq[i])[2] 
                for i in range(len(self.xq))
            ]
            self.min_dist = [
                find_closest_point(zip(self.xr, self.yr), self.xq[i], self.yq[i])[3] 
                for i in range(len(self.xq))
            ]

            # Filter query frames based on distance tolerance
            self.filtered_query_ids = [ i
                for i in range(len(self.lpr.framesQuery)) 
                if self.min_dist[i] < self.err_tolerance
            ]

            self.logger.info(
                f'{len(self.lpr.framesQuery) - len(self.filtered_query_ids)} '
                f'queries removed from {len(self.lpr.framesQuery)} '
                f'for being outside {self.err_tolerance}m from ref'
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize LiDAR Place Recognition: {e}")
            raise

    def create_animation(self, start_frame: int = 0):
        """
        Create an animation of place recognition results
        
        Args:
            start_frame (int): Starting frame for animation
        """
        # Create figure with specific layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))
        plt.tight_layout(pad=3)
        # plt.subplots_adjust(top=0.9, wspace=0.3, hspace=0)  # Increase top margin
    

        # Tracking variables
        save_good_count, total_matches = 0, 0

        def update_save_good_bad(i):
            nonlocal save_good_count, total_matches
            i = self.filtered_query_ids[i]

            # Perform place recognition update
            rot_inc = 10
            query_scan, ref_scan, ref_shift_match_scan, ref_closest, idx, \
            max_x, max_y, match_x_delta, match_y_delta, shifted_x, shifted_y, \
            rotations, alignment, dist, align_vals = self.lpr.update(
                i, rot_inc, self.xr, self.yr, self.yaw_r, 
                self.xq, self.yq, self.yaw_q, 
                self.center_xs, self.center_ys, 
                self.close_ids, 
                returnArray=True
            )

            # Clear previous plots
            ax1.clear(), ax2.clear()
            
            # Overlay plots with better transparency and color schemes
            overlay_params = {
                'alpha': 0.6,
                'interpolation': 'nearest'
            }
            
            # Plot query and reference scans with side-by-side comparison
            ax1.imshow(query_scan, cmap='Greys', **overlay_params)
            ax1.imshow(ref_closest, cmap='Greens', **overlay_params)
            ax1.set_title('Query and Closest Reference')
            
            ax2.imshow(query_scan, cmap='Greys', **overlay_params)
            ax2.imshow(ref_shift_match_scan, cmap='Blues', **overlay_params)
            
            # Determine match quality and apply color-coded border
            match_quality = dist <= self.err_tolerance
            border_color = 'green' if match_quality else 'red'
            border_width = 3
            
            for ax in [ax1, ax2]:
                for spine in ax.spines.values():
                    spine.set_color(border_color)
                    spine.set_linewidth(border_width)
                ax.set_xticks([])
                ax.set_yticks([])
            
            ax2.set_title(f'Query and Matched Aligned Reference')
            
            # Compute and display metrics
            total_matches += 1
            if dist< self.err_tolerance:
                save_good_count += 1
            recall = round(save_good_count / total_matches, 2) if total_matches > 0 else 0
            
            plt.suptitle(
                f'LPR with {self.dataset_name}_ref{self.ref_num}_qry{self.query_num}-(Frame {i}/{len(self.filtered_query_ids)})\n'
                f'Distance: {round(dist, 2)} | '
                f'Recall: {recall} | '
                f'Closest Ref ID: {self.close_ids[i]}',
                fontsize=10
            )
            
            # # Optional: Add colorbar or legend
            # plt.colorbar(ax1.imshow(query_scan, cmap='Blues'), ax=ax1, fraction=0.046, pad=0.04)
            # plt.colorbar(ax1.imshow(ref_closest, cmap='Greens'), ax=ax1, fraction=0.046, pad=0.08)
            
            # Log match details with more structured output
            print(f"Frame Analysis:")
            print(f"  Frame Index: {i}/{len(self.lpr.framesQuery)}")
            print(f"  Match Quality: {'Good' if match_quality else 'Bad'}")
            print(f"  Distance Error: {round(dist, 2)}")
            print(f"  Closest Reference ID: {self.close_ids[i]}")
            print(f"  Minimum Distance: {round(self.min_dist[i], 2)}")
            print(f"  Cumulative Metrics:")
            print(f"    Good Matches: {save_good_count}")
            print(f"    Bad Matches: {total_matches-save_good_count}")
            print(f"    Recall: {recall}")

        # Create animation
        print(f"Total query frames: {len(self.filtered_query_ids)}")
        ani = animation.FuncAnimation(
            fig, 
            update_save_good_bad,  
            frames=range(start_frame, len(self.filtered_query_ids)), 
            repeat=False
        )
        
        plt.show()
        return ani

    def save_animation(self, ani, filename: str = None):
        """
        Save the animation to a video file
        
        Args:
            ani (animation.FuncAnimation): Animation to save
            filename (str, optional): Output filename
        """
        if filename is None:
            filename = (
                f'./videos/{self.dataset_name}_REF_{self.ref_num}_'
                f'QUERY{self.query_num}_RES{self.map_res}_'
                f'DIM{self.scan_dim_x}_qIncr{self.query_incr}_'
                f'refIncr{self.ref_incr}.mp4'
            )
        
        writer_video = animation.FFMpegWriter(fps=4, bitrate=-1)
        ani.save(filename, writer=writer_video)
        self.logger.info(f"Animation saved to {filename}")

def main():
    '''
    # NCLT  --reference sequences: 10,11  --query sequences 1,2,3,4,5,6,7,8,9
    # OxfordRadar --reference sequences: 6, 7 --query sequences 1,2,3,4,5
    
    '''

    # Initialize visualizer for WildPlaces dataset
    visualizer = PlaceRecognitionVisualizer(
        dataset_name='WildPlaces_Venman',
        ref_num=1,
        query_num=2
    )

    # Initialize visualizer for NCLT dataset
    # visualizer = PlaceRecognitionVisualizer(
    #     dataset_name='NCLT',
    #     ref_num=11,
    #     query_num=2
    # )
    
    
    # Create and optionally save animation
    animation = visualizer.create_animation(start_frame=1300)
    visualizer.save_animation(animation)
        

if __name__ == '__main__':
    main()