import os
import json
import numpy as np
import pandas as pd
import logging
import pickle
import math
from typing import List, Dict, Optional, Tuple, Union
from helper import (
    LiDAR_PlaceRec
)
class LiDARPlaceRecognitionProcessor:
    """
    A comprehensive processor for LiDAR-based place recognition across different datasets
    """
    def __init__(
        self, 
        config_path: str = './scripts/config.json', 
        dataset_name: str = 'WildPlaces_Karawatha',
        ref_num: int = 1, 
        query_num: int = 2,
        save_base_path: str = './results'
    ):
        """
        Initialize the LiDAR Place Recognition Processor
        
        Args:
            config_path (str): Path to configuration JSON file
            dataset_name (str): Name of the dataset to analyze
            ref_num (int): Reference dataset number
            query_num (int): Query dataset number
            error_tolerance (float): Error tolerance for matching
            save_base_path (str): Base path for saving results
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
        
        # Set core parameters
        self.dataset_name = dataset_name
        self.ref_num = ref_num
        self.query_num = query_num
        self.save_base_path = save_base_path

        # Initialize dataset-specific parameters
        self._initialize_parameters()

        # LiDAR Place Recognition object
        self.LPR = None
        
    def _initialize_parameters(self):
        """Extract and set parameters from configuration"""

        self.error_tolerance = self.config.get('errTolerance', 1)
        params = self.config.get('parameters', {})
        
        # Incrementing and sampling parameters
        self.ref_incr = self.config.get('refIncr', 1)
        self.query_incr = self.config.get('queryIncr', 1)
        
        # Scan and mapping parameters
        self.scan_dim_x = params.get('scanDimX', 120)
        self.map_res = params.get('mapRes', 0.3)
        
        # Radius and filtering parameters 
        self.ref_radius = params.get('refRadius', 2)
        self.query_radius = params.get('queryRadius', 2)
        self.num_matches = params.get('topN', 2)
        
        # Background and thresholding
        self.background_weight = params.get('unoccWeight', -0.15)
        self.ref_thresh = params.get('refThresh', 2)
        
        # Rotation increment
        self.rot_inc = params.get('rotIncr', 10)

    def _get_dataset_frames(self) -> Tuple[List[int], str]:
        """
        Get dataset-specific frames and reference grid path
        
        Returns:
            Tuple of query frames and reference grid file path
        """
        if self.dataset_name.startswith('WildPlaces'):
            # Load evaluation info 
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
            ref_grid_name = (
                f"_refGrid_dim{self.scan_dim_x}_mapRes{self.map_res}"
                f"_refIncr{self.ref_incr}_refRad{self.ref_radius}"
                f"_rThresh{self.ref_thresh}_nptsMult{self.num_matches}_AvgDownsamp.npy"
            )
            ref_grid_path = f'./data_processed_files/{self.dataset_name}{self.ref_num}{ref_grid_name}'

            # Initialize LiDAR Place Recognition
            self.LPR = LiDAR_PlaceRec(
                self.dataset_name, 
                self.config, 
                self.ref_num, 
                self.query_num, 
                framesQuery=frames_query, 
                refGridFilePath=ref_grid_path, 
                background=self.background_weight
            )

        else:  # For NCLT, OxfordRadar
            # Dynamically generate file paths
            frames_base_path = f'{self.save_base_path}/LPR_{self.dataset_name}/FramesFiles'
            frames_query_path = (
                f'{frames_base_path}/Q:{self.query_num}_framesQuery_'
                f'qInc:{self.query_incr}_qRad:{self.query_radius}_everyN.npy'
            )
            frames_ref_path = (
                f'{frames_base_path}/R:{self.ref_num}_framesRef_'
                f'rInc:{self.ref_incr}_rRad:{self.ref_radius}_Napart.npy'
            )

            frames_query = list(np.load(frames_query_path))
            frames_ref = list(np.load(frames_ref_path))

            ref_grid_name = (
                f"_refGrid_dim{self.scan_dim_x}_mapRes{self.map_res}"
                f"_refIncr{self.ref_incr}_refRad{self.ref_radius}"
                f"_rThresh{self.ref_thresh}_nptsMult{self.num_matches}_AvgDownsamp.npy"
            )
            ref_grid_path = f'./data_processed_files/{self.dataset_name}{self.ref_num}{ref_grid_name}'

            # Initialize LiDAR Place Recognition
            self.LPR = LiDAR_PlaceRec(
                self.dataset_name, 
                self.config, 
                self.ref_num, 
                self.query_num, 
                framesQuery=frames_query, 
                framesRef=frames_ref,
                refGridFilePath=ref_grid_path, 
                background=self.background_weight
            )

        return frames_query, ref_grid_path

    def _construct_save_path(self) -> str:
        """
        Construct a save path for results based on current processor parameters
        
        Returns:
            str: Constructed save path for Excel results
        """
        # Create base folder
        if self.dataset_name.startswith('WildPlaces'):
            folder_name='LPR_Wildplaces'
        else:
            folder_name = f'LPR_{self.dataset_name}' 
        save_folder = os.path.join(self.save_base_path, folder_name,'Testing')
        os.makedirs(save_folder, exist_ok=True)

        # Construct filename with key parameters
        save_path = os.path.join(save_folder, 
            f'{self.dataset_name}_'
            f'R:{self.ref_num}_Q:{self.query_num}_'
            f'RInc:{self.ref_incr}_Rrad:{self.ref_radius}_'
            f'QInc:{self.query_incr}_Qrad:{self.query_radius}_'
            f'Res:{self.map_res}_Dim:{self.scan_dim_x}_'
            f'blkAvg:{self.config.get("parameters", {}).get("blkSize", 2)}_'
            f'N2ndsearch:{self.num_matches}_'
            f'background:{self.background_weight}.xlsx'
        )
        print(save_path)
        return save_path
    
    def process_place_recognition(
        self, 
        save_excel: bool = True, 
        return_times: bool = False):
        """
        Perform place recognition processing
        
        Args:
            save_excel (bool): Whether to save results to Excel
            return_times (bool): Whether to return timing metrics
        
        Returns:
            Processed data or timing metrics
        """
        try:
            # Get frames and reference grid path
            frames_query, ref_grid_path = self._get_dataset_frames()

            # Construct save path for results
            if save_excel==True:
                save_path = self._construct_save_path() if save_excel else None
            else:
                save_path = None
            
            print(f"Total ref frames: {self.LPR.numRefScans()}")
            # Prepare data for processing
            centerXs, centerYs = zip(*self.LPR.scanCenter())
            xq, yq, rollQ, pitchQ, yawQ = self.LPR.scanPoses('QUERY')
            xr, yr, rollR, pitchR, yawR = self.LPR.scanPoses('REF')
            
            
            # Find closest points
            closeIds = [self.LPR.find_closest_point(zip(xr, yr), xq[i], yq[i])[2] for i in range(len(xq))]
            minDist = [self.LPR.find_closest_point(zip(xr, yr), xq[i], yq[i])[3] for i in range(len(xq))]

            # Filter frames based on error tolerance
            filteredFramesQuery = [
                frames_query[i] for i in range(len(frames_query)) 
                if minDist[i] < self.error_tolerance
            ]
            
            self.logger.info(
                f'{len(frames_query) - len(filteredFramesQuery)} queries removed '
                f'from {len(frames_query)} for being outside {self.error_tolerance}m from ref'
            )

            # Process place recognition
            return self._run_place_recognition(
                save_path, 
                xq, yq, yawQ, 
                xr, yr, yawR, 
                centerXs, centerYs, 
                closeIds, 
                minDist, 
                return_times
            )

        except Exception as e:
            self.logger.error(f"Place recognition processing failed: {e}")
            return None

    def _run_place_recognition(
        self, 
        save_path: Optional[str], 
        xq, yq, yawQ, 
        xr, yr, yawR, 
        centerXs, centerYs, 
        closeIds, 
        minDist, 
        return_times: bool = False
    ):
        """
        Internal method to run place recognition processing
        
        Args:
            save_path (str): Path to save Excel results
            xq, yq, yawQ: Query scan poses
            xr, yr, yawR: Reference scan poses
            centerXs, centerYs: Scan centers
            closeIds: Closest point indices
            minDist: Minimum distances
            return_times (bool): Whether to return timing metrics
        
        Returns:
            Results or timing metrics
        """

        dataStorage = []
        timelapse, err, corr, numEval = 0, 0, 0, 0
        processTimelapse, shiftTimelapse, queryLoadTimelapse = 0, 0, 0

        # Determine starting point
        start = 0
        if save_path and os.path.exists(save_path):
            numRows = len(pd.read_excel(save_path)) * self.query_incr
            start = numRows

        for i in range(start, len(self.LPR.framesQuery), self.query_incr):
            # Update and get matching results
            idx, maxX, maxY, matchXDelta, matchYDelta, shiftedX, shiftedY, rotations, alignment, dist, queryLoadTime, processTime, shiftTime, duration = self.LPR.update(
                i, self.rot_inc, xr, yr, yawR, xq, yq, yawQ, centerXs, centerYs, closeIds
            )

            numEval += 1
            if dist > self.error_tolerance:
                err += 1
            else:
                corr += 1
            recall = round(corr/numEval, 2)

            # Track time metrics
            timelapse += duration
            processTimelapse += processTime
            shiftTimelapse += shiftTime
            queryLoadTimelapse += queryLoadTime

            # Log processing details
            self.logger.info(
                f'{i}/{len(self.LPR.framesQuery)}, err:{err}, recall: {recall}, '
                f'distErr:{round(dist,2)}, idx:{idx}, closestId: {closeIds[i]}, '
                f'minDist: {round(minDist[i],2)}'
            )

            # Prepare test result
            testResult = {
                'queryPose': (xq[i], yq[i], yawQ[i]),
                'matchID': idx,
                'refgridPeak': (maxX, maxY),
                'matchPose': (xr[idx], yr[idx], yawR[idx]),
                'shiftAmount': (matchXDelta, matchYDelta),
                'matchShiftedPosition': (shiftedX, shiftedY),
                'dist': dist,
                'peakRotation': rotations[np.argmax(alignment)],
                'peakAlignment': np.max(alignment),
                'closestId': closeIds[i],
                'closePose': (xr[closeIds[i]], yr[closeIds[i]], yawR[closeIds[i]]),
                'closeDist': minDist[i],
                'timePerQuery': round(duration, 3),
                'descGenTime': round(queryLoadTime, 3),
                'searchTime': round(processTime, 3),
                'poseCorrectTime': round(shiftTime, 3)
            }
            dataStorage.append(testResult)

            # Save to Excel if path provided
            if save_path:
                self._save_to_excel(save_path, dataStorage)

        # Return timing metrics or results based on flag
        if return_times:
            return (
                queryLoadTimelapse/numEval if numEval else 0, 
                processTimelapse/numEval if numEval else 0, 
                shiftTimelapse/numEval if numEval else 0, 
                timelapse/numEval if numEval else 0, 
                corr, 
                numEval
            )
        return dataStorage

    def _save_to_excel(self, save_path: str, dataStorage: List[Dict]):
        """
        Save results to Excel file
        
        Args:
            save_path (str): Path to save Excel file
            dataStorage (List[Dict]): Data to save
        """
        if os.path.exists(save_path):
            df1 = pd.read_excel(save_path)
            df1 = pd.concat([df1, pd.DataFrame(dataStorage[-1:])], ignore_index=True)
        else:
            df1 = pd.DataFrame(dataStorage)
        
        with pd.ExcelWriter(save_path) as writer:
            df1.to_excel(writer, index=False)

    def save_ref_query_split(
        self, 
        force_resave: bool = False
    ) -> Tuple[str, str]:
        """
        Save reference and query frame splits
        
        Args:
            force_resave (bool): Force resaving even if files exist
        
        Returns:
            Tuple of query and reference save file paths
        """
        # Construct paths
        base_path = f'{self.save_base_path}/LPR_{self.dataset_name}/FramesFiles'
        os.makedirs(base_path, exist_ok=True)

        query_save_path = os.path.join(base_path, 
            f'Q:{self.query_num}_framesQuery_'
            f'qInc:{self.query_incr}_qRad:{self.query_radius}_everyN.npy'
        )
        ref_save_path = os.path.join(base_path, 
            f'R:{self.ref_num}_framesRef_'
            f'rInc:{self.ref_incr}_rRad:{self.ref_radius}_Napart.npy'
        )

        # If force_resave is True, remove existing files
        if force_resave:
            if os.path.exists(query_save_path):
                os.remove(query_save_path)
            if os.path.exists(ref_save_path):
                os.remove(ref_save_path)

        # Load configuration
        with open('./scripts/config.json') as f:
            config = json.load(f)
        
        # Get dataset-specific configuration
        dataset_config = config.get(self.dataset_name, {})
        query_filenames = dataset_config.get('details', {}).get(str(self.query_num), [])
        params = dataset_config.get('parameters', {})

        # Process reference frames
        if not os.path.exists(ref_save_path):
            self.logger.info(f'Creating file to store reference frames: {ref_save_path}')
            
            # Filter numeric filenames
            filenames = os.listdir(query_filenames[0])
            filenames = [f for f in filenames if f[:-4].replace('.','',1).isdigit()]
            
            # Create frames for query
            frames_query = range(0, len(filenames), self.query_incr)
            
            # Initialize LiDAR Place Recognition
            LPR = LiDAR_PlaceRec(
                self.dataset_name, 
                dataset_config, 
                self.ref_num, 
                self.query_num, 
                framesQuery=frames_query, 
                refGridFilePath=None
            )
            
            # Get all reference poses
            xr_all, yr_all, roll_r, pitch_r, yaw_r_all = LPR.scanPoses('All')
            
            # Subsample reference frames
            frames_ref = LPR.subsample_Napart(xr_all, yr_all, self.ref_radius)
            
            # Save reference frames
            np.save(ref_save_path, np.array(frames_ref))
            self.logger.info(f'Reference frames saved. Length: {len(frames_ref)}')
        else:
            frames_ref = list(np.load(ref_save_path))
            self.logger.info(f'Existing reference frames loaded. Length: {len(frames_ref)}')

        # Process query frames
        if not os.path.exists(query_save_path):
            self.logger.info(f'Creating file to store query frames: {query_save_path}')
            
            # Filter numeric filenames
            filenames = os.listdir(query_filenames[0])
            filenames = [f for f in filenames if f[:-4].replace('.','',1).isdigit()]
            
            # Create frames for query
            frames_query = range(0, len(filenames), self.query_incr)
            
            # Initialize LiDAR Place Recognition
            LPR = LiDAR_PlaceRec(
                self.dataset_name, 
                dataset_config, 
                self.ref_num, 
                self.query_num, 
                frames_query, 
                refGridFilePath=None, 
                framesRef=frames_ref
            )
            
            # Get all query poses
            xq_all, yq_all, roll_q, pitch_q, yaw_q_all = LPR.scanPoses('AllQuery')
            
            # Subsample query frames
            frames_query = LPR.subsample_everyN(xq_all, yq_all, self.query_radius)
            
            # Save query frames
            np.save(query_save_path, np.array(frames_query))
            self.logger.info(f'Query frames saved. Length: {len(frames_query)}')
        else:
            frames_query = list(np.load(query_save_path))
            self.logger.info(f'Existing query frames loaded. Length: {len(frames_query)}')

        return query_save_path, ref_save_path


# Example usage
def main():
    """Example usage of LiDARPlaceRecognitionProcessor"""
    # Example for WildPlaces dataset
    # processor_wild = LiDARPlaceRecognitionProcessor(
    #     dataset_name='WildPlaces_Venman', 
    #     ref_num=1, 
    #     query_num=2
    # )
    # results_wild = processor_wild.process_place_recognition(save_excel=False)

    # Example for NCLT dataset
    # processor_nclt = LiDARPlaceRecognitionProcessor(
    #     dataset_name='NCLT', 
    #     ref_num=10, 
    #     query_num=6
    # )
    # processor_nclt.save_ref_query_split()
    # results_nclt = processor_nclt.process_place_recognition(save_excel=True)

    # Example for OxfordRadar dataset
    processor_nclt = LiDARPlaceRecognitionProcessor(
        dataset_name='OxfordRadar', 
        ref_num=6, 
        query_num=1
    )
    processor_nclt.save_ref_query_split()
    results_nclt = processor_nclt.process_place_recognition(save_excel=True)


if __name__ == "__main__":
    main()




