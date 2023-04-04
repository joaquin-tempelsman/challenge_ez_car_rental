import pandas as pd
import logging

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


from preprocessing_funcs import journey_prep, utilization_prep1, journey_prep2

def get_preproccessed_data(raw_data_folder, EDA=False):

    logging.info('preprocessing raw data')

    logging.info('reading journeys data')
    journeys_df = pd.read_csv(raw_data_folder + 'journeys.csv')
    
    logging.info('reading utilization data')
    utilization_df = pd.read_csv(raw_data_folder + 'utilization.csv')
  
    logging.info('running data prep 1 on journeys data')
    journey_df_prep1 = journey_prep(journeys_df)
    
    logging.info('running data prep 1 on utilization data')
    utilization_df_prep1 = utilization_prep1(utilization_df, journey_df_prep1)
    
    logging.info('running data prep 2 on journey_prep1 data')
    journey_df_prep2 = journey_prep2(journey_df_prep1, utilization_df_prep1)

    logging.info('data preprocessing complete, returning train dataframe to train module')
    
    if EDA:
        return journey_df_prep2, utilization_df_prep1
    
    else:
        return journey_df_prep2
