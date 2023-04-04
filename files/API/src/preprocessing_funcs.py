import pandas as pd
import datetime
from datetime import datetime, timedelta
import numpy as np


def journey_prep(journeys_df):

    _journeys_df = journeys_df.copy()

    #cols to lower and space to underscore
    _journeys_df.columns = _journeys_df.columns.str.lower().str.replace(' ', '_')

    #cols to datetime
    _journeys_df['trip_start_at_local_time'] = _journeys_df['trip_start_at_local_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    _journeys_df['trip_end_at_local_time'] = _journeys_df['trip_end_at_local_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    _journeys_df['trip_created_at_local_time'] = _journeys_df['trip_created_at_local_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

    #clean price and dtype to float
    _journeys_df['trip_sum_trip_price'] = _journeys_df['trip_sum_trip_price'].str.replace('$','').str.replace(',','').astype(float)

    # add trip features [time_until_arrival, trip_duration, cost_per_minute]
    _journeys_df['reservation_lead_time'] = (_journeys_df.trip_start_at_local_time - _journeys_df.trip_created_at_local_time).dt.total_seconds() / 60
    _journeys_df['trip_duration'] = (_journeys_df.trip_end_at_local_time - _journeys_df.trip_start_at_local_time).dt.total_seconds() / 60
    _journeys_df['cost_per_minute'] = _journeys_df['trip_sum_trip_price'].astype(float) / _journeys_df['trip_duration'].astype(float)

    # add time features [year, month, weekday, day_nr, hour, weekend_day, weekend]
    _journeys_df['year'] = _journeys_df.trip_start_at_local_time.dt.year
    _journeys_df['month'] = _journeys_df.trip_start_at_local_time.dt.month
    _journeys_df['date'] = _journeys_df.trip_start_at_local_time.dt.strftime("%Y-%m-%d")
    _journeys_df['month_name'] = _journeys_df.trip_start_at_local_time.dt.strftime("%B")
    _journeys_df['weekday'] = _journeys_df.trip_start_at_local_time.dt.strftime("%A")
    _journeys_df['day_nr'] = _journeys_df.trip_start_at_local_time.dt.day
    _journeys_df['hour'] = _journeys_df.trip_start_at_local_time.dt.hour
    _journeys_df['weekend_day'] = _journeys_df.trip_start_at_local_time.dt.strftime("%A")
    _journeys_df['weekend'] = np.where(_journeys_df['weekday'].isin(['Saturday','Sunday']), 'weekend', 'weekday')
    _journeys_df['7_to_12hs'] = np.where(_journeys_df.hour.isin([0,1,2,3,4,5,6]),0,1)
    _journeys_df['month_nr'] = _journeys_df.trip_start_at_local_time.dt.month
    _journeys_df['trip_start_at_local_time_floor'] =  _journeys_df.trip_start_at_local_time.dt.floor("H")

    return _journeys_df

def utilization_prep1(utilization_df, journeys_df):

    _journeys_df = journeys_df.copy()
    _utilization_df = utilization_df.copy()

    #cols to lower and space to underscore
    _utilization_df.columns = _utilization_df.columns.str.lower().str.replace(' ', '_')

    #cols to datetime and hour feature
    _utilization_df['car_hourly_utilization_aggregated_at_time'] = _utilization_df['car_hourly_utilization_aggregated_at_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    _utilization_df['hour'] = _utilization_df.car_hourly_utilization_aggregated_at_time.dt.hour
    _utilization_df['month_nr'] = _utilization_df.car_hourly_utilization_aggregated_at_time.dt.month
    _utilization_df['weekday'] = _utilization_df.car_hourly_utilization_aggregated_at_time.dt.strftime("%A")
    _utilization_df['weekend'] = np.where(_utilization_df['weekday'].isin(['Saturday','Sunday']), 'weekend', 'weekday')

    # fix cases where the utilization is bigger than the availability. We input the availability as the utilization. We create a flag to mark this data was corrected.
    _utilization_df['imputed_utilization'] = np.where(_utilization_df.car_hourly_utilization_sum_utilized_minutes > _utilization_df.car_hourly_utilization_sum_available_minutes, True, False)
    _utilization_df.loc[_utilization_df.car_hourly_utilization_sum_utilized_minutes > _utilization_df.car_hourly_utilization_sum_available_minutes, 'car_hourly_utilization_sum_utilized_minutes'] = _utilization_df.car_hourly_utilization_sum_available_minutes

    # add percentage of utilization feature and correction for div 0
    _utilization_df['hourly_capacity_usage'] =  _utilization_df['car_hourly_utilization_sum_utilized_minutes'] / _utilization_df['car_hourly_utilization_sum_available_minutes'] 
    
    # REV - PROBABLY NAN MEANS NOT AVAILABLE #
    #_utilization_df.loc[_utilization_df['car_hourly_utilization_sum_available_minutes'] == 0, 'hourly_capacity_usage'] = 0

    # 
    _utilization_df['car_hourly_unused_minutes'] = _utilization_df['car_hourly_utilization_sum_available_minutes'] - _utilization_df['car_hourly_utilization_sum_utilized_minutes'] 

    # merge parking location data from journeys dataframe
    cols_to_join = ['car_id_hash','car_parking_address_city','car_parking_address_postcode']
    _utilization_df = _utilization_df.merge(_journeys_df[cols_to_join].drop_duplicates(), on='car_id_hash', how='left')

    return _utilization_df


# agg function for pandas groupby
def q10(x):
        return x.quantile(0.1)

def count_zeros(x):
    return x.eq(0).sum()

def journey_prep2( journeys_df_prep1, utilization_df_prep1):

    _journeys_df_prep1 = journeys_df_prep1.copy()
    _utilization_df_prep1 = utilization_df_prep1.copy()

    # we suspect that the capacity available is a strong predictor of the price
    # we will use utilization data so we can give memory (long and short) as features
    # first, historical month usage will bring the capacity for each postcode by month, hour, segmented
    # by weekend or labour day as long memory

    # second,  previous_hour will bring short memory with information on the capacity
    # on the latest hour, segmented by postcode, date and hour


    # with this utilization group data, we will get the seasonal information on usage of the fleet
    # we will join this data to the journeys df by the keys: # ['car_parking_address_postcode','hour','month_nr',
                                                              # 'weekend']
                                                              
    hist_month_usage_df_group = _utilization_df_prep1.groupby(['car_parking_address_postcode','hour','month_nr','weekend']).agg(
                                                                    hist_usage_q10=('hourly_capacity_usage', q10), 
                                                                    hist_usage_median=('hourly_capacity_usage', 'median'),
                                                                    hist_free_cars=('hourly_capacity_usage', count_zeros)).reset_index()


    # with this utilization group data, we get the previous hour information on usage of the fleet
    # we will join this data to the journeys df by the keys ['last_hour_usage_q10','car_hourly_utilization_aggregated_at_time_minus_1_hour']
    previous_hour_usage_df_group = _utilization_df_prep1.groupby(['car_parking_address_postcode','car_hourly_utilization_aggregated_at_time']).agg(
                                                                    prev_hour_usage_q10=('hourly_capacity_usage', q10), 
                                                                    prev_hour_usage_median=('hourly_capacity_usage', 'median'),
                                                                    prev_free_cars=('hourly_capacity_usage', count_zeros)).reset_index()

    # we add 1 hour to the timestamp so we can join to journeys_df and drop unrequired col
    previous_hour_usage_df_group['car_hourly_utilization_aggregated_at_time_plus_1_hour'] = previous_hour_usage_df_group.car_hourly_utilization_aggregated_at_time + timedelta(hours=1)
    previous_hour_usage_df_group.drop(columns=['car_hourly_utilization_aggregated_at_time'],inplace=True)

    # we merge the data to journeys df with the specific keys for each dataframe
    _journeys_df_prep1 = _journeys_df_prep1.merge(hist_month_usage_df_group,how='left' ,on=['car_parking_address_postcode', 'hour', 'month_nr', 'weekend'])
    _journeys_df_prep1 = _journeys_df_prep1.merge(previous_hour_usage_df_group,how='left' ,left_on=['car_parking_address_postcode','trip_start_at_local_time_floor'], right_on=['car_parking_address_postcode','car_hourly_utilization_aggregated_at_time_plus_1_hour'])

    return _journeys_df_prep1

def prep_data_api(df):
    
    cols = ['car_parking_address_city','trip_duration','date','trip_start_at_local_time','trip_sum_trip_price']
    _df = df.copy()
    _df =_df.loc[:,cols]
    _df['month'] = _df.date.str.split('-').str[1]
    _df['time'] = _df.trip_start_at_local_time.dt.hour
    _df.drop(columns=['trip_start_at_local_time'],inplace=True)
    return _df