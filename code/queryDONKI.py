#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 15:14:05 2025

@author: mrutala
"""
import astropy.units as u
import time

def CME(start, end):
    """
    Read the DONKI CME list
    """
    import pandas as pd
    import requests
    import datetime
    import numpy as np
    
    strptime = datetime.datetime.strptime
    
    # Split into chunks to read faster
    starts = np.arange(start, end, datetime.timedelta(days=365)).astype(datetime.datetime)
    ends = np.append(starts[1:], end)
    dfs = []
    
    for start, end in zip(starts, ends):
        # Construct URL for the request
        baseurl = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME"
        start_str = "?startDate=" + start.strftime("%Y-%m-%d")
        end_str = "&endDate=" + end.strftime("%Y-%m-%d")
        url = baseurl + start_str + end_str
        
        # Check for a response and get the data
        try:
            response = requests.get(url)
        except:
            breakpoint()
        if response.status_code != 200:
            print('cannot successfully get an http response')
        
        # read the data
        print("Getting data from", url)
        dfs.append(pd.read_json(url))
    df = pd.concat(dfs, ignore_index=True)
    
    # If df is length 0, create a dummy df with same columns
    if len(df) == 0:
        df = pd.DataFrame(columns=['activityID', 'catalog', 'startTime', 
                                   'instruments', 'sourceLocation',
                                   'activeRegionNum', 'note', 'submissionTime', 
                                   'versionId', 'link', 'cmeAnalyses', 
                                   'linkedEvents', 'sentNotifications'])
    
    # Get useful datetimes
    time_fmt = "%Y-%m-%dT%H:%MZ"
    df['startTime'] = [strptime(t, time_fmt) for t in df['startTime'].to_list()]
    
    # Must have CME analysis:
    notAnalyzed_index = []
    for index, row in df.iterrows():
        if (row['cmeAnalyses'] is None) or (row['cmeAnalyses'] == []):
            notAnalyzed_index.append(index)
    df = df.drop(notAnalyzed_index, axis='index')
    df = df.reset_index(drop=True)
    
    # Only keep the most recent CME analysis, and simplify instruments & linkedEvents
    for index, row in df.iterrows():
        cmeAnalysis_datetimes = [strptime(d['submissionTime'], time_fmt) for d in row['cmeAnalyses']]
        most_recent = np.argmax(cmeAnalysis_datetimes)   
        
        
        # Keep the most recent analysis in this list of dicts
        row['cmeAnalyses'] = row['cmeAnalyses'][most_recent]
        # Drop the ENLIL model results?
        _ = row['cmeAnalyses'].pop('enlilList')
        # Flatten the instrument names
        row['instruments'] = [d['displayName'] for d in row['instruments']]
        # Flatten the linked events
        if row['linkedEvents'] is not None:
            row['linkedEvents'] = [d['activityID'] for d in row['linkedEvents']]
        else:
            row['linkedEvents'] = []
        df.loc[index] = row
    
    return df

def ICME(start, end, location='Earth', duration=1.5*u.day, ensureCME=True):
    """
    Read the DONKI InterPlanetary Shock (IPS) list, and optionally cross-
    reference with the DONKI CME list to identify ICMEs
    """
    import pandas as pd
    import requests
    import datetime
    import numpy as np
    import time
    
    strptime = datetime.datetime.strptime
    
    # Construct URL for the request
    baseurl = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/IPS"
    start_str = "?startDate=" + start.strftime("%Y-%m-%d")
    end_str = "&endDate=" +  end.strftime("%Y-%m-%d")
    location_str = "&location=" + location.replace(' ', '%20') #no space in URL
    url = baseurl + start_str + end_str + location_str
   
    # Check for a response and get the data
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print('cannot successfully get an http response')
        # read the data
        print("Getting data from", url)
        df = pd.read_json(url)
    except:
        print('cannot successfully get an http response')
        df = pd.DataFrame(columns=['catalog', 'activityID', 'location', 
                                   'eventTime', 'submissionTime',
                                   'versionId', 'link', 'instruments', 
                                   'linkedEvents', 'sentNotifications'])
    
    # If df is length 0, create a dummy df with same columns
    if len(df) == 0:
        df = pd.DataFrame(columns=['catalog', 'activityID', 'location', 
                                   'eventTime', 'submissionTime',
                                   'versionId', 'link', 'instruments', 
                                   'linkedEvents', 'sentNotifications'])
    
    # Get useful datetimes
    time_fmt = "%Y-%m-%dT%H:%MZ"
    try: 
        df['eventTime'] = [strptime(t, time_fmt) for t in df['eventTime'].to_list()]
    except:
        # breakpoint()
        pass
    
    for index, row in df.iterrows():
        # Flatten the instrument names
        row['instruments'] = [d['displayName'] for d in row['instruments']]
        df.loc[index] = row
        
    # ICMEs must be linked to CMEs; drop IPSs that aren't
    if ensureCME:
        delta = datetime.timedelta(days = 27)
        cme_df = CME(start - delta, end)
        
        df['linkedEvents'] = ''
        for index, row in df.iterrows():
            matches = [row['activityID'] in l for l in cme_df['linkedEvents']]
            
            if np.array(matches).any() == False:
                df = df.drop(index, axis='index')
            else:
                row['linkedEvents'] = cme_df.loc[matches, 'activityID'].values[0]
                df.loc[index] = row
        
    # Set the duration of the event
    df['duration'] = duration

    return df.reset_index()