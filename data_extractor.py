"""
CARENET
@author: aaq109
"""

import pickle

class DataReaderRHIP:
    
    def __init__(self, fname):

        with open(fname+'_train'+'.pkl', 'rb') as handle:
            self.data_train = pickle.load(handle)           

        with open(fname+'_val'+'.pkl', 'rb') as handle:
            self.data_val = pickle.load(handle) 

        with open(fname+'_test'+'.pkl', 'rb') as handle:
            self.data_test = pickle.load(handle) 


'''
The input data is a list of dictionaries. 
data = []
A dictionary is unique for each patient visit and build as follows:
pat_visit = dict()
pat_visit['Id'] = i   Unique patient visit identifier (int)
pat_visit['text'] = [] List of list of list of all clinical codes. 
For instance for 2 time periods, 3 contacts the list looks like as follows;
[[[a,b],[c,d],[e,f]],[[a1,b1],[c1,d1,d2,e4],[NA]]]
Here, [a,b] are the clinical codes appearing in context 0 in time period 0
[c1,d1,d2,e4] are the clinical codes appearing in context 1 in time period 1
The number of time periods and contexts should be the same for all patient visits.
The number of clinical codes inside the context may vary.
If there is no code in any context, add "NA"
pat_visit['onsite'] = [] List all onsite numeric features. One for each patient_visit
pat_visit['hdf'] = [] List of list of all clinical features. One for each time period.
pat_visit['Label_multi'] = [] List of all label IDs (int)
pat_visit['catgy'] = [] List with only one primary label ID (int) 
Finally, data.append(pat_visit)
Save data as a pkl file "ehr_input.pkl"
'''









