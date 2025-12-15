P_CANDIDATES = range(1, 41)
TEST_PROP = 0.1
DATASETS = [
    {
        'name': 'PJME_hourly',
        'target': 'PJME_MW',
        'parser': None 
    },
    {
        'name': 'SN_m_tot_V2.0',
        'target': 'sunspot_number',
        'parser': {
            'sep': ';',
            'header': None,
            'names': ['year', 'month', 'date_fraction', 'sunspot_number', 'std', 'num_obs', 'flag']
        }
    }
]

