import pandas as pd
import numpy as np
from tridesclous import median_mad



def test_get_median_mad():
    df = pd.DataFrame(np.random.randn(20, 8), index = np.arange(20), columns = list('abcdefgh'))
    
    med, mad = median_mad(df, axis=0)
    assert np.all(med.index == df.columns)
    assert np.all(mad.index == df.columns)
    
    med, mad = median_mad(df, axis=1)
    assert np.all(med.index == df.index)
    assert np.all(mad.index == df.index)
    
    
if __name__ == '__main__':
    test_get_median_mad()