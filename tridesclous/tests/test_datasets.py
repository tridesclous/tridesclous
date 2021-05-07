from tridesclous import download_dataset, get_dataset


def test_download_dataset():
    
    download_dataset(name='locust')
    download_dataset(name='olfactory_bulb')

def test_get_dataset():
    data, sample_rate = get_dataset(name='locust')
    assert data.shape == (431548, 4)
    assert sample_rate == 15000.0
    
    data, sample_rate = get_dataset(name='olfactory_bulb')
    assert data.shape == (150000, 14)
    assert sample_rate == 10000.0
    
    
    data, sample_rate = get_dataset(name='purkinje')
    
    data, sample_rate = get_dataset(name='striatum_rat')
    



if __name__ == '__main__':
    test_download_dataset()
    test_get_dataset()