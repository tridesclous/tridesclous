from tridesclous.online import make_empty_catalogue
from pprint import pprint



def test_make_empty_catalogue():

    empty_catalogue = make_empty_catalogue()
    
    pprint(empty_catalogue)


if __name__ == '__main__':
    test_make_empty_catalogue()
    
