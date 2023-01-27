import pytest
from pprint import pprint

from tridesclous.tests.testingtools import ON_CI_CLOUD
from tridesclous.online import HAVE_PYACQ

if HAVE_PYACQ:
    from tridesclous.online import make_empty_catalogue


@pytest.mark.skipif(not HAVE_PYACQ, reason='no pyacq')
@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_make_empty_catalogue():

    empty_catalogue = make_empty_catalogue()

    pprint(empty_catalogue)


if __name__ == '__main__':
    test_make_empty_catalogue()

