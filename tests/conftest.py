import pytest
import pickle
import joblib
from pathlib import Path


@pytest.fixture(scope='session')
def titanic():
    root_path = Path(__file__).resolve().parents[1]
    file = root_path / 'pdpbox' / 'datasets' / 'test_titanic.pkl'
    return joblib.load(file)


@pytest.fixture(scope='module')
def titanic_data(titanic):
    return titanic['data']


@pytest.fixture(scope='module')
def titanic_features(titanic):
    return titanic['features']


@pytest.fixture(scope='module')
def titanic_target(titanic):
    return titanic['target']


@pytest.fixture(scope='module')
def titanic_model(titanic):
    return titanic['xgb_model']
