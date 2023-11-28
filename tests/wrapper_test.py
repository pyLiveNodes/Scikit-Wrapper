import pytest
from sklearn.neighbors import KNeighborsClassifier

from livenodes import Node
from livenodes_core_nodes.ports import Ports_any, Port_Any
from typing import NamedTuple
from ln_scikit import LN_Estimator

from livenodes import get_registry
registry = get_registry()

class Ports_dual(NamedTuple):
    x: Port_Any = Port_Any("X")
    y: Port_Any = Port_Any("y")

@registry.nodes.decorator
class KNN_fit(Node):
    ports_in = Ports_dual()
    ports_out = Ports_any()

    def __init__(self, name="KNN", n=1, **kwargs):
        super().__init__(name, **kwargs)
        self.n = n

    def process(self, x, y, **kwargs):
        # IMPORTANT: this is lilkely not how this should be implemented. As this re-creates a model on only one input and every call.
        # Please consider when and how a model should be built for your usecase.
        knn = KNeighborsClassifier(self.n)
        knn.fit(x, y)
        return self.ret(any=knn)
    
@registry.nodes.decorator
class KNN_prd(Node):
    ports_in = Ports_dual()
    ports_out = Ports_any()

    def process(self, x, y, **kwargs):
        # IMPORTANT: this is lilkely not how this should be implemented. As this re-creates a model on only one input and every call.
        # Please consider when and how a model should be built for your usecase.
        # assuming y is the model and x the data
        return self.ret(any=y.predict(x))


@pytest.fixture
def est():
    fit_graph = KNN_fit(name="KNN").to_compact_dict(graph=True)
    prd_graph = KNN_prd(name="KNN").to_compact_dict(graph=True)

    est = LN_Estimator(fit_graph=fit_graph, prd_graph=prd_graph, 
                        fit_x_channel="KNN [KNN_fit].x", fit_y_channel="KNN [KNN_fit].y", fit_model_channel="KNN [KNN_fit].any", 
                        prd_x_channel="KNN [KNN_prd].x", prd_y_channel="KNN [KNN_prd].any", prd_model_channel="KNN [KNN_prd].y", 
                        fit_params=dict(n=10), prd_params=dict(n=10))
    
    return est

class TestWrapper():

    def test_creation(self, est):
        X = [[0], [1], [2], [3]]
        y = [0, 0, 1, 1]
        initial_node, model_node = est.construct_fit_graph(est.fit_graph_dct, est.fit_x_channel, est.fit_y_channel, est.fit_model_channel, est.fit_params, [X], [y])
        print(initial_node.to_dict())