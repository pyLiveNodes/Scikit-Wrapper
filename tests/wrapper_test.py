import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict

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

    def __init__(self, name="KNN", n=10, **kwargs):
        super().__init__(name, **kwargs)
        self.n = n
        self.x = []
        self.y = []

    def _settings(self):
        return {"n": self.n}

    def process(self, x, y, **kwargs):
        self.x.append(x)
        self.y.append(y)
        # IMPORTANT: this is lilkely not how this should be implemented. As this re-creates a model on only one input and every call.
        # Please consider when and how a model should be built for your usecase.
        knn = KNeighborsClassifier(self.n)
        knn.fit(self.x, self.y)
        return self.ret(any=knn)
    
@registry.nodes.decorator
class KNN_prd(Node):
    ports_in = Ports_dual() # this obviously should be (x, model) not (x, y)... but this is just a test
    ports_out = Ports_any()

    def process(self, x, y=None, **kwargs):
        if y is not None:
            self.model = y

        # IMPORTANT: this is lilkely not how this should be implemented. As this re-creates a model on only one input and every call.
        # Please consider when and how a model should be built for your usecase.
        # assuming y is the model and x the data
        return self.ret(any=self.model.predict([x]))


@pytest.fixture
def est():
    fit_graph = KNN_fit(name="KNN").to_compact_dict(graph=True)
    prd_graph = KNN_prd(name="KNN").to_compact_dict(graph=True)

    est = LN_Estimator(fit_graph_dct=fit_graph, prd_graph_dct=prd_graph, 
                        fit_x_channel="KNN [KNN_fit].x", fit_y_channel="KNN [KNN_fit].y", fit_model_channel="KNN [KNN_fit].any", 
                        prd_x_channel="KNN [KNN_prd].x", prd_y_channel="KNN [KNN_prd].any", prd_model_channel="KNN [KNN_prd].y", 
                        fit_params={"KNN [KNN_fit]": dict(n=1)}, prd_params=dict())
    
    return est

class TestWrapper():

    def test_creation(self, est):
        X = [[0, 1], [1, 2], [3, 4], [4, 5]]
        y = [0, 0, 1, 1]

        initial_node, model_node = est.construct_fit_graph(est.fit_graph_dct, est.fit_x_channel, est.fit_y_channel, est.fit_model_channel, est.fit_params, [X], [y])
        dct = initial_node.to_compact_dict(graph=True)

        assert dct['Nodes']['KNN [KNN_fit]']['n'] == 1
        assert dct['Nodes']['KNN [KNN_fit]']['name'] == "KNN"


    def test_process(self, est):
        X = [[0, 1], [1, 2], [3, 4], [4, 5]]
        y = [0, 0, 1, 1]

        est.fit(X, y)
        assert y == list(est.predict(X))


    def test_cross_val(self, est):
        X = [[0, 1], [1, 2], [3, 4], [4, 5]]
        y = [0, 0, 1, 1]

        assert y == list(cross_val_predict(est, X, y, cv=2))



if __name__ == "__main__":
    from sklearn.model_selection import cross_val_predict

    X = [[0, 1], [1, 2], [3, 4], [4, 5]]
    y = [0, 0, 1, 1]

    fit_graph = KNN_fit(name="KNN").to_compact_dict(graph=True)
    prd_graph = KNN_prd(name="KNN").to_compact_dict(graph=True)

    estimator = LN_Estimator(fit_graph_dct=fit_graph, prd_graph_dct=prd_graph, 
                        fit_x_channel="KNN [KNN_fit].x", fit_y_channel="KNN [KNN_fit].y", fit_model_channel="KNN [KNN_fit].any", 
                        prd_x_channel="KNN [KNN_prd].x", prd_y_channel="KNN [KNN_prd].any", prd_model_channel="KNN [KNN_prd].y", 
                        fit_params={"KNN [KNN_fit]": dict(n=1)}, prd_params=dict())

    initial_node, model_node = estimator.construct_fit_graph(estimator.fit_graph_dct, estimator.fit_x_channel, estimator.fit_y_channel, estimator.fit_model_channel, estimator.fit_params, [X], [y])
    dct = initial_node.to_compact_dict(graph=True)

    assert dct['Nodes']['KNN [KNN_fit]']['n'] == 1
    assert dct['Nodes']['KNN [KNN_fit]']['name'] == "KNN"

    estimator.fit(X, y)
    assert y == list(estimator.predict(X))

    print(cross_val_predict(estimator, X, y, cv=2))