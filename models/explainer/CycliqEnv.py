
import collections
import copy
import itertools
import torch_geometric
import numpy as np
import networkx as nx
import torch
import random

from torch.nn import functional as F
from torch_geometric.utils import to_undirected, to_networkx, from_networkx
from models.explainer.Environment import Result
from utils import get_similarity

def get_valid_actions(graph, allow_removal, allow_no_modification):
  valid_actions = set()
  N = []

  valid_actions.update(_node_addition(graph))
  N.append(len(valid_actions))

  if allow_removal:
    valid_actions.update(_edge_removal(graph))
    N.append(len(valid_actions) - N[0])

  valid_actions.update(_edge_addition(graph, max_action=sum(N)//len(N)))

  if allow_no_modification:
    valid_actions.add(graph.clone())

  return valid_actions


def _node_addition(graph):
  num_nodes, num_features = graph.x.size()

  actions = set()

  for i in range(num_nodes):
    a = graph.clone()
    a.x = torch.cat((a.x, torch.ones((1, num_features))))
    a.edge_index = to_undirected(
      torch.cat((a.edge_index, torch.tensor([[num_nodes], [i]])), dim=1)
    )
    a.num_nodes = a.num_nodes + 1
    actions.add(a)
  return actions


def _edge_addition(graph, max_action=1000):
  actions = set()
  num_nodes, _ = graph.x.size()
  g = to_networkx(graph, to_undirected=True)
  edges = []


  edge_set = set(itertools.combinations(range(num_nodes), 2))
  edge_set = edge_set.difference(set(g.edges))
  edge_set = random.sample(edge_set, max_action)

  for (u, v) in edge_set:
    a = graph.clone()

    a.edge_index = to_undirected(
      torch.cat((a.edge_index, torch.tensor([[u], [v]])), dim=1)
    )

    actions.add(a)

  return actions


def _edge_removal(graph):
  actions = set()

  _, num_edges = graph.edge_index.size()
  edges = graph.edge_index.t().numpy().tolist()
  for i in range(num_edges//2):
    tmp = to_networkx(graph, to_undirected=True)
    tmp.remove_edge(*edges[i])
    tmp = from_networkx(tmp)
    a = graph.clone()
    a.edge_index = tmp.edge_index

    actions.add(a)

  return actions


class GraphEnvironment():
  def __init__(self,
               init_graph=None,
               allow_removal=True,
               allow_no_modification=True,
               max_steps=10,
               record_path=False,
               target_fn=None
  ):

    assert isinstance(init_graph, torch_geometric.data.Data)

    self.init_graph = init_graph
    self.allow_removal = allow_removal
    self.allow_no_modification = allow_no_modification
    self.max_steps = max_steps
    self._state = None
    self._valid_actions = []
    self._counter = self.max_steps
    self.record_path = record_path
    self._target_fn = target_fn
    self._path = []

  @property
  def state(self):
    return self._state

  @property
  def num_steps_taken(self):
    return self._counter

  def get_path(self):
    return self._path

  def initialize(self):
    self._state = self.init_graph
    if self.record_path:
      self._path = [self._state]
    self._valid_actions = self.get_valid_actions(force_rebuild=True)
    self._counter = 0

  def get_valid_actions(self, state=None, force_rebuild=False):
    if state is None:
      if self._valid_actions and not force_rebuild:
        return copy.deepcopy(self._valid_actions)
      state = self._state

    self._valid_actions = get_valid_actions(
        state,
        allow_removal=self.allow_removal,
        allow_no_modification=self.allow_no_modification)
    return copy.deepcopy(self._valid_actions)

  def _reward(self):
    return 0.0

  def _goal_reached(self):
    if self._target_fn is None:
      return False
    return self._target_fn(self._state)

  def step(self, action):
    if self._counter >= self.max_steps or self._goal_reached():
      raise ValueError('This episode is terminated.')

    self._state = action
    if self.record_path:
      self._path.append(self._state)
    self._valid_actions = self.get_valid_actions(force_rebuild=True)
    self._counter += 1

    result = Result(
        state=self._state,
        reward=self._reward(),
        terminated=(self._counter >= self.max_steps) or self._goal_reached())
    return result


class CF_Cycliq(GraphEnvironment):
  def __init__(self,
               model_to_explain,
               original_graph,
               discount_factor,
               weight_sim=0.5,
               similarity_measure="neural_encoding",
               **kwargs):

    super(CF_Cycliq, self).__init__(**kwargs)
    self.class_to_optimise = 1 - original_graph.y.item()

    self.discount_factor = discount_factor
    self.model_to_explain = model_to_explain
    self.weight_sim = weight_sim


    self.similarity, self.make_encoding, \
      self.original_encoding = get_similarity(similarity_measure,
                                              None,
                                              model_to_explain,
                                              original_graph)

  def _reward(self):

    g = self._state.clone()

    out, encoding = self.model_to_explain(g.x, g.edge_index)
    out = F.softmax(out, dim=-1).squeeze().detach()

    sim_score = self.similarity(self.make_encoding(g), self.original_encoding)
    pred_score = out[self.class_to_optimise].item()
    pred_class = torch.argmax(out).item()



    reward = pred_score * (1 - self.weight_sim) + sim_score * self.weight_sim

    return {
      'pyg': g,
      'reward': reward * self.discount_factor,
      'reward_pred': pred_score,
      'reward_sim': sim_score,
      'encoding': encoding.numpy(),
      'prediction': {
        'type': 'bin_classification',
        'output': out.numpy().tolist(),
        'for_explanation': pred_class,
        'class': pred_class
      }
    }


class NCF_Cycliq(CF_Cycliq):
    def __init__(
            self,
            **kwargs
    ):
        super(NCF_Cycliq, self).__init__(**kwargs)
        self.class_to_optimise = kwargs['original_graph'].y.item()
