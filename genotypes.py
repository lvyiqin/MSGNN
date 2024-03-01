from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')



NODE_PRIMITIVES = [
  'node_add_1',
  'node_add_2',
  'node_remove_1',
  'node_remove_2',
  'none',
]

EDGE_PRIMITIVES=[
  'edge_add_1',
  'edge_add_2',
  'edge_remove_1',
  'edge_remove_2',
  'none',
]
COMBINE_PRIMITIVES = [
  'two_two',
  'three_three',
  'none',
]



