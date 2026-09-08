import random
import pytest
from types import SimpleNamespace
from robot_agent_sim.scene.directions import normalize_direction
from robot_agent_sim.scene.composer import SceneComposer

@pytest.mark.parametrize('word,expected', [('东','right'),('西','left'),('南','back'),('北','front')])
def test_direction_alias(word, expected):
    assert normalize_direction(word) == expected

@pytest.mark.parametrize('relation', ['above', 'below'])
def test_vertical_pair_preserved(relation):
    composer = SceneComposer()
    a, b = composer._relation_pair_positions(relation, (.06,)*3, (.06,)*3)
    intent = SimpleNamespace(spatial_relations=[])
    a = composer._position('a', (.06,)*3, [], random.Random(7), intent, a)
    b = composer._position('b', (.06,)*3, [], random.Random(7), intent, b)
    assert (a[2] > b[2]) if relation == 'above' else (a[2] < b[2])
