"""Offline import gate, independent of any dataset, checkpoint or download."""


def test_eff_imports_without_world_model_runtime():
    from tdwm.methods.eff import EffCritic, EffModel, EffSuccessor

    assert EffModel is not None
    assert EffSuccessor is not None
    assert EffCritic is not None
