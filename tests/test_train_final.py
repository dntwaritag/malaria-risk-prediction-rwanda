from src.train_final import tier_for


def test_tiers():
    assert tier_for(0.0) == "Highest" and tier_for(0.19) == "Highest" and tier_for(0.2) == "High"
    assert tier_for(0.5) == "Medium" and tier_for(0.95) == "Lower"
