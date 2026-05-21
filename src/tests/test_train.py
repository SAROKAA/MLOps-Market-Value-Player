from pathlib import Path
import pandas as pd


def test_training_data_exists():
    path = Path("data/processed/final_training.csv")

    assert path.exists()


def test_training_data_not_empty():
    df = pd.read_csv(
        "data/processed/final_training.csv"
    )

    assert len(df) > 0


def test_required_columns():
    df = pd.read_csv(
        "data/processed/final_training.csv"
    )

    required_cols = [
        "market_value_in_eur",
        "goals",
        "assists",
        "games",
        "minutes_played",
        "age"
    ]

    for col in required_cols:
        assert col in df.columns