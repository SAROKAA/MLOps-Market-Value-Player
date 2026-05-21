from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# ==========================================================
# CONFIG
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

SEASON = 2022


# ==========================================================
# LOAD DATA
# ==========================================================

def load_data():
    print("Loading datasets...")

    players_df = pd.read_csv(RAW_DIR / "players.csv")
    clubs_df = pd.read_csv(RAW_DIR / "clubs.csv")
    appearances_df = pd.read_csv(RAW_DIR / "appearances.csv")
    games_df = pd.read_csv(RAW_DIR / "games.csv")
    player_valuations_df = pd.read_csv(
        RAW_DIR / "player_valuations.csv"
    )

    return (
        players_df,
        clubs_df,
        appearances_df,
        games_df,
        player_valuations_df,
    )


# ==========================================================
# PLAYER PREPROCESSING
# ==========================================================

def preprocess_players(players_df, player_valuations_df, appearances_df):

    print("Preprocessing players...")

    now = datetime.now()

    # -------------------------
    # AGE
    # -------------------------

    players_df["date_of_birth"] = pd.to_datetime(
        players_df["date_of_birth"],
        errors="coerce"
    )

    players_df = players_df.dropna(
        subset=["date_of_birth"]
    )

    players_df["age"] = (
        (now - players_df["date_of_birth"])
        .dt.days / 365.25
    )

    players_df["age"] = (
        players_df["age"]
        .round()
        .astype(int)
    )

    # -------------------------
    # CONTRACT REMAINING
    # -------------------------

    players_df["contract_expiration_date"] = pd.to_datetime(
        players_df["contract_expiration_date"],
        errors="coerce"
    )

    players_df["term_days_remaining"] = (
        players_df["contract_expiration_date"]
        - now
    ).dt.days

    # -------------------------
    # PLAYER VALUATION YEAR
    # -------------------------

    player_valuations_df["date"] = pd.to_datetime(
        player_valuations_df["date"],
        errors="coerce"
    )

    player_valuations_df["year"] = (
        player_valuations_df["date"]
        .dt.year
    )

    # -------------------------
    # APPEARANCES YEAR
    # -------------------------

    appearances_df["datetime"] = pd.to_datetime(
        appearances_df["date"],
        errors="coerce"
    )

    appearances_df["year"] = (
        appearances_df["datetime"]
        .dt.year
    )

    print("Player preprocessing complete")

    return (
        players_df,
        player_valuations_df,
        appearances_df
    )


# ==========================================================
# CLUB FEATURES
# ==========================================================

def preprocess_clubs(players_df, clubs_df):

    print("Merging club features...")

    merged_players_df = players_df.drop(
        columns=[
            "club_id",
            "city_of_birth",
            "date_of_birth",
            "first_name",
            "last_name",
            "player_code",
            "image_url",
            "url",
        ],
        errors="ignore"
    )

    club_features = clubs_df[
        [
            "club_id",
            "total_market_value",
            "squad_size"
        ]
    ].copy()

    club_features = club_features.rename(
        columns={
            "total_market_value": "club_value"
        }
    )

    merged_players_df = merged_players_df.merge(
        club_features,
        left_on="current_club_id",
        right_on="club_id",
        how="left"
    )

    print("Club preprocessing complete")

    return merged_players_df


# ==========================================================
# PLAYER STATS
# ==========================================================

def player_stats(player_id, season, df):

    df = df[df["player_id"] == player_id]
    df = df[df["season"] == season]

    if df.empty:
        return {
            "games": 0,
            "goals": 0,
            "assists": 0,
            "minutes_played": 0,
            "goals_for": 0,
            "goals_against": 0,
            "clean_sheet": 0,
            "yellow_cards": 0,
            "red_cards": 0,
        }

    df["goals_for"] = np.where(
        df["home_club_id"] == df["player_club_id"],
        df["home_club_goals"],
        df["away_club_goals"]
    )

    df["goals_against"] = np.where(
        df["home_club_id"] == df["player_club_id"],
        df["away_club_goals"],
        df["home_club_goals"]
    )

    df["clean_sheet"] = np.where(
        df["goals_against"] == 0,
        1,
        0
    )

    grouped = df.groupby(
        ["player_id", "season"],
        as_index=False
    ).agg({
        "goals": "sum",
        "game_id": "nunique",
        "assists": "sum",
        "minutes_played": "sum",
        "goals_for": "sum",
        "goals_against": "sum",
        "clean_sheet": "sum",
        "yellow_cards": "sum",
        "red_cards": "sum",
    })

    row = grouped.iloc[0]

    return {
        "games": row["game_id"],
        "goals": row["goals"],
        "assists": row["assists"],
        "minutes_played": row["minutes_played"],
        "goals_for": row["goals_for"],
        "goals_against": row["goals_against"],
        "clean_sheet": row["clean_sheet"],
        "yellow_cards": row["yellow_cards"],
        "red_cards": row["red_cards"],
    }


# ==========================================================
# APPEARANCES FEATURES
# ==========================================================

def preprocess_appearances(merged_players_df, appearances_df, games_df, season):
    print("Processing appearances...")
    
    # 1. Gabungkan games dan appearances secara massal
    games_and_appearances_df = appearances_df.merge(games_df, on=['game_id'], how='left')

    # 2. Filter data berdasarkan season agar komputasi jauh lebih ringan
    df_season = games_and_appearances_df[games_and_appearances_df['season'] == season].copy()

    if not df_season.empty:
        # 3. Hitung goals_for dan goals_against menggunakan numpy select (Vektor)
        is_home = df_season['home_club_id'] == df_season['player_club_id']
        is_away = df_season['away_club_id'] == df_season['player_club_id']

        df_season["goals_for"] = np.select(
            [is_home, is_away], 
            [df_season['home_club_goals'], df_season['away_club_goals']], 
            default=np.nan
        )

        df_season["goals_against"] = np.select(
            [is_home, is_away], 
            [df_season['away_club_goals'], df_season['home_club_goals']], 
            default=np.nan
        )

        # 4. Hitung clean sheet secara vektor
        df_season['clean_sheet'] = np.where(df_season['goals_against'] == 0, 1, np.where(df_season['goals_against'] > 0, 0, np.nan))

        # 5. Groupby untuk mengagregasi data SELURUH PEMAIN sekaligus
        aggregated_stats = df_season.groupby(['player_id'], as_index=False).agg({
            'goals': 'sum', 
            'game_id': 'nunique', 
            'assists': 'sum', 
            'minutes_played': 'sum', 
            'goals_for': 'sum',
            'goals_against': 'sum', 
            'clean_sheet': 'sum',
            'yellow_cards': 'sum',
            'red_cards': 'sum'
        })

        # Rename game_id menjadi games sesuai struktur data asli kamu (tanpa format season)
        aggregated_stats = aggregated_stats.rename(columns={'game_id': 'games'})

        # 6. Tentukan kolom ID yang ada di merged_players_df (antisipasi kalau namanya player_id atau id)
        player_id_col = 'player_id' if 'player_id' in merged_players_df.columns else merged_players_df.columns[0]

        # 7. Gabungkan hasil agregasi langsung ke merged_players_df
        merged_players_df = merged_players_df.merge(
            aggregated_stats, 
            left_on=player_id_col, 
            right_on='player_id', 
            how='left'
        )

        # Drop duplikasi jika kolom key aslinya bukan bernama 'player_id' (biar tidak ada kolom player_id_x / player_id_y)
        if player_id_col != 'player_id' and 'player_id' in merged_players_df.columns:
            merged_players_df = merged_players_df.drop(columns=['player_id'])

        # 8. Isi nilai NaN dengan 0 untuk pemain yang tidak memiliki record bermain di season tersebut
        cols_to_fill = ['games', 'goals', 'assists', 'minutes_played', 'goals_for', 'goals_against', 'clean_sheet', 'yellow_cards', 'red_cards']
        merged_players_df[cols_to_fill] = merged_players_df[cols_to_fill].fillna(0)
    else:
        print(f"Warning: No appearance data found for season {season}")
        # Buat kolom kosong dengan nilai 0 jika data season kosong
        cols_to_fill = ['games', 'goals', 'assists', 'minutes_played', 'goals_for', 'goals_against', 'clean_sheet', 'yellow_cards', 'red_cards']
        for col in cols_to_fill:
            merged_players_df[col] = 0

    print('Appearance, goal and card data merged successfully!')
    return merged_players_df


# ==========================================================
# MAIN
# ==========================================================

def main():

    (
        players_df,
        clubs_df,
        appearances_df,
        games_df,
        player_valuations_df,
    ) = load_data()

    (
        players_df,
        player_valuations_df,
        appearances_df
    ) = preprocess_players(
        players_df,
        player_valuations_df,
        appearances_df
    )

    merged_players_df = preprocess_clubs(
        players_df,
        clubs_df
    )

    merged_players_df = preprocess_appearances(
        merged_players_df,
        appearances_df,
        games_df,
        SEASON
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    save_path = (
        PROCESSED_DIR
        / "final_training.csv"
    )

    merged_players_df.to_csv(
        save_path,
        index=False
    )

    print(f"Saved to: {save_path}")
    print(merged_players_df.shape)


if __name__ == "__main__":
    main()