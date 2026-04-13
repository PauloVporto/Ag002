from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "penguins.csv"

ISLAND_MAP = {"Biscoe": 0, "Dream": 1, "Torgersen": 2}
SEX_MAP = {"FEMALE": 0, "MALE": 1}
SPECIES_MAP = {"Adelie": 0, "Adeline": 0, "Chinstrap": 1, "Gentoo": 2}
SPECIES_LABELS = {0: "Adeline", 1: "Chinstrap", 2: "Gentoo"}

DISPLAY_COLUMNS = [
    "species",
    "island",
    "culmen length mm",
    "culmen depth mm",
    "flipper length mm",
    "body mass g",
    "sex",
]

ORDERED_COLUMNS = [
    "island",
    "sex",
    "culmen length mm",
    "culmen depth mm",
    "flipper length mm",
    "body mass g",
    "species",
]

FEATURE_COLUMNS = ORDERED_COLUMNS[:-1]


@dataclass
class TrainingArtifacts:
    data: pd.DataFrame
    processed_data: pd.DataFrame
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    predictions: pd.Series
    model: DecisionTreeClassifier
    accuracy: float
    report_text: str


def load_raw_data(path: Path = DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.dropna().copy()
    cleaned = cleaned.rename(
        columns={
            "bill_length_mm": "culmen length mm",
            "bill_depth_mm": "culmen depth mm",
            "flipper_length_mm": "flipper length mm",
            "body_mass_g": "body mass g",
        }
    )

    cleaned["island"] = cleaned["island"].map(ISLAND_MAP).astype("int64")
    cleaned["sex"] = cleaned["sex"].map(SEX_MAP).astype("int64")
    cleaned["species"] = cleaned["species"].map(SPECIES_MAP).astype("int64")

    cleaned = cleaned.reindex(columns=DISPLAY_COLUMNS)
    cleaned = cleaned.reindex(columns=ORDERED_COLUMNS)
    return cleaned


def train_model(test_size: float = 0.2, random_state: int = 42) -> TrainingArtifacts:
    raw_df = load_raw_data()
    processed_df = preprocess_data(raw_df)

    x = processed_df[FEATURE_COLUMNS]
    y = processed_df["species"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = DecisionTreeClassifier(max_depth=5, random_state=random_state)
    model.fit(x_train, y_train)

    predictions = pd.Series(model.predict(x_test), index=y_test.index)
    accuracy = accuracy_score(y_test, predictions)
    report_text = classification_report(
        y_test,
        predictions,
        target_names=[SPECIES_LABELS[0], SPECIES_LABELS[1], SPECIES_LABELS[2]],
        zero_division=0,
    )

    return TrainingArtifacts(
        data=raw_df,
        processed_data=processed_df,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        predictions=predictions,
        model=model,
        accuracy=accuracy,
        report_text=report_text,
    )


def predict_species(
    model: DecisionTreeClassifier,
    island: int,
    sex: int,
    culmen_length: float,
    culmen_depth: float,
    flipper_length: float,
    body_mass: float,
) -> tuple[int, str]:
    input_df = pd.DataFrame(
        [
            {
                "island": island,
                "sex": sex,
                "culmen length mm": culmen_length,
                "culmen depth mm": culmen_depth,
                "flipper length mm": flipper_length,
                "body mass g": body_mass,
            }
        ]
    )
    prediction = int(model.predict(input_df)[0])
    return prediction, SPECIES_LABELS[prediction]
