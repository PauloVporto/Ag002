from __future__ import annotations

from src.penguin_model import ISLAND_MAP, SEX_MAP, predict_species, train_model


def ask_float(label: str) -> float:
    return float(input(f"{label}: ").strip())


def ask_choice(label: str, valid_map: dict[str, int]) -> int:
    while True:
        value = input(f"{label} {list(valid_map.keys())}: ").strip()
        if value in valid_map:
            return valid_map[value]
        print("Valor inválido. Tente novamente.")


def main() -> None:
    artifacts = train_model()

    print("Relatório de classificação do modelo:\n")
    print(artifacts.report_text)
    print(f"Acurácia: {artifacts.accuracy * 100:.2f}%\n")

    print("Digite os dados do pinguim para classificação.")
    island = ask_choice("Ilha", ISLAND_MAP)
    sex = ask_choice("Sexo", SEX_MAP)
    culmen_length = ask_float("Culmen length mm")
    culmen_depth = ask_float("Culmen depth mm")
    flipper_length = ask_float("Flipper length mm")
    body_mass = ask_float("Body mass g")

    _, species_name = predict_species(
        artifacts.model,
        island=island,
        sex=sex,
        culmen_length=culmen_length,
        culmen_depth=culmen_depth,
        flipper_length=flipper_length,
        body_mass=body_mass,
    )

    print(f"\nEspécie prevista pelo modelo: {species_name}")


if __name__ == "__main__":
    main()
