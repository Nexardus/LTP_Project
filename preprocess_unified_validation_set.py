"""Preprocess the raw datasets into a unified validation set."""

import numpy as np
import pandas as pd

ARGOTARIO_TO_MAFALDA = {
    "Appeal to Emotion": "Appeal to Emotion",
    "Red Herring": "Fallacy of Logic",
    "Irrelevant Authority": "Appeal to False Authority",
    "Ad Hominem": "Abusive Ad Hominem",
    "Hasty Generalization": "Hasty Generalization",
    "No Fallacy": "No fallacy",
}

ELECDEB_TO_MAFALDA = {
    "AppealtoEmotion": "Appeal to Emotion",
    "Loaded Language": "Appeal to Emotion",
    "Loaded language": "Appeal to Emotion",
    "loaded language": "Appeal to Emotion",
    "Without Evidence": "Fallacy of Credibility",
    "Slogan": "Appeal to Emotion",
    "Slogans": "Appeal to Emotion",
    "Flag waving": "Appeal to Positive Emotion",
    "Name-Calling, Labeling": "Guilt by Association",
    "Name-calling": "Guilt by Association",
    "Appeal to hypocrisy": "Appeal to Emotion",
    "Appeal to pity": "Appeal to Pity",
    "Appeal to fear": "Appeal to Fear",
    "Popular opinion": "Ad Populum",
    "Appeal to popular opinion": "Ad Populum",
    "False Authority": "Appeal to False Authority",
    "Appeal to false authority": "Appeal to False Authority",
    "AppealtoAuthority": "Appeal to False Authority",
    "Appeal to authority without evidence": "Appeal to False Authority",
    "Appeal to Authority without evidence": "Appeal to False Authority",
    "False cause": "False Causality",
    "Slipperyslope": "Slippery Slope",
    "Slippery slope": "Slippery Slope",
    "Tu quoque": "Tu Quoque",
    "Ad hominem": "Abusive Ad Hominem",
    "AdHominem": "Abusive Ad Hominem",
    "Circumstantial Ad hominem": "Abusive Ad Hominem",
    np.nan: "No fallacy",
}

LOGICLIMATE_TO_MAFALDA = {
    "intentional": "Appeal to Emotion",
    "fallacy of credibility": "Fallacy of Credibility",
    "false dilemma": "False Dilemma",
    "appeal to emotion": "Appeal to Emotion",
    "equivocation": "Equivocation",
    "faulty generalization": "Hasty Generalization",
    "fallacy of relevance": "Fallacy of Logic",
    "fallacy of logic": "Fallacy of Logic",
    "ad populum": "Ad Populum",
    "False causality": "False Causality",
    "ad hominem": "Abusive Ad Hominem",
    "fallacy of extension": "Straw Man",
    "circular reasoning": "Circular Reasoning",
}

LOGIEDU_TO_MAFALDA = {
    "intentional": "Appeal to Emotion",
    "fallacy of credibility": "Fallacy of Credibility",
    "false dilemma": "False Dilemma",
    "appeal to emotion": "Appeal to Emotion",
    "equivocation": "Equivocation",
    "faulty generalization": "Hasty Generalization",
    "fallacy of relevance": "Fallacy of Logic",
    "fallacy of logic": "Fallacy of Logic",
    "ad populum": "Ad Populum",
    "false causality": "False Causality",
    "ad hominem": "Abusive Ad Hominem",
    "fallacy of extension": "Straw Man",
    "circular reasoning": "Circular Reasoning",
}

MAFALDA_TO_SUPERLABELS = {
    "Appeal to Emotion": "Appeal to Emotion",
    "Fallacy of Logic": "Fallacy of Logic",
    "Appeal to False Authority": "Fallacy of Credibility",
    "No fallacy": "No fallacy",
    "Abusive Ad Hominem": "Fallacy of Credibility",
    "Hasty Generalization": "Fallacy of Logic",
    "Fallacy of Credibility": "Fallacy of Credibility",
    "Appeal to Pity": "Appeal to Emotion",
    "Appeal to Fear": "Appeal to Emotion",
    "Ad Populum": "Fallacy of Credibility",
    "False Causality": "Fallacy of Logic",
    "Slippery Slope": "Fallacy of Logic",
    "Appeal to Positive Emotion": "Appeal to Emotion",
    "Tu Quoque": "Fallacy of Credibility",
    "Guilt by Association": "Fallacy of Credibility",
    "False Dilemma": "Fallacy of Logic",
    "Equivocation": "Fallacy of Logic",
    "Straw Man": "Fallacy of Logic",
    "Circular Reasoning": "Fallacy of Logic",
}


def main() -> None:
    """Run the main function."""
    argotario = pd.read_csv("datasets/Argotario.tsv", sep="\t")
    elecdeb = pd.read_csv("datasets/ElecDeb60to20.csv")
    logicclimate = pd.read_csv("datasets/LogicClimate.csv")
    logicedu = pd.read_csv("datasets/LogicEdu.csv")

    argotario["MAFALDA Label"] = argotario["Intended Fallacy"].map(ARGOTARIO_TO_MAFALDA)
    elecdeb["MAFALDA Label"] = elecdeb["subcategory"].map(ELECDEB_TO_MAFALDA)
    logicclimate["MAFALDA Label"] = logicclimate["logical_fallacies"].map(LOGICLIMATE_TO_MAFALDA)
    logicedu["MAFALDA Label"] = logicedu["updated_label"].map(LOGIEDU_TO_MAFALDA)

    argotario["Input"] = argotario["Topic"] + " " + argotario["Text"]
    elecdeb["Input"] = elecdeb["text"]
    logicclimate["Input"] = logicclimate["source_article"]
    logicedu["Input"] = logicedu["source_article"]

    unified = pd.concat([argotario, elecdeb, logicclimate, logicedu], ignore_index=True)
    unified = unified[["Input", "MAFALDA Label"]]
    unified = unified.dropna(subset=["MAFALDA Label", "Input"]).sort_values(by=["Input"]).drop_duplicates()

    unified["MAFALDA Superlabel"] = unified["MAFALDA Label"].map(MAFALDA_TO_SUPERLABELS)
    unified["MAFALDA Label"] = unified["MAFALDA Label"].replace(
        ["Appeal to Emotion", "Fallacy of Credibility", "Fallacy of Logic"], None
    )

    print(unified)
    unified.groupby("MAFALDA Label").sample(n=20, random_state=42).to_csv(
        "cleaned_datasets/unified_validation_set_downsampled.tsv", index=False, sep="\t"
    )
    unified.to_csv("cleaned_datasets/unified_validation_set.tsv", index=False, sep="\t")


if __name__ == "__main__":
    main()
