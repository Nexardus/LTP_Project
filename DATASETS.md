# Potential validation/development datasets and their labels

Argotario (Habernal et al., 2017): Around 1345 data points (how many do we need???)
Fallacies sourced from serious gaming about everyday argumentation
→ Ad hominem, appeal to emotion, hasty generalisation, irrelevant authority, non-fallacious
ElecDeb60to20 (Goffredo et al., 2023): 
Political debates from US presidential candidates from 1960 to 2020; 2020 election not included in MAFALDA
→ Ad hominem, appeal to authority, appeal to emotion, false cause, slippery slope

Riposte! (Reisert et al., 2019): 18,887 data points
Counter-argument generation dataset for fallacious statements, can be repurposed as a fallacy detection dataset
→ Circular reasoning, hasty generalisation, false cause

LogicClimate


# Note that Argotario and LogicClimate are (partially) included in MAFALDA. But this is somewhat fine ish because we don't train our models on that data.
To offset this, we discard samples with labels that are not easily mapped to the MAFALDA labels, to ensure that the used samples are less ambiguous.
Only using e.g., ElecDeb is also problematic because it only contains ~5 unique ish labels.
Label casting is based on Table 3 in the MAFALDA paper.

Overall, the annotations in all datasets are very subjective. This is a fundamental limitation, and is also present in MAFALDA.