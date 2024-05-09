
# First, load relevant columns

# Argotario - Topic + Text | Intended Fallacy
# ElecDeb60to20 - text | fallacy / subcategory
# LogicClimate - source_article | logical_fallacies
# LogicEdu - source_article | updated_label

# Then, map existing labels to MAFALDA (if possible)

# But these datasets are a bit problematic: they not always map cleanly to MAFALDA's labels and sometimes the input text is a description of the fallacy.
# Sanity check: if the label column doesn't match one of the expected labels, then it is likely poorly formatted.
# Also, be sure to deduplicate in the end.
# Compare with MAFALDA's distributions, too