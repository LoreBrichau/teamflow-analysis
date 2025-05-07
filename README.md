Purpose of the Script

This Python script was developed to preprocess, merge and analyze data from the team flow study. Specifically, it combines self-reported Team Flow Monitor scores with observed behavioral indicators and calculates bootstrapped Spearman correlations to test exploratory hypotheses.

Overview of Steps

Gini Coefficient Calculation

Used to quantify inequality in speaking time across team members (i.e., equal participation).
Gini values were computed for each team using speaker proportion data.

Data Extraction and Restructuring

Imported self-report data from the Team Flow Monitor (TFM) Excel file.
Averaged scores across raters per team for each of the seven TFM prerequisites.
Created a nested dictionary with team × dimension structures for cleaner access.
Manually removed one team (Team 12) from the analysis due to missing or unusable audio.

Observed Behavioral Indicators

Imported coded Vosaic output from Excel (Teamobservaties.Output.VOSAIC).
Extracted six observed behavioral variables per team:
I/We ratio
Proportion of silence
Number of questions asked
Number of speaking turns
Number of overlapping speech bursts
Number of interruptions
Also included: Gini coefficient as a proxy for equal communication.

Team Performance Data

A weighted error score (lower = better performance) was manually added for each team.

Data Matching

All variables were aligned by team, with data from TFM, observed indicators and performance organized in parallel lists.

Bootstrapped Spearman Correlation

A custom function (bootstrap_spearman) was written to compute non-parametric Spearman correlations with bootstrapped 90% confidence intervals (n=2000 iterations).
This method was chosen for its robustness to small samples and non-normality.
Correlations were only computed for theoretically relevant variable pairs, based on the study’s exploratory hypotheses.
