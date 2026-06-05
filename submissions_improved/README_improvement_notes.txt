Improved Tata Steel submission candidates
Current problem: final_best_submission.csv predicts 223/339 defects (65.8%), caused by threshold 0.0092. Training defect rate is only 66/1352 (4.9%), so this creates too many false positives.
Recommended first upload: RECOMMENDED_top48_precision_recall.csv (48 defects, 14.2%).
If leaderboard says precision is still low: try backup_top34_high_precision.csv.
If leaderboard says recall is too low: try backup_top60_more_recall.csv.

Files created:
- RECOMMENDED_top48_precision_recall.csv: top 48 coils, threshold >= 0.090529
- backup_top34_high_precision.csv: top 34 coils, threshold >= 0.212955
- backup_top60_more_recall.csv: top 60 coils, threshold >= 0.070609
