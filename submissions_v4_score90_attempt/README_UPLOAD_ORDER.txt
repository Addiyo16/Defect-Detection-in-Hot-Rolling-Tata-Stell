Tata Steel score-90 attempt submissions

These are new rankers, not just thresholds on the old CatBoost probability.
Validation AP improved from about 0.309 in the old rich CatBoost to about 0.43 for HGB/XGB.

Recommended upload order if you have limited attempts:
1. TRY1_rank_hgb_xgb_top60.csv
2. TRY2_rank_hgb_xgb_top75.csv
3. TRY4_xgb_precise_top50.csv
4. TRY3_rank_hgb_xgb_top83_best_oof_f1.csv

If the score increases with more predicted defects, try the BACKUP top100/top120 files.
If the score decreases with more predicted defects, try TRY7_rank_xgb_pair_top50.csv.
