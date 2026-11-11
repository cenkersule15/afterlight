# AFTERLIGHT pilot model card

## Intended use

Educational, non-commercial geospatial ML research. Inspect the limitations of scene-level generalisation. Not operational disaster response, structural inspection or a commercial damage report.

## Models

- Pair U-Net: six-channel RGB-before/RGB-after input, encoder widths 16/32/64, skip connections, five-class semantic output. Trained from scratch with AdamW (learning rate 0.001, weight decay 0.0001), inverse-frequency class-weighted cross-entropy plus 0.5 binary building Dice loss. Horizontal-flip augmentation on training only. Best checkpoint selected by validation foreground macro F1.
- Random Forest: 80 trees, max depth 14, balanced class weights, seed 42, at most 300 pixels per class per training tile. Ten pixel features: before RGB, after RGB, absolute RGB difference and mean-intensity change.

## Data

66 real xBD pairs; at most 8 per named event from the first mirror training shard. This selection is small and not representative. 37 training / 13 validation / 16 test; disjoint entire events. Official challenge test data are not used. Severe damage and some disasters are sparse.

Inputs: 512 px saved scenes, resized again to 128 px for both models. Unknown labels ignored as 255. Background 0; no damage 1; minor 2; major 3; destroyed 4.

## Measured limitations

Bundled held-out damage macro F1 is about 0.103 for CNN and 0.111 for Random Forest. These low values expose domain shift and class imbalance; major-damage recall is especially weak. Prediction overlay appearance must not be mistaken for reliability. Increasing epoch count alone is not evidence of improvement.

The first 14-epoch development attempt had weak validation performance. Class weighting and a building Dice term were then revised. Final training uses 40 epochs. A final test report is bundled. That final test result should not be repeatedly used to tune this same pilot; use fresh external events for subsequent independent evaluation.

Metrics: pooled pixel confusion matrix, per-class F1/IoU, foreground macro F1 and binary-building Dice/IoU. Classes absent from both truth and prediction are excluded from macro average and listed. Not object-level metrics and not official xView2 challenge scoring. Validation/test are small; no confidence intervals or probability calibration are claimed.

Connected components can merge neighbouring buildings or fragment one building. The UI calls them regions and caps the displayed list at 100 components of at least four model pixels. Scores are uncalibrated maxima, not certified probabilities. Uploaded images are resized, not registered; users must provide aligned pairs.

## Next evaluation

Predefine a larger event-stratified dataset, use high-resolution crops and pretrained encoders, tune only on validation events, evaluate on new independent events with uncertainty intervals and object-level metrics. Document any preprocessing and all data licences.
