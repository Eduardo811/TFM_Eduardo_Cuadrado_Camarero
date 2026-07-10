# Deep-learning registration comparison

## Mean Dice

- Classical baseline: 0.913
- VoxelMorph-style selected: 0.956 (+0.043 vs baseline)
- TransMorph-style selected: 0.932 (+0.020 vs baseline)
- DeeperHistReg approximate: 0.738 (-0.175 vs baseline)

## Conclusion

VoxelMorph-style is the main deep-learning result for the current dataset. TransMorph-style provides a valid transformer-based exploratory comparison but does not outperform VoxelMorph globally. DeeperHistReg is not competitive for the H&E-HSI modality gap. All neural results remain instance-specific proof-of-concept experiments because only six paired specimens are available.

Important: DeeperHistReg Dice is approximate and should not be interpreted as strictly equivalent to the other mask Dice values.
