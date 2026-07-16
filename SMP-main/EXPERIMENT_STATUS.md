# SMP experiment status

Statistics date: 2026-07-16.  Each setting contains four datasets
(`CUB200`, `CIFAR100`, `ImageNet-R`, and `miniImageNet`), for a total of
16 planned runs.

| Setting | Completed | Status |
| --- | ---: | --- |
| `IN1K-NoShuffle1993` | 4 / 4 | Complete |
| `IN1K-Shuffle1993` | 4 / 4 | Complete |
| `IN1K-Shuffle2025` | 0 / 4 | Not started |
| `IN1K-Shuffle42` | 0 / 4 | Pending |
| **Total** | **8 / 16 (50%)** | **In progress** |

The submitted queue stopped before `IN1K-Shuffle2025` started because its
then-active output path did not exist.  The checked-in runner now writes to
`../logs/<setting>/`; resume the remaining settings with:

```bash
cd SMP-main
CUDA_VISIBLE_DEVICES=0 ./train_smp.sh IN1K-Shuffle2025
CUDA_VISIBLE_DEVICES=0 ./train_smp.sh IN1K-Shuffle42
```

## Completed metrics

Values below are taken from the final `Top1 curve` in each local
`SMP-main/logs/px_d1/**/log.log`.  `Base`, `Last`, and `Avg` are recorded in
[`../FSCIL_Results.xlsx`](../FSCIL_Results.xlsx); `PD` is calculated in the
workbook as `Base - Last`.

| Setting | Dataset | Base | Last | Avg | Top-1 curve |
| --- | --- | ---: | ---: | ---: | --- |
| `IN1K-NoShuffle1993` | CUB200 | 89.33 | 86.17 | 86.78 | 89.33, 88.74, 87.87, 86.98, 86.73, 85.71, 85.58, 85.94, 85.47, 86.04, 86.17 |
| `IN1K-NoShuffle1993` | CIFAR100 | 94.20 | 88.45 | 90.87 | 94.20, 92.63, 92.04, 90.61, 90.70, 90.01, 89.88, 89.33, 88.45 |
| `IN1K-NoShuffle1993` | ImageNet-R | 87.63 | 76.38 | 80.75 | 87.63, 84.89, 84.18, 81.80, 80.71, 80.18, 78.94, 78.22, 77.94, 77.38, 76.38 |
| `IN1K-NoShuffle1993` | miniImageNet | 98.23 | 95.71 | 96.72 | 98.23, 97.98, 96.91, 96.76, 96.81, 96.36, 95.87, 95.82, 95.71 |
| `IN1K-Shuffle1993` | CUB200 | 93.01 | 84.86 | 88.50 | 93.01, 92.33, 90.18, 89.17, 88.54, 87.95, 88.00, 87.06, 86.56, 85.84, 84.86 |
| `IN1K-Shuffle1993` | CIFAR100 | 95.28 | 88.45 | 91.50 | 95.28, 94.11, 94.01, 92.27, 90.61, 90.06, 89.39, 89.34, 88.45 |
| `IN1K-Shuffle1993` | ImageNet-R | 87.30 | 77.00 | 81.48 | 87.30, 85.78, 84.33, 82.81, 82.23, 81.16, 80.13, 78.91, 78.93, 77.69, 77.00 |
| `IN1K-Shuffle1993` | miniImageNet | 97.83 | 94.92 | 96.13 | 97.83, 97.48, 97.30, 95.99, 95.81, 95.52, 95.47, 94.86, 94.92 |

The runner does not report BWT or FWT, so those workbook fields are left
blank rather than inferred from aggregate accuracy.
