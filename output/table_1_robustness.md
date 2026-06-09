# Table 1: Robustness Comparison under Legitimate Operations

| Legitimate Operation | Authentication Method | Avg NCC | Avg BER | Status |
| --- | --- | --- | --- | --- |
| JPEG 90 | ViT-S (MantiQ) | 0.9959 | 0.0320 | ✅ Pass |
| JPEG 90 | ResNet-18 | 0.9883 | 0.0534 | ✅ Pass |
| JPEG 90 | SURF+SVD | 0.3979 | 0.3010 | ❌ Fail |
| JPEG 90 | Fragile Hash | N/A | N/A | ❌ Fail (Acc=0.0%) |
| JPEG 70 | ViT-S (MantiQ) | 0.9474 | 0.1166 | ✅ Pass |
| JPEG 70 | ResNet-18 | 0.9519 | 0.1027 | ✅ Pass |
| JPEG 70 | SURF+SVD | 0.2820 | 0.3590 | ❌ Fail |
| JPEG 70 | Fragile Hash | N/A | N/A | ❌ Fail (Acc=0.0%) |
| JPEG 50 | ViT-S (MantiQ) | 0.8855 | 0.1767 | ❌ Fail |
| JPEG 50 | ResNet-18 | 0.9220 | 0.1321 | ✅ Pass |
| JPEG 50 | SURF+SVD | 0.2542 | 0.3729 | ❌ Fail |
| JPEG 50 | Fragile Hash | N/A | N/A | ❌ Fail (Acc=0.0%) |
| Gaussian Noise | ViT-S (MantiQ) | 0.7657 | 0.2372 | ❌ Fail |
| Gaussian Noise | ResNet-18 | 0.8879 | 0.1778 | ❌ Fail |
| Gaussian Noise | SURF+SVD | 0.2829 | 0.3585 | ❌ Fail |
| Gaussian Noise | Fragile Hash | N/A | N/A | ❌ Fail (Acc=0.0%) |
| Rotation 2° | ViT-S (MantiQ) | 0.9725 | 0.0854 | ✅ Pass |
| Rotation 2° | ResNet-18 | 0.9794 | 0.0703 | ✅ Pass |
| Rotation 2° | SURF+SVD | 0.4118 | 0.2941 | ❌ Fail |
| Rotation 2° | Fragile Hash | N/A | N/A | ❌ Fail (Acc=0.0%) |
| Brightness 1.1 | ViT-S (MantiQ) | 0.8674 | 0.1877 | ❌ Fail |
| Brightness 1.1 | ResNet-18 | 0.8786 | 0.1775 | ❌ Fail |
| Brightness 1.1 | SURF+SVD | 0.1333 | 0.4403 | ❌ Fail |
| Brightness 1.1 | Fragile Hash | N/A | N/A | ❌ Fail (Acc=0.0%) |
| Contrast 0.9 | ViT-S (MantiQ) | 0.9832 | 0.0582 | ✅ Pass |
| Contrast 0.9 | ResNet-18 | 0.9892 | 0.0470 | ✅ Pass |
| Contrast 0.9 | SURF+SVD | 0.3852 | 0.3074 | ❌ Fail |
| Contrast 0.9 | Fragile Hash | N/A | N/A | ❌ Fail (Acc=0.0%) |
| Cropping 5% | ViT-S (MantiQ) | 0.9385 | 0.1181 | ✅ Pass |
| Cropping 5% | ResNet-18 | 0.8500 | 0.1810 | ❌ Fail |
| Cropping 5% | SURF+SVD | 0.1965 | 0.4018 | ❌ Fail |
| Cropping 5% | Fragile Hash | N/A | N/A | ❌ Fail (Acc=0.0%) |
