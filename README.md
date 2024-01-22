# EXPATNET
Code for "EXPATNET: EXPLAINABLE ATTACK FOR HIGH-DEFINITION ADVERSARIAL EXAMPLES" 
![Uploading Framework.png…]()

Abstract: With the prevalence of deep learning (DL) applications, there is a growing concern regarding the vulnerabilities of DL-based models. The intentional perturbations to clean images can generate adversarial examples (AEs), resulting in undesirable consequences. Previous adversarial attacks solely prioritize attack success rates (ASRs), leading to global divergence of AEs. To fill the niche market, we propose an interpretability-guided localized adversarial attack network called ExpATNet. The ExpATNet transforms the disturbance space from the spatial to a frequency domain and drops features to improve the imperceptibility of AEs. The explainable artificial intelligence (XAI) enables class activation maps (CAM) to guide the attack region, which reduces the attack surface. Extensive experiments on three benchmark datasets show our approach can parameterize perceptually aligned adversaries while achieving state-of-the-art attack performance. 

## Requirements

* python ==3.8.18
* torch ==2.1.0
* torchvision ==0.16.0
* torchattacks ==3.5.1
* numpy ==1.24.4
* Pillow ==10.0.1
  

## Datasets

We evaluate the performance of the ExpATNet on three classic datasets: ImageNet, Oxford-IIIT Pets, and Caltech-256 datasets.

## Experiments

We provide four perceptual metrics to measure imperceptibility, including PSNR, SSIM, LPIPS, and A-DISTS. 

### Quick Start

```python
python EXPATNET.py
```

## Examples

![图片质量放大](https://github.com/huangqiangbo/EXPATNET/assets/63629128/b06d14bf-d1bf-49ff-b5d2-80d8ba639583)

Here we offer some experiment results. You can get more results in our paper.

## 

