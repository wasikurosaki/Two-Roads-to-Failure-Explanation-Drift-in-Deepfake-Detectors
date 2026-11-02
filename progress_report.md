# Progress Report: Two Roads to Failure? Explanation Drift in Deepfake Detectors
**CMPT 819** · Washikur Rahman (ozj079), Tahsina Moiukh (rnk048)

---

## 1. Goal

Deepfake detectors usually fail in one of two ways:

1. **Unseen generator:** the fake was made by a method the detector never saw during training.
2. **Degraded image:** the image was blurred, resized, compressed, or made noisy, for example by a social-media upload.

We want to know whether these two failures are related, and **whether they make each other worse when they happen together**. Later trials (Trials 3–4) will look at *where* the detector looks (Grad-CAM) to explain the failures. This report covers the setup, **Trial 1** (how good the detectors are in normal conditions), and **Trial 2** (how much each failure, and both together, hurt the detectors).

---

## 2. Setup (short)

- **Dataset:** DeepFakeFace. It has real photos (IMDB-WIKI) and three fake generators:
  - **text2img:** Stable Diffusion 1.5 creates a whole new image from a text prompt.
  - **inpainting:** Stable Diffusion repaints only the face of a real photo.
  - **insight:** InsightFace swaps another person's face into a real photo.
- **Size:** 25,000 images per class, split 70% train, 20% test, 10% validation, **by person** (no person appears in two splits). Each test comparison uses **4,984 real and 4,984 fake images**.
- **Preprocessing:** faces are cropped to 320 px. Fakes are resized to their real partner's resolution, and every image gets the same JPEG save (Q = 95). This removes the most obvious shortcut: all fakes in the raw data are 512 px, while real photos vary in size.
- **Models:** **ResNet-18** (11 M parameters, 224 px input) and **Xception** (21 M parameters, 299 px input), both pretrained on ImageNet. Each outputs one score: the probability that the image is fake.
- **Training:** AdamW, learning rate 1e-4, batch size 16, mixed precision. The best epoch is kept based on validation AUC.
- **Decision threshold:** chosen once on *clean validation* images and kept fixed for every test. A model therefore cannot adjust its threshold to the test images.

### Metrics used

| Metric | Meaning | Good value |
|---|---|---|
| **AUC** | How well the scores rank fakes above real images, independent of any threshold | 1.0 = perfect, 0.5 = guessing |
| **Accuracy** | Share of all test images labelled correctly | high |
| **Precision** | Of the images called *fake*, the share that really are fake | high |
| **Recall** | Of the real fakes, the share that were caught | high |
| **F1** | One number that balances precision and recall | high |
| **FPR** (false-positive rate) | Share of *real* photos wrongly called fake | low |

AUC shows whether the model *can* separate real from fake. Accuracy, precision, recall, F1 and FPR show how it behaves *with the fixed threshold*, which is what a real user would experience.

---

## 3. Trial 1: Baseline (how good are the detectors in normal conditions?)

### What we did
We trained **8 detectors**. For each architecture there is one model per generator (*single-text2img*, *single-inpainting*, *single-insight*) and one model trained on **all three** generators (*all*). Each model was tested on clean test images from every generator.

### 3.1 Results on the generators the model was trained on

| Model | Test generator | AUC [95% CI] | Acc | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|---|---|
| ResNet-18 single-text2img | text2img | 0.9995 [0.9993–0.9997] | 0.989 | 0.996 | 0.981 | 0.988 | 0.004 |
| ResNet-18 single-inpainting | inpainting | 0.9977 [0.9971–0.9981] | 0.976 | 0.971 | 0.982 | 0.976 | 0.030 |
| ResNet-18 single-insight | insight | 0.9933 [0.9921–0.9943] | 0.960 | 0.970 | 0.950 | 0.960 | 0.030 |
| ResNet-18 all | text2img | 0.9970 [0.9965–0.9976] | 0.971 | 0.959 | 0.985 | 0.972 | 0.042 |
| ResNet-18 all | inpainting | 0.9918 [0.9906–0.9929] | 0.954 | 0.957 | 0.949 | 0.953 | 0.042 |
| ResNet-18 all | insight | 0.9869 [0.9852–0.9885] | 0.945 | 0.957 | 0.932 | 0.944 | 0.042 |
| Xception single-text2img | text2img | 0.9998 [0.9997–0.9999] | 0.994 | 0.995 | 0.994 | 0.994 | 0.005 |
| Xception single-inpainting | inpainting | 0.9993 [0.9991–0.9995] | 0.986 | 0.993 | 0.980 | 0.986 | 0.007 |
| Xception single-insight | insight | 0.9965 [0.9957–0.9972] | 0.973 | 0.973 | 0.973 | 0.973 | 0.028 |
| Xception all | text2img | 0.9991 [0.9987–0.9994] | 0.985 | 0.979 | 0.992 | 0.986 | 0.022 |
| Xception all | inpainting | 0.9978 [0.9972–0.9984] | 0.980 | 0.978 | 0.981 | 0.980 | 0.022 |
| Xception all | insight | 0.9946 [0.9936–0.9955] | 0.963 | 0.978 | 0.948 | 0.963 | 0.022 |

**What this means**
- On generators they have seen, both detectors are **almost perfect**: AUC 0.987–0.9998, accuracy 94–99%, and only 0.4–4% of real photos falsely flagged.
- **Xception is better than ResNet-18 in every row**, with fewer false alarms (FPR 2.2% vs 4.2% for the *all* models).
- **InsightFace is the hardest generator** for both models, and **text2img is the easiest**.
- Training on all three generators costs only a little accuracy on each generator compared with a specialised model.

### 3.2 Results on generators the model was *not* trained on (single-generator models)

| Model | → text2img | → inpainting | → insight |
|---|---|---|---|
| ResNet-18 single-text2img | *(trained)* | AUC 0.897 · Recall 0.262 · F1 0.414 | AUC **0.617** · Recall **0.011** · F1 0.023 |
| ResNet-18 single-inpainting | AUC 0.988 · Recall 0.901 · F1 0.933 | *(trained)* | AUC 0.929 · Recall 0.645 · F1 0.770 |
| ResNet-18 single-insight | AUC 0.700 · Recall 0.119 · F1 0.207 | AUC 0.912 · Recall 0.639 · F1 0.766 | *(trained)* |
| Xception single-text2img | *(trained)* | AUC 0.932 · Recall 0.416 · F1 0.585 | AUC **0.646** · Recall **0.030** · F1 0.057 |
| Xception single-inpainting | AUC 0.994 · Recall 0.848 · F1 0.914 | *(trained)* | AUC 0.930 · Recall 0.444 · F1 0.611 |
| Xception single-insight | AUC 0.743 · Recall 0.123 · F1 0.213 | AUC 0.939 · Recall 0.699 · F1 0.810 | *(trained)* |

FPR is the same in every row of a model as in its Section 3.1 row, because the real test photos are the same.

**What this means**
- A detector trained on one generator **does not transfer well** to another. The worst case is text2img → insight: AUC 0.62–0.65, and **only 1–3% of face swaps are caught**.
- **Precision stays high** (0.74–0.99) but **recall collapses**. The model rarely raises a false alarm; it simply *misses* the new kind of fake and calls it real.
- **Inpainting transfers best.** It sits between the other two: it is a diffusion model like text2img, and it edits only the face like insight.
- **Why the near-perfect scores in 3.1 are trustworthy:** if the models were using a shortcut shared by all fakes, such as file size or compression, they would also score high on the other generators. They don't. In addition, AUC stays ≥ 0.991 on pairs whose resolution was matched exactly. The models are learning **generator-specific traces**.

---

## 4. Trial 2: Unseen generator × image degradation

### 4.1 Design

**Leave-one-generator-out (LOGO).** For each architecture we trained 3 more models. Each one is trained on **two** generators and never sees the third:

| Model | Trained on | Never sees (unseen) |
|---|---|---|
| logo-text2img | inpainting, insight | text2img |
| logo-inpainting | text2img, insight | inpainting |
| logo-insight | text2img, inpainting | insight |

Each test generator is therefore scored by one model that has **never seen it** and two models that **have seen it**. All six models reached validation AUC 0.990–0.9995. Validation contains only the generators the model trained on, so this says nothing yet about the unseen generator.

**Degradations.** Every test image was also scored under 15 degraded versions, with severity 1 = light and 3–4 = strong:

| Type | Levels | What it simulates |
|---|---|---|
| blur | σ = 1, 2, 3 | out-of-focus or soft images |
| resize | ×0.5, ×0.25 (then back) | thumbnails, low-resolution copies |
| noise | σ = 5, 10, 20 | grainy, low-light images |
| jpeg | quality 90, 70, 50, 30 | compression |
| social | 3 levels (resize + sharpen + JPEG) | a social-media upload |

**The two-roads comparison (2×2).** For each generator and each damage type at medium strength, we compare four AUCs:

|  | Clean | Degraded |
|---|---|---|
| **Seen generator** | ① baseline | ③ damage only |
| **Unseen generator** | ② new generator only | ④ **both** |

- Drop from the unseen generator = ① − ②
- Drop from degradation = ① − ③
- Drop from both = ① − ④
- **Interaction** = (drop from both) − (drop unseen + drop degradation)

A **positive interaction** means that the two problems together hurt *more than the sum* of each alone, so they compound.

### 4.2 Road 1: unseen generator (clean images)

| Model | Test generator | Status | AUC [95% CI] | Acc | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|---|---|---|
| ResNet-18 | text2img | seen (avg of 2) | 0.998 | 0.976 | 0.972 | 0.981 | 0.976 | 0.028 |
| ResNet-18 | text2img | **UNSEEN** | **0.969** [0.966–0.972] | 0.886 | 0.947 | 0.818 | 0.878 | 0.046 |
| ResNet-18 | inpainting | seen (avg of 2) | 0.994 | 0.962 | 0.965 | 0.958 | 0.962 | 0.034 |
| ResNet-18 | inpainting | **UNSEEN** | **0.969** [0.966–0.972] | 0.888 | 0.961 | 0.809 | 0.879 | 0.033 |
| ResNet-18 | insight | seen (avg of 2) | 0.989 | 0.950 | 0.960 | 0.939 | 0.949 | 0.040 |
| ResNet-18 | insight | **UNSEEN** | **0.914** [0.908–0.919] | 0.758 | 0.959 | **0.539** | 0.690 | 0.023 |
| Xception | text2img | seen (avg of 2) | 0.999 | 0.990 | 0.988 | 0.992 | 0.990 | 0.012 |
| Xception | text2img | **UNSEEN** | **0.985** [0.983–0.987] | 0.920 | 0.975 | 0.863 | 0.916 | 0.022 |
| Xception | inpainting | seen (avg of 2) | 0.998 | 0.984 | 0.984 | 0.985 | 0.984 | 0.016 |
| Xception | inpainting | **UNSEEN** | **0.986** [0.985–0.988] | 0.913 | 0.983 | 0.841 | 0.907 | 0.014 |
| Xception | insight | seen (avg of 2) | 0.995 | 0.964 | 0.981 | 0.946 | 0.963 | 0.018 |
| Xception | insight | **UNSEEN** | **0.910** [0.904–0.915] | 0.709 | 0.976 | **0.428** | 0.595 | 0.010 |

**What this means**
- Every generator becomes harder when it is unseen. **The main loss is in recall**: precision and FPR barely change, but the model starts calling new fakes "real".
- For text2img and inpainting the drop is moderate: AUC falls by 0.01–0.03, and recall falls from about 0.97 to 0.81–0.86.
- **InsightFace is the weak spot.** Unseen, AUC falls to about 0.91 and **only 43–54% of face swaps are caught**. Diffusion fakes (text2img, inpainting) share traces with each other, but a face swap leaves very different traces.
- All 12 seen-vs-unseen comparisons are statistically significant (paired DeLong test, Holm-corrected, all p < 10⁻⁵⁰). The size of the drop (ΔAUC) is −0.012 to −0.030 for the diffusion generators and **−0.075 to −0.085 for InsightFace**.

### 4.3 Road 2: image degradation (seen vs unseen)

The tables below average over all three generators. "Seen" is the average over the models that saw the generator; "unseen" is the model that did not. FPR is the same in both columns because it is measured only on real photos, which are the same for every generator.

**ResNet-18**

| Condition | AUC seen | AUC unseen | Acc seen | Acc unseen | Recall seen | Recall unseen | F1 seen | F1 unseen | FPR |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.994 | 0.951 | 0.963 | 0.844 | 0.959 | 0.722 | 0.962 | 0.816 | 0.034 |
| blur@1 | 0.981 | 0.910 | 0.869 | 0.808 | 0.985 | 0.864 | 0.883 | 0.813 | 0.247 |
| blur@2 | 0.945 | 0.842 | 0.701 | 0.664 | 0.990 | 0.916 | 0.770 | 0.730 | **0.587** |
| blur@3 | 0.923 | 0.809 | 0.665 | 0.630 | 0.988 | 0.919 | 0.747 | 0.711 | **0.659** |
| resize@1 | 0.990 | 0.935 | 0.941 | 0.850 | 0.974 | 0.793 | 0.943 | 0.833 | 0.092 |
| resize@2 | 0.977 | 0.896 | 0.878 | 0.805 | 0.979 | 0.834 | 0.889 | 0.805 | 0.223 |
| noise@1 | 0.986 | 0.928 | 0.901 | 0.721 | 0.809 | 0.449 | 0.891 | 0.605 | 0.006 |
| noise@2 | 0.956 | 0.890 | 0.734 | 0.600 | 0.470 | **0.202** | 0.638 | 0.311 | 0.002 |
| noise@3 | 0.883 | 0.833 | 0.552 | 0.530 | 0.104 | **0.061** | 0.186 | 0.105 | 0.000 |
| jpeg@1 | 0.993 | 0.948 | 0.961 | 0.832 | 0.949 | 0.692 | 0.960 | 0.797 | 0.027 |
| jpeg@2 | 0.992 | 0.943 | 0.959 | 0.837 | 0.950 | 0.704 | 0.959 | 0.804 | 0.031 |
| jpeg@3 | 0.990 | 0.936 | 0.948 | 0.817 | 0.921 | 0.659 | 0.946 | 0.776 | 0.025 |
| jpeg@4 | 0.984 | 0.923 | 0.915 | 0.777 | 0.848 | 0.572 | 0.908 | 0.715 | 0.019 |
| social@1 | 0.991 | 0.939 | 0.954 | 0.845 | 0.958 | 0.740 | 0.954 | 0.820 | 0.049 |
| social@2 | 0.984 | 0.920 | 0.930 | 0.790 | 0.889 | 0.609 | 0.927 | 0.740 | 0.028 |
| social@3 | 0.959 | 0.876 | 0.841 | 0.702 | 0.702 | 0.425 | 0.814 | 0.568 | 0.020 |

**Xception**

| Condition | AUC seen | AUC unseen | Acc seen | Acc unseen | Recall seen | Recall unseen | F1 seen | F1 unseen | FPR |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.998 | 0.960 | 0.979 | 0.848 | 0.974 | 0.711 | 0.979 | 0.806 | 0.016 |
| blur@1 | 0.986 | 0.915 | 0.887 | 0.822 | 0.989 | 0.858 | 0.898 | 0.820 | 0.214 |
| blur@2 | 0.938 | 0.834 | 0.650 | 0.624 | 0.994 | 0.940 | 0.740 | 0.713 | **0.693** |
| blur@3 | 0.869 | 0.748 | 0.555 | 0.541 | 0.995 | 0.967 | 0.691 | 0.678 | **0.884** |
| resize@1 | 0.993 | 0.928 | 0.954 | 0.849 | 0.980 | 0.771 | 0.955 | 0.821 | 0.072 |
| resize@2 | 0.975 | 0.888 | 0.882 | 0.806 | 0.975 | 0.825 | 0.892 | 0.802 | 0.212 |
| noise@1 | 0.969 | 0.882 | 0.727 | 0.582 | 0.454 | 0.165 | 0.617 | 0.270 | 0.001 |
| noise@2 | 0.932 | 0.831 | 0.574 | 0.522 | 0.149 | **0.043** | 0.243 | 0.080 | 0.000 |
| noise@3 | 0.858 | 0.765 | 0.509 | 0.501 | 0.019 | **0.002** | 0.036 | 0.005 | 0.000 |
| jpeg@1 | 0.997 | 0.951 | 0.973 | 0.808 | 0.957 | 0.627 | 0.973 | 0.746 | 0.011 |
| jpeg@2 | 0.996 | 0.940 | 0.970 | 0.823 | 0.964 | 0.670 | 0.970 | 0.773 | 0.023 |
| jpeg@3 | 0.991 | 0.920 | 0.951 | 0.798 | 0.935 | 0.628 | 0.951 | 0.739 | 0.033 |
| jpeg@4 | 0.980 | 0.890 | 0.918 | 0.763 | 0.878 | 0.570 | 0.914 | 0.692 | 0.043 |
| social@1 | 0.995 | 0.937 | 0.968 | 0.827 | 0.963 | 0.680 | 0.968 | 0.779 | 0.027 |
| social@2 | 0.980 | 0.877 | 0.908 | 0.726 | 0.846 | 0.483 | 0.900 | 0.615 | 0.031 |
| social@3 | 0.938 | 0.812 | 0.854 | 0.707 | 0.774 | 0.479 | 0.841 | 0.598 | 0.066 |

Precision stays between 0.53 and 1.0 throughout. It is lowest under blur (about 0.52–0.63, because of the many false alarms) and close to 1.0 under noise (because almost nothing is flagged).

**What this means**
- **AUC falls as damage gets stronger, and the unseen column is always lower.** At every damage level the unseen generator scores 0.04–0.12 AUC below the seen ones.
- **Blur and noise break the detectors in opposite directions**, which AUC alone does not show:
  - **Blur → "everything is fake."** Recall stays high (0.92–0.99), but **59–88% of real photos are falsely flagged**. Blurring seems to remove fine detail that the model associates with real photos.
  - **Noise → "everything is real."** FPR drops to 0, but **recall collapses**. At strong noise Xception catches 0.2% of unseen fakes and ResNet-18 catches 6%. The added grain hides the generator's traces.
- **JPEG and resize are mild.** Even JPEG quality 30 costs only 0.01–0.02 AUC on seen generators. JPEG does, however, cut recall on unseen fakes a lot: Xception falls from 0.71 to 0.57.
- **Social-media processing** sits in between, getting steadily worse with level. At level 3 about half of unseen fakes are missed.
- **Which damage is worst depends on the model:** blur for ResNet-18, noise for Xception.
- **Statistics:** 266 of the 270 clean-vs-degraded comparisons are significant (paired DeLong, Holm-corrected). The 4 that are not are all JPEG quality 90, where the change is about 0.000.

### 4.4 Both roads at once: the 2×2 result

**Average over both models, 3 generators and 5 damage types (medium strength):**

| | Clean | Degraded |
|---|---|---|
| **Seen** | AUC 0.996 | drop **0.026** |
| **Unseen** | drop **0.040** | drop **0.104** |

If the two problems simply added up, the combined drop would be 0.040 + 0.026 = **0.066**. The real combined drop is **0.104**, so the **interaction is +0.038**. Positive in 28 of 30 cases; Wilcoxon signed-rank p = 1.3 × 10⁻⁷.

**By architecture**

| Model | Seen clean | Unseen clean | Seen degraded | Unseen degraded | Drop unseen | Drop degradation | Drop both | Interaction |
|---|---|---|---|---|---|---|---|---|
| ResNet-18 | 0.994 | 0.951 | 0.973 | 0.905 | 0.043 | 0.020 | 0.089 | **+0.025** |
| Xception | 0.998 | 0.960 | 0.967 | 0.878 | 0.037 | 0.031 | 0.119 | **+0.051** |

**By damage type (average AUC drops)**

| Model | Type | Drop unseen | Drop degradation | Drop both | Interaction |
|---|---|---|---|---|---|
| ResNet-18 | blur@2 | 0.043 | 0.049 | **0.151** | +0.060 |
| ResNet-18 | noise@2 | 0.043 | 0.037 | 0.104 | +0.024 |
| ResNet-18 | social@2 | 0.043 | 0.009 | 0.074 | +0.021 |
| ResNet-18 | resize@1 | 0.043 | 0.004 | 0.059 | +0.012 |
| ResNet-18 | jpeg@3 | 0.043 | 0.004 | 0.057 | +0.011 |
| Xception | noise@2 | 0.037 | 0.065 | **0.166** | +0.064 |
| Xception | blur@2 | 0.037 | 0.060 | 0.164 | +0.067 |
| Xception | social@2 | 0.037 | 0.018 | 0.120 | +0.065 |
| Xception | jpeg@3 | 0.037 | 0.007 | 0.078 | +0.034 |
| Xception | resize@1 | 0.037 | 0.005 | 0.069 | +0.027 |

**By generator (average AUC drops)**

| Model | Generator | Drop unseen | Drop degradation | Drop both | Interaction |
|---|---|---|---|---|---|
| ResNet-18 | text2img | 0.029 | 0.011 | 0.071 | +0.031 |
| ResNet-18 | inpainting | 0.025 | 0.019 | 0.061 | +0.017 |
| ResNet-18 | insight | 0.075 | 0.031 | **0.135** | +0.028 |
| Xception | text2img | 0.015 | 0.020 | 0.117 | **+0.082** |
| Xception | inpainting | 0.012 | 0.027 | 0.061 | +0.022 |
| Xception | insight | 0.085 | 0.046 | **0.181** | +0.050 |

**A worked example** (Xception, InsightFace, blur σ = 2):
- Seen and clean: AUC 0.995.
- Unseen only: 0.910 (−0.085).
- Blur only: 0.882 (−0.113).
- **Unseen + blur: 0.708 (−0.287)**, against −0.198 if the drops just added up. That is an interaction of +0.089.

At AUC 0.71 the detector is close to useless.

**What this means**
- **The two failures compound.** An unseen generator in a degraded image is much worse than either problem alone would predict. This holds for both architectures, every damage type and every generator, with 2 small exceptions out of 30 (both InsightFace + noise, interaction about −0.01).
- **Xception compounds more** (+0.051 vs +0.025). It is the stronger model on clean, seen data but the more fragile one when both problems occur, so better in-distribution accuracy does *not* mean better robustness.
- Even "mild" damage that barely hurts seen generators (JPEG, resize: drop ≤ 0.007) **adds 3–5× that amount** when the generator is unseen. The generalisation gap makes the model more sensitive to damage.
- Practical meaning: published detectors are usually tested on clean images of known generators, which is exactly the setting where they look best. A realistic deepfake (new generator, shared on social media) falls into the worst cell of this table.

---

## 5. Known issues and caveats

1. **Unequal training length.** Two ResNet-18 single-generator models (text2img, inpainting) trained for 8 epochs; all others trained for 2. This affects only two rows of Trial 1. We will retrain them.
2. **Resolution matching is one-sided.** Fakes are only ever *shrunk* to their real partner's resolution, so in 41% of InsightFace pairs (21–32% for the other generators) the fake carries less detail. On real photos, smaller faces receive higher fake scores (ρ = −0.22 for ResNet-18, −0.31 for Xception). This is a mild "soft looks fake" bias and likely contributes to the high blur false-positive rate. It does *not* explain the near-perfect baseline: AUC stays ≥ 0.991 on exactly matched pairs. We will either match both images in each pair or report it as a limitation.
3. **Statistical significance vs size.** With about 10,000 test images per comparison, even a 0.001 AUC change is significant. We therefore always report the **size** of the drop (ΔAUC) next to the p-value.
4. **Fixed threshold.** The threshold is tuned on clean validation data, so accuracy, recall and FPR under degradation reflect a deployed detector that was not recalibrated. AUC shows the threshold-free picture.
5. **Face-swap donors.** InsightFace pastes in a face from another person, and that donor's split is not controlled. This could cause a small leak in the insight results.

---

## 6. Next steps

- **Trial 3 (explanation drift):** use Grad-CAM heat maps to measure whether the model's attention moves when it fails, and whether it loses the *same* evidence on both roads.
- **Trial 4 (region degradation):** damage only the face, only the background, or a random patch, to find which region the evidence comes from.
- Fix caveat 1 (and possibly caveat 2), then finalise the full report.
