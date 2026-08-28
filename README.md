# Landslide Analysis from Optical Remote Sensing Imagery with Multimodal Large-Small Models


**Short description.** This repository provides the open-source code accompanying the manuscript *"LMHM"*, submitted to *Computers & Geosciences*. It implements a multimodal large-model framework for the detection, description, and dataset construction of landslides from optical remote sensing imagery, together with a text data-augmentation pipeline driven by Qwen3-VL.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Quick Test / Example](#quick-test--example)
7. [How to Run the Test](#how-to-run-the-test)
8. [License](#license)
9. [Computer Code Availability](#computer-code-availability)

---

## Overview

The code in this repository supports three main stages of the study:

- **Visual grounding of landslide features** — aligning optical remote sensing imagery with standardized, machine-readable textual descriptions of landslide characteristics (scar zone, head scarp, debris accumulation, surface texture discontinuity, vegetation removal, displaced material, landslide boundary).
- **Text data augmentation** — rewriting existing question–answer pairs into a canonical form suitable for training multimodal large models, using the Qwen3-VL vision-language model.
- **Landslide detection** — benchmark and evaluation of detection models on the study dataset.

The software is written in Python and is intended to run on a standard workstation or a GPU-enabled machine.

---

## Repository Structure

The repository contains individual source files (no compressed archives), organised as follows:

```
.
├── README.md                     # This file
├── LICENSE                       # Open-source license (MIT)
├── src/
│   ├── augment_qwen3vl.py        # Text data augmentation via Qwen3-VL (DashScope API)
│   └── ...                       # Additional pipeline modules (to be released)
├── configs/
│   └── ...                       # Configuration files
├── examples/
│   ├── quick_test.py             # Minimal end-to-end test
│   ├── sample_image.tif          # Small sample optical image
│   └── sample_text.json          # Small sample question–answer pair
└── requirements.txt              # Python dependencies
```

---

## Requirements

- Python 3.9+
- An active [DashScope (Alibaba Cloud Model Studio)](https://dashscope.aliyuncs.com) API key for the Qwen3-VL model.

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The main third-party packages are:

- `openai` — OpenAI-compatible client used to call Qwen3-VL through DashScope.
- `pillow` — image handling and TIFF-to-PNG conversion.
- `numpy` — numerical utilities.

---

## Installation

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
```

Set the DashScope API key as an environment variable:

```bash
# Linux / macOS
export DASHSCOPE_API_KEY="sk-xxxx"

# Windows PowerShell
$env:DASHSCOPE_API_KEY="sk-xxxx"
```

---

## Usage

The text data-augmentation pipeline reads an image folder and a text (JSON) folder, pairs every image with its corresponding text file by filename stem, and rewrites the question–answer pair with Qwen3-VL.

```bash
python src/augment_qwen3vl.py --limit 5      # dry run on 5 samples
python src/augment_qwen3vl.py                 # process the full dataset
```

The script writes one JSON object per sample to an output `.jsonl` file, with the fields `stem`, `image`, `question`, and `answer`.

> The full training and evaluation pipeline, including the complete detection code and configuration files, will be released in this repository upon publication of the associated manuscript (see [Computer Code Availability](#computer-code-availability)).

---

## Quick Test / Example

A minimal, self-contained example is provided under [`examples/`](examples/):

- `examples/sample_image.tif` — a small optical remote sensing image.
- `examples/sample_text.json` — a matching question–answer pair in JSON format.
- `examples/quick_test.py` — a script that reproduces one full augmentation step on the sample data and checks that the output is a valid JSON object with the required fields.

---

## How to Run the Test

Run the following command from the repository root:

```bash
python examples/quick_test.py
```

A successful run prints the augmented `question` and `answer` to the console and exits with status code `0`. No network call to the model is required by default (the example uses a bundled sample output); to run the test against the live Qwen3-VL API, pass the `--live` flag and ensure `DASHSCOPE_API_KEY` is set:

```bash
python examples/quick_test.py --live
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Computer Code Availability

The source code developed for and used in this study is released in this public repository and is available for anonymous download.

- **Repository:** <https://github.com/your-username/your-repo>
- **License:** MIT License (see [LICENSE](LICENSE)).

The code that forms a substantial part of this work is open-source and freely available. This repository currently provides the core components and a runnable quick-test example. **The complete source code, including the full training and evaluation pipeline, configuration files, and auxiliary scripts, will be fully released in this repository upon publication of the associated manuscript.**

---

*This README follows the guidelines of Computers & Geosciences for sharing source code. No proprietary software is required to reproduce the results presented in the manuscript.*
