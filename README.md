<a id="readme-top"></a>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#elisabeth-b4-implementation">Elisabeth-b4 implementation</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#usage">Usage</a></li>
      </ul>
    </li>
    <li><a href="#side-channel-analysis">Side-channel Analysis</a></li>
      <ul>
        <li><a href="#prerequisites-1">Prerequisites</a></li>
        <li><a href="#without-countermeasures">Without countermeasures</a></li>
        <li><a href="#with-countermeasures">With countermeasures</a></li>
        <li><a href="#an-innovative-countermeasure">An innovative countermeasure</a></li>
      </ul>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains the code written by Pierugo Pace as part of his Master's thesis at EPFL & ETHZ carried out at the Nagra Kudelski Group, supervised by Hervé Pelletier and Serge Vaudenay. The code is segmented following the structure of the thesis, i.e.:

* The implementation of Elisabeth-b4 in C++.
* The side-channel analysis of Elisabeth-b4 without any countermeasures.
* The side-channel analysis of Elisabeth-b4 with the presence of 2-share arithmetic masking and shuffling.
* The side-channel analysis of Elisabeth-b4 with the presence of an innovative countermeasure described in the thesis.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ELISABETH-B4 -->
## Elisabeth-b4 implementation

Elisabeth-b4 is implemented in C++ under `elisabeth`. It was inspired by the cipher's authors' [implementation](https://github.com/princess-elisabeth/Elisabeth/tree/master) in Rust.

### Prerequisites

* The code is intended to be embedded on an Arduino board (we tested Arduino DUE).
* The only library required to run the code is Crypto ([Reference](https://www.arduino.cc/reference/en/libraries/crypto/), [Documentation](https://rweather.github.io/arduinolibs/crypto.html), [Source](https://github.com/OperatorFoundation/Crypto)). It can easily be installed through the Arduino Library Manager.

### Usage

Compile and upload the code to an Arduino board. The program waits for user input interpreted as commands and executes the latter with chosen inputs. The commands can be an entire encryption or only subparts of it.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- SIDE-CHANNEL ANALYSIS -->
## Side-channel Analysis

Side-channel analysis is performed with Python through the use of Jupyter notebooks. We consider three different scenarios: without and with well-known countermeasures, as well as our own innovative one.

### Prerequisites

* Python 3
* The requirements found in the respective scenarios: `side_channel_attack/requirements.txt`, `protected_side_channel_attack/requirements.txt` (or `protected_side_channel_attack/deep_learning/requirements.txt` for Deep Learning with TensorFlow), `innovative_countermeasure/requirements.txt`.

### Without countermeasures

This part is located under `side_channel_attack/`.

* Helper files are `utils.py` for general constants and function definitions, and `log_parser.py` for the parsing of the acquisitions.
* A Test-Vector Leakage Assessment is computed in `tvla.ipynb`.
* Different consumption models are analyzed in `cpa_analyze_hyps.ipynb`, with the best one being the fourth S-Box with which we mount a Correlation Power Analysis to recover the entire key in `cpa_sbox_4.ipynb`.
* Template and Machine-Learning based attacks are mounted in `template_ml_attacks.ipynb`.

### With countermeasures

This part is located under `protected_side_channel_attack/`.

* Helper files are `utils.py` for general constants and function definitions, `data_loader.py` and `log_parser.py` for the parsing of the acquisitions, `signal_strength.py` for the signal strength definitions, and `templates.py` for the templates and recovery of the key from prediction probabilities.
* A Test-Vector Leakage Assessment is computed in `tvla.ipynb`.
* Template attacks are mounted in `template_attacks.ipynb`.
* `deep_learning` contains further files used to mount Deep Learning (DL) attacks.
   * `deep_learning.py` contains the definitions of different DL model architectures.
   * `deep_learning_training_*.ipynb` are used to train our different DL models (profiling phase).
   * `deep_learning_inference_*.ipynb` are used to evaluate our models on new observed traces (extraction phase).
   * `deep_learning_evaluation.ipynb` is used to evaluate the performance of different DL architectures and compare them.
   * `subwindows.ipynb` describes how we derive subwindows of the traces to train less complex models to achieve a better performance. It is done entirely visually by looking at the PoIs with SOST of different variables.
   * `wavelets.ipynb` describes how we compute the 2nd order Haar wavelet transform of the traces to reduce their size.
   * `utils.py`, `signal_strength.py` and `templates.py` are identical to the parent directory.

### An innovative countermeasure

This part is located under `innovative_countermeasure/`.

* Helper files are `utils.py` for general constants and function definitions, `data_loader.py` and `log_parser.py` for the parsing of the acquisitions, and `signal_strength.py` for the signal strength definitions.
* A Test-Vector Leakage Assessment is computed in `tvla.ipynb`.
* The Sum Of Squared pairwise T-differences (SOST) of different critical values were computed in parallel with `generate_sost.py` for different devices and keys (as described in our countermeasure).
* The effect of the countermeasure on the SOST of critical values is analyzed in `sost_analysis.ipynb`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Pierugo Pace - pierugo.pace@gmail.com

Hervé Pelletier - herve.pelletier@nagra.com

Serge Vaudenay - serge.vaudenay@epfl.ch

<p align="right">(<a href="#readme-top">back to top</a>)</p>
