# FCN_GCI
Detection of GCIs from raw speech signals using a fully-convolutional network (FCN)

Code for running Glottal Closure Instants (GCI) detection using the fully-convolutional neural network models described in the following publication :
> [GCI detection from raw speech using a fully-convolutional network](https://arxiv.org/abs/1910.10235)<br>
> Luc Ardaillon, Axel Roebel.<br>
> Submitted on arxiv on 22 Oct 2019.

We kindly request academic publications making use of our FCN models to cite the aforementioned paper.

## Description
The code provided in this repository aims at performing GCI dectection using a Fully-Convolutional Neural Network. Note that it also allows to perform the prediction of the glottal flow shape (normalized in amplitude) from which more information than the GCIs may be extracted.

The provided code allows to run the GCI detection on given speech sound files using the provided pretrained models, but no code is currently provided to train the model on new data.
All pre-trained models evaluated in the above-mentionned paper are provided.
The models "FCN_synth_GF" and "FCN_synth_tri have been trained on a large database of high-quality synthetic speech (obtained by resynthesizing the BREF and TIMIT database using the PaN vocoder [4]). The difference between the 2 is that "FCN_synth_tri" predicts a triangular curve from which the GCIs are extracted by simple peak-picking on the maximums, while "FCN_synth_GF" predicts the glottal flow shape and performs the peak-picking on its negative derivative. The "FCN_CMU__10_90" and "FCN_CMU__60_20_20" models have been trained on the CMU database (with different train/validation/test splits) using a triangle shape as target.

The models, algorithm, training, and evaluation procedures have been described in a publication entitled "GCI detection from raw speech using a fully-convolutional network" (https://arxiv.org/abs/1910.10235).

Below are the results of our evaluations comparing our models to the SEDREAMS and DPI algorithms, in terms of IDR, MR, FAR, and IDA. The evaluation has been conducted on both a test database of synthetic speech and two datasets of real speech samples from the CMU [XX] and PTDB-TUG [XX] databases). All model and algorithms have been evaluated on 16kHz audio.

<div class="tg-wrap">
 <table>
  <tr>
    <th rowspan="2"></th>
    <th colspan="3"><sub>IDR</sub></th>
    <th colspan="3"><sub>MR</sub></th>
    <th colspan="3"><sub>FAR</sub></th>
    <th colspan="3"><sub>IDA</sub></th>
  </tr>
  <tr>
    <td><sub>synth</sub></td>
    <td><sub>CMU</sub></td>
    <td><sub>PTDB</sub></td>
    <td><sub>synth</sub></td>
    <td><sub>CMU</sub></td>
    <td><sub>PTDB</sub></td>
    <td><sub>synth</sub></td>
    <td><sub>CMU</sub></td>
    <td><sub>PTDB</sub></td>
    <td><sub>synth</sub></td>
    <td><sub>CMU</sub></td>
    <td><sub>PTDB</sub></td>
  </tr>
  <tr>
    <td><sub>FCN-synth-tri</sub></td>
    <td><sub>99.90</sub></td>
    <td><sub>97.95</sub></td>
    <td><sub>95.37</sub></td>
    <td><sub>0.08</sub></td>
    <td><sub>1.89</sub></td>
    <td><sub>3.40</sub></td>
    <td><span style="font-weight:bold"><sub>0.02</sub></span></td>
    <td><sub>0.17</sub></td>
    <td><sub>1.22</sub></td>
    <td><span style="font-weight:bold"><sub>0.08</sub></span></td>
    <td><sub>0.26</sub></td>
    <td><sub>0.32</sub></td>
  </tr>
  <tr>
    <td><sub>FCN-synth-GF</td>
    <td><span style="font-weight:bold"><sub>99.91</sub></span></td>
    <td><sub>98.43</sub></td>
    <td><span style="font-weight:bold"><sub>95.64</sub></span></td>
    <td><span style="font-weight:bold"><sub>0.06</sub></span></td>
    <td><sub>1.20</sub></td>
    <td><sub>2.91</sub></td>
    <td><sub>0.04</sub></td>
    <td><sub>0.37</sub></td>
    <td><sub>1.45</sub></td>
    <td><sub>0.11</sub></td>
    <td><sub>0.34</sub></td>
    <td><sub>0.38</sub></td>
  </tr>
  <tr>
    <td><sub>FCN-CMU-10/90</sub></td>
    <td><sub>49.63</sub></td>
    <td><sub>99.39</sub></td>
    <td><sub>90.13</sub></td>
    <td><sub>48.05</sub></td>
    <td><sub>0.50</sub></td>
    <td><sub>8.91</sub></td>
    <td><sub>0.51</sub></td>
    <td><sub>0.11</sub></td>
    <td><sub>0.95</sub></td>
    <td><sub>0.52</sub></td>
    <td><sub>0.10</sub></td>
    <td><span style="font-weight:bold"><sub>0.26</sub></span></td>
  </tr>
  <tr>
    <td><sub>FCN-CMU-60/20/20</sub></td>
    <td><sub>60.06</sub></td>
    <td><span style="font-weight:bold"><sub>99.52</sub></span></td>
    <td><sub>88.17</sub></td>
    <td><sub>39.14</sub></td>
    <td><sub>0.40</sub></td>
    <td><sub>11.00</sub></td>
    <td><sub>0.64</sub></td>
    <td><span style="font-weight:bold"><sub>0.08</sub></span></td>
    <td><span style="font-weight:bold"><sub>0.81</sub></span></td>
    <td><sub>0.50</sub></td>
    <td><span style="font-weight:bold"><sub>0.09</sub></span></td>
    <td><span style="font-weight:bold"><sub>0.26</sub></span></td>
  </tr>
  <tr>
    <td><sub>SEDREAMS</sub></td>
    <td><sub>89.26</sub></td>
    <td><sub>99.04</sub></td>
    <td><sub>95.34</sub></td>
    <td><sub>3.86</sub></td>
    <td><span style="font-weight:bold"><sub>0.21</sub></span></td>
    <td><span style="font-weight:bold"><sub>2.15</sub></span></td>
    <td><sub>6.88</sub></td>
    <td><sub>0.75</sub></td>
    <td><sub>2.51</sub></td>
    <td><sub>0.68</sub></td>
    <td><sub>0.36</sub></td>
    <td><sub>0.62</sub></td>
  </tr>
  <tr>
    <td><sub>DPI</sub></td>
    <td><sub>88.22</sub></td>
    <td><sub>98.69</sub></td>
    <td><sub>91.3</sub></td>
    <td><sub>2.14</sub></td>
    <td><sub>0.23</sub></td>
    <td><sub>2.16</sub></td>
    <td><sub>9.64</sub></td>
    <td><sub>1.08</sub></td>
    <td><sub>6.53</sub></td>
    <td><sub>0.83</sub></td>
    <td><sub>0.23</sub></td>
    <td><sub>0.49</sub></td>
  </tr>
  <tr>
    <td><sub>DCNN (from [5])</sub></td>
    <td></td>
    <td><sub>99.3</sub></td>
    <td></td>
    <td></td>
    <td><sub>0.3</sub></td>
    <td></td>
    <td></td>
    <td><sub>0.4</sub></td>
    <td></td>
    <td></td>
    <td><sub>0.2</sub></td>
    <td></td>
  </tr>
 </table>
</div>

Our synthetic speech database has been created by resynthesizing the BREF [2] and TIMIT [3] databases using the PAN synthesis engine, described in [4, Section 3.5.2].

## Example command-line usage (using provided pretrained models)
TODO
<!--
#### Default analysis : This will run the FCN-993 model and output the result as a csv file in the same folder than the input file (replacing the file extension by ".csv")
python /path_to/FCN-f0/FCN_GCI.py /path_to/test.wav
-->

<!--
#### Run the analysis on a whole folder of audio files
python /path_to/FCN-f0/FCN_GCI.py /path_to/audio_files
-->

<!--
#### Choose a specific model for running the analysis (default is FCN-993)
Use FCN-synth-tri model :
python /path_to/FCN-f0/FCN_GCI.py /path_to/test.wav -m FCN-synth-tri -o /path_to/output.FCN-synth-tri.GCI.sdif
-->

<!--
XXX ...
-->

<!-- 
#### Specify an output directory or file name with "-o" option(if directory doesn't exist, it will be created)
python /path_to/FCN-f0/FCN_GCI.py /path_to/test.wav -o /path_to/output.GCI.lab
python /path_to/FCN-f0/FCN_GCI.py /path_to/audio_files -o /path_to/output_dir
-->
<!-- 
#### Output result to sdif format (requires installing the eaSDIF python library. Default format is lab)
python /path_to/FCN-f0/FCN_GCI.py /path_to/test.wav -f sdif
-->

## Example figures
Example of prediction of triangle shape from real speech extract :
![Example of prediction of triangle shape from real speech extract](examples/figures/pred_triangle.png?raw=true "Example of prediction of triangle shape from real speech extract")

Example of prediction of glottal flow shape from real speech extract :
![Example of prediction of glottal flow shape from real speech extract](examples/figures/pred_GF.png?raw=true "Example of prediction of glottal flow shape from real speech extract")

## References
[1] XXX

[2] J. L. Gauvain, L. F. Lamel, and M. Eskenazi, “Design Considerations and Text Selection for BREF, a large French Read-Speech Corpus,” 1st International Conference on Spoken Language Processing, ICSLP, no. January 2013, pp. 1097–1100, 1990. http://www.limsi.fr/~lamel/kobe90.pdf

[3] V. Zue, S. Seneff, and J. Glass, “Speech Database Development At MIT : TIMIT And Beyond,” vol. 9, pp. 351–356, 1990.

[4] L. Ardaillon, “Synthesis and expressive transformation of singing voice,” Ph.D. dissertation, EDITE; UPMC-Paris 6 Sorbonne Universités, 2017.
