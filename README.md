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
<!--
<table>
    <thead>
        <tr>
            <th> </th>
            <th><sub>IDR</sub></th>
            <th><sub>MR</sub></th>
            <th><sub>FAR</sub></th>
            <th><sub>IDA</sub></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><sub>XXX</sub></td>
            <td><sub><strong>XXX</strong></sub></td>
        </tr>        
    </tbody>
</table>
-->

<!--
<table>
  <thead>
    <tr>
        <th class="tg-baqh" rowspan="2"></th>
        <th class="tg-baqh" colspan="3">IDR</th>
        <th class="tg-baqh" colspan="3">MR</th>
        <th class="tg-baqh" colspan="3">FAR</th>
        <th class="tg-baqh" colspan="3">IDA</th>
    </tr>
  </thead>
  <tbody>
  <tr>
    <td class="tg-baqh"></td>
    <td class="tg-baqh">synth</td>
    <td class="tg-baqh">CMU</td>
    <td class="tg-baqh">PTDB</td>
    <td class="tg-baqh">synth</td>
    <td class="tg-baqh">CMU</td>
    <td class="tg-baqh">PTDB</td>
    <td class="tg-baqh">synth</td>
    <td class="tg-baqh">CMU</td>
    <td class="tg-baqh">PTDB</td>
    <td class="tg-baqh">synth</td>
    <td class="tg-baqh">CMU</td>
    <td class="tg-baqh">PTDB</td>
  </tr>
  <tr>
    <td class="tg-0lax">FCN-synth-tri</td>
    <td class="tg-baqh">99.90</td>
    <td class="tg-baqh">97.95</td>
    <td class="tg-baqh">95.37</td>
    <td class="tg-baqh">0.08</td>
    <td class="tg-baqh">1.89</td>
    <td class="tg-baqh">3.40</td>
    <td class="tg-baqh"><span style="font-weight:bold">0.02</span></td>
    <td class="tg-baqh">0.17</td>
    <td class="tg-baqh">1.22</td>
    <td class="tg-baqh"><span style="font-weight:bold">0.08</span></td>
    <td class="tg-baqh">0.26</td>
    <td class="tg-baqh">0.32</td>
  </tr>
  <tr>
    <td class="tg-0lax">FCN-synth-GF</td>
    <td class="tg-baqh"><span style="font-weight:bold">99.91</span></td>
    <td class="tg-baqh">98.43</td>
    <td class="tg-baqh"><span style="font-weight:bold">95.64</span></td>
    <td class="tg-baqh"><span style="font-weight:bold">0.06</span></td>
    <td class="tg-baqh">1.20</td>
    <td class="tg-baqh">2.91</td>
    <td class="tg-baqh">0.04</td>
    <td class="tg-baqh">0.37</td>
    <td class="tg-baqh">1.45</td>
    <td class="tg-baqh">0.11</td>
    <td class="tg-baqh">0.34</td>
    <td class="tg-baqh">0.38</td>
  </tr>
  <tr>
    <td class="tg-0lax">FCN-CMU-10/90</td>
    <td class="tg-baqh">49.63</td>
    <td class="tg-baqh">99.39</td>
    <td class="tg-baqh">90.13</td>
    <td class="tg-baqh">48.05</td>
    <td class="tg-baqh">0.50</td>
    <td class="tg-baqh">8.91</td>
    <td class="tg-baqh">0.51</td>
    <td class="tg-baqh">0.11</td>
    <td class="tg-baqh">0.95</td>
    <td class="tg-baqh">0.52</td>
    <td class="tg-baqh">0.10</td>
    <td class="tg-baqh"><span style="font-weight:bold">0.26</span></td>
  </tr>
  <tr>
    <td class="tg-0lax">FCN-CMU-60/20/20</td>
    <td class="tg-baqh">60.06</td>
    <td class="tg-baqh"><span style="font-weight:bold">99.52</span></td>
    <td class="tg-baqh">88.17</td>
    <td class="tg-baqh">39.14</td>
    <td class="tg-baqh">0.40</td>
    <td class="tg-baqh">11.00</td>
    <td class="tg-baqh">0.64</td>
    <td class="tg-baqh"><span style="font-weight:bold">0.08</span></td>
    <td class="tg-baqh"><span style="font-weight:bold">0.81</span></td>
    <td class="tg-baqh">0.50</td>
    <td class="tg-baqh"><span style="font-weight:bold">0.09</span></td>
    <td class="tg-baqh"><span style="font-weight:bold">0.26</span></td>
  </tr>
  <tr>
    <td class="tg-0lax">SEDREAMS</td>
    <td class="tg-baqh">89.26</td>
    <td class="tg-baqh">99.04</td>
    <td class="tg-baqh">95.34</td>
    <td class="tg-baqh">3.86</td>
    <td class="tg-baqh"><span style="font-weight:bold">0.21</span></td>
    <td class="tg-baqh"><span style="font-weight:bold">2.15</span></td>
    <td class="tg-baqh">6.88</td>
    <td class="tg-baqh">0.75</td>
    <td class="tg-baqh">2.51</td>
    <td class="tg-baqh">0.68</td>
    <td class="tg-baqh">0.36</td>
    <td class="tg-baqh">0.62</td>
  </tr>
  <tr>
    <td class="tg-0lax">DPI</td>
    <td class="tg-baqh">88.22</td>
    <td class="tg-baqh">98.69</td>
    <td class="tg-baqh">91.3</td>
    <td class="tg-baqh">2.14</td>
    <td class="tg-baqh">0.23</td>
    <td class="tg-baqh">2.16</td>
    <td class="tg-baqh">9.64</td>
    <td class="tg-baqh">1.08</td>
    <td class="tg-baqh">6.53</td>
    <td class="tg-baqh">0.83</td>
    <td class="tg-baqh">0.23</td>
    <td class="tg-baqh">0.49</td>
  </tr>
  <tr>
    <td class="tg-0lax">DCNN (from [21])</td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh">99.3</td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh">0.3</td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh">0.4</td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh">0.2</td>
    <td class="tg-baqh"></td>
  </tr>
  </tbody>
</table>
-->

<div class="tg-wrap">
 <table>
  <tr>
    <th rowspan="2"></th>
    <th colspan="3">IDR</th>
    <th colspan="3">MR</th>
    <th colspan="3">FAR</th>
    <th colspan="3">IDA</th>
  </tr>
  <tr>
    <td>synth</td>
    <td>CMU</td>
    <td>PTDB</td>
    <td>synth</td>
    <td>CMU</td>
    <td>PTDB</td>
    <td>synth</td>
    <td>CMU</td>
    <td>PTDB</td>
    <td>synth</td>
    <td>CMU</td>
    <td>PTDB</td>
  </tr>
  <tr>
    <td>FCN-synth-tri</td>
    <td>99.90</td>
    <td>97.95</td>
    <td>95.37</td>
    <td>0.08</td>
    <td>1.89</td>
    <td>3.40</td>
    <td><span style="font-weight:bold">0.02</span></td>
    <td>0.17</td>
    <td>1.22</td>
    <td><span style="font-weight:bold">0.08</span></td>
    <td>0.26</td>
    <td>0.32</td>
  </tr>
  <tr>
    <td>FCN-synth-GF</td>
    <td><span style="font-weight:bold">99.91</span></td>
    <td>98.43</td>
    <td><span style="font-weight:bold">95.64</span></td>
    <td><span style="font-weight:bold">0.06</span></td>
    <td>1.20</td>
    <td>2.91</td>
    <td>0.04</td>
    <td>0.37</td>
    <td>1.45</td>
    <td>0.11</td>
    <td>0.34</td>
    <td>0.38</td>
  </tr>
  <tr>
    <td>FCN-CMU-10/90</td>
    <td>49.63</td>
    <td>99.39</td>
    <td>90.13</td>
    <td>48.05</td>
    <td>0.50</td>
    <td>8.91</td>
    <td>0.51</td>
    <td>0.11</td>
    <td>0.95</td>
    <td>0.52</td>
    <td>0.10</td>
    <td><span style="font-weight:bold">0.26</span></td>
  </tr>
  <tr>
    <td>FCN-CMU-60/20/20</td>
    <td>60.06</td>
    <td><span style="font-weight:bold">99.52</span></td>
    <td>88.17</td>
    <td>39.14</td>
    <td>0.40</td>
    <td>11.00</td>
    <td>0.64</td>
    <td><span style="font-weight:bold">0.08</span></td>
    <td><span style="font-weight:bold">0.81</span></td>
    <td>0.50</td>
    <td><span style="font-weight:bold">0.09</span></td>
    <td><span style="font-weight:bold">0.26</span></td>
  </tr>
  <tr>
    <td>SEDREAMS</td>
    <td>89.26</td>
    <td>99.04</td>
    <td>95.34</td>
    <td>3.86</td>
    <td><span style="font-weight:bold">0.21</span></td>
    <td><span style="font-weight:bold">2.15</span></td>
    <td>6.88</td>
    <td>0.75</td>
    <td>2.51</td>
    <td>0.68</td>
    <td>0.36</td>
    <td>0.62</td>
  </tr>
  <tr>
    <td>DPI</td>
    <td>88.22</td>
    <td>98.69</td>
    <td>91.3</td>
    <td>2.14</td>
    <td>0.23</td>
    <td>2.16</td>
    <td>9.64</td>
    <td>1.08</td>
    <td>6.53</td>
    <td>0.83</td>
    <td>0.23</td>
    <td>0.49</td>
  </tr>
  <tr>
    <td>DCNN (from [21])</td>
    <td></td>
    <td>99.3</td>
    <td></td>
    <td></td>
    <td>0.3</td>
    <td></td>
    <td></td>
    <td>0.4</td>
    <td></td>
    <td></td>
    <td>0.2</td>
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
