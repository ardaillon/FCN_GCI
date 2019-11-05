# FCN_GCI
Detection of GCIs from raw speech signals using a fully-convolutional network (FCN)

Code for running Glottal Closure Instants (GCI) detection using the fully-convolutional neural network models described in the following publication :
> [GCI detection from raw speech using a fully-convolutional network](https://arxiv.org/abs/1910.10235)<br>
> Luc Ardaillon, Axel Roebel.<br>
> Submitted on arxiv on 22 Oct 2019.

We kindly request academic publications making use of our FCN models to cite the aforementioned paper.

## Description
The code provided in this repository aims at performing GCI dectection using a Fully-Convolutional Neural Network.

The provided code allows to run the GCI detection on given speech sound files using the provided pretrained models, but no code is currently provided to train the model on new data.
All pre-trained models evaluated in the above-mentionned paper are provided.
The default model "XXX" has been trained on a large database of high-quality synthetic speech (obtained by resynthesizing ...)

The models, algorithm, training, and evaluation procedures have been described in a publication entitled "GCI detection from raw speech using a fully-convolutional network" (https://arxiv.org/abs/1910.10235).

Below are the results of our evaluations comparing our models to the SEDREAMS and DPI algorithms, in terms of XXX (XXX, on both a test database of synthetic speech "XXX" and two datasets of real speech samples with XXX "XXX"). All model and algorithms have been evaluated on 16kHz audio (???).
<table>
    <thead>
        <tr>
            <th> </th>
            <th><sub>XXX</sub></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><sub>XXX</sub></td>
            <td><sub><strong>XXX</strong></sub></td>
        </tr>        
    </tbody>
</table>

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
![Example of prediction of triangle shape from real speech extract](examples/figures/prediction_triangle_example.png?raw=true "Example of prediction of triangle shape from real speech extract")

Example of prediction of glottal flow shape from real speech extract :
![Example of prediction of glottal flow shape from real speech extract](examples/figures/prediction_glottal_flow_example.png?raw=true "Example of prediction of glottal flow shape from real speech extract")

## References
[1] XXX

[2] J. L. Gauvain, L. F. Lamel, and M. Eskenazi, “Design Considerations and Text Selection for BREF, a large French Read-Speech Corpus,” 1st International Conference on Spoken Language Processing, ICSLP, no. January 2013, pp. 1097–1100, 1990. http://www.limsi.fr/~lamel/kobe90.pdf

[3] V. Zue, S. Seneff, and J. Glass, “Speech Database Development At MIT : TIMIT And Beyond,” vol. 9, pp. 351–356, 1990.

[4] L. Ardaillon, “Synthesis and expressive transformation of singing voice,” Ph.D. dissertation, EDITE; UPMC-Paris 6 Sorbonne Universités, 2017.
