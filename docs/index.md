# OverFlow: Putting flows on top of neural transducers for better TTS
##### [Shivam Mehta][shivam_profile], [Ambika Kirkland][ambika_profile], [Harm Lameris][harm_profile], [Jonas Beskow][jonas_profile], [Éva Székely][eva_profile], and [Gustav Eje Henter][gustav_profile]

[arxiv_link]: https://arxiv.org/abs/2108.13320
[github_link]: https://github.com/shivammehta25/OverFlow
[shivam_profile]: https://www.kth.se/profile/smehta
[ambika_profile]: https://www.kth.se/profile/kirkland
[harm_profile]: https://www.kth.se/profile/lameris
[eva_profile]: https://www.kth.se/profile/szekely
[jonas_profile]: https://www.kth.se/profile/beskow
[gustav_profile]: https://people.kth.se/~ghe/
[HiFi_GAN_LJ_FT_V1_link]: https://github.com/jik876/hifi-gan#pretrained-model
[Neural_HMM_link]: https://shivammehta25.github.io/Neural-HMM/

<head>
<link rel="apple-touch-icon" sizes="180x180" href="favicon/apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="32x32" href="favicon/favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="favicon/favicon-16x16.png">
<link rel="manifest" href="/site.webmanifest">
<link rel="mask-icon" href="/safari-pinned-tab.svg" color="#5bbad5">
<meta name="msapplication-TileColor" content="#da532c">
<meta name="theme-color" content="#ffffff">
</head>

<style type="text/css">
  .tg {
    border-collapse: collapse;
    border-color: #9ABAD9;
    border-spacing: 0;
  }

  .tg td {
    background-color: #EBF5FF;
    border-color: #9ABAD9;
    border-style: solid;
    border-width: 1px;
    color: #444;
    font-family: Arial, sans-serif;
    font-size: 14px;
    overflow: hidden;
    padding: 0px 20px;
    word-break: normal;
    font-weight: bold;
    vertical-align: middle;
  }

  .tg th {
    background-color: #409cff;
    border-color: #9ABAD9;
    border-style: solid;
    border-width: 1px;
    color: #fff;
    font-family: Arial, sans-serif;
    font-size: 14px;
    font-weight: normal;
    overflow: hidden;
    padding: 0px 20px;
    word-break: normal;
    font-weight: bold;
    vertical-align: middle;

  }

  .tg .tg-0pky {
    border-color: inherit;
    text-align: center;
    vertical-align: top,
  }

  .tg .tg-fymr {
    border-color: inherit;
    font-weight: bold;
    text-align: center;
    vertical-align: top
  }
  .slider {
  -webkit-appearance: none;
  width: 75%;
  height: 15px;
  border-radius: 5px;
  background: #d3d3d3;
  outline: none;
  opacity: 0.7;
  -webkit-transition: .2s;
  transition: opacity .2s;
}

.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 25px;
  height: 25px;
  border-radius: 50%;
  background: #409cff;
  cursor: pointer;
}

.slider::-moz-range-thumb {
  width: 25px;
  height: 25px;
  border-radius: 50%;
  background: #409cff;
  cursor: pointer;
}

audio {
    width: 110px;
}
</style>

## Summary

We propose a new approach **[OverFlow][github_link]** to address the shortcomings of [neural-HMM TTS][Neural_HMM_link] by adding flows over them. We model the distribution of the latent space of the neural-HMM TTS using normalizing flows. Having a stronger probabilistic model, we can now model the highly non-Gaussian distribution of speech acoustics resulting in improvements in pronunciation and naturalness. We showed that our model converges to lower word error rate (WER) faster and achieves higher naturalness scores than comparable methods. The resulting system:
* Learns to speak and align fast
* Is fully probabilistic and compute tighter bounds on the data likelihood
* Can generate samples at different temperatures
* Requires small amount of data
* Can adapt to new speaker with limited data (Should we put it here?)

[Neural-HMM TTS][Neural_HMM_link] fuse classical HMM-based statistical speech synthesis and modern neural text-to-speech (TTS) retaining the best of both approaches. They require fewer data and fewer training updates to work while being less prone to gibberish output caused by neural attention failures. However, they could not model the highly non-Gaussian distribution of speech acoustics and despite being probabilistic we could not sample high-quality speech samples. To address these limitations, we propose a new approach **[OverFlow][github_link]** to improve the performance of neural HMMs by putting flows on top of them. This results in a powerful fully probabilistic model of durations and speech acoustics that can be trained using exact maximum likelihood giving more accurate pronunciations and better speech quality than comparable methods with or without sampling from the model.


## Architecture


## Code

Code is available in our [Github repository][github_link], along with a pre-trained models.

<!-- <script >
function playAudio(url) {
  new Audio(url).play();
  audio.play();
}
<img src='images/play.png' onclick="playAudio('./audio/VOC/ListeningTest/1.wav')" />

</script> -->
## Stimuli from the listening tests

<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">Sentence</th>
      <th class="tg-0pky">Vocoded speech</th>
      <th class="tg-0pky" colspan="3">Proposed (OverFlow)</th>
      <th class="tg-0pky">Tacotron 2</th>
      <th class="tg-0pky">Glow-TTS</th>
      <th class="tg-0pky">Neural HMM TTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th class="tg-fymr"></th>
      <th class="tg-fymr">VOC</th>
      <th class="tg-fymr">OF</th>
      <th class="tg-fymr">OFND (No Dropout)</th>
      <th class="tg-fymr">OFZT (Zero Temperature)</th>
      <th class="tg-fymr">T2</th>
      <th class="tg-fymr">GTTS</th>
      <th class="tg-fymr">NHMM</th>
    </tr>
    <tr>
        <td nowrap class="tg-0pky">
            <span style="font-weight:bold">Sentence 1</span>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/VOC/ListeningTest/1.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OF/ListeningTest/1.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OFND/ListeningTest/1.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OFZT/ListeningTest/1.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/T2/ListeningTest/1.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/GTTS/ListeningTest/1.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/NHMM/ListeningTest/1.wav" type="audio/wav">
          </audio>
        </td>
    </tr>
    <tr>
        <td nowrap class="tg-0pky">
            <span style="font-weight:bold">Sentence 2</span>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/VOC/ListeningTest/2.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OF/ListeningTest/2.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OFND/ListeningTest/2.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OFZT/ListeningTest/2.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/T2/ListeningTest/2.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/GTTS/ListeningTest/2.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/NHMM/ListeningTest/2.wav" type="audio/wav">
          </audio>
        </td>
    </tr>
    <tr>
        <td nowrap class="tg-0pky">
            <span style="font-weight:bold">Sentence 3</span>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/VOC/ListeningTest/3.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OF/ListeningTest/3.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OFND/ListeningTest/3.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OFZT/ListeningTest/3.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/T2/ListeningTest/3.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/GTTS/ListeningTest/3.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/NHMM/ListeningTest/3.wav" type="audio/wav">
          </audio>
        </td>
    </tr>
    <tr>
        <td nowrap class="tg-0pky">
            <span style="font-weight:bold">Sentence 4</span>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/VOC/ListeningTest/4.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OF/ListeningTest/4.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OFND/ListeningTest/4.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OFZT/ListeningTest/4.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/T2/ListeningTest/4.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/GTTS/ListeningTest/4.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/NHMM/ListeningTest/4.wav" type="audio/wav">
          </audio>
        </td>
    </tr>
    <tr>
        <td nowrap class="tg-0pky">
            <span style="font-weight:bold">Sentence 5</span>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/VOC/ListeningTest/5.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OF/ListeningTest/5.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OFND/ListeningTest/5.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OFZT/ListeningTest/5.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/T2/ListeningTest/5.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/GTTS/ListeningTest/5.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/NHMM/ListeningTest/5.wav" type="audio/wav">
          </audio>
        </td>
    </tr>
    <tr>
        <td nowrap class="tg-0pky">
            <span style="font-weight:bold">Sentence 6</span>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/VOC/ListeningTest/6.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OF/ListeningTest/6.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OFND/ListeningTest/6.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OFZT/ListeningTest/6.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/T2/ListeningTest/6.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/GTTS/ListeningTest/6.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/NHMM/ListeningTest/6.wav" type="audio/wav">
          </audio>
        </td>
    </tr>
    <tr>
        <td nowrap class="tg-0pky">
            <span style="font-weight:bold">Sentence 7</span>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/VOC/ListeningTest/7.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OF/ListeningTest/7.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OFND/ListeningTest/7.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/OFZT/ListeningTest/7.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/T2/ListeningTest/7.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/GTTS/ListeningTest/7.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls>
            <source src="./audio/NHMM/ListeningTest/7.wav" type="audio/wav">
          </audio>
        </td>
    </tr>
  </tbody>
</table>


## Sampling at different temperatures


<div class="slidecontainer">
  <label for="myRange"><span style="font-weight:bold"> 0 </span></label>
  <input type="range" min="0" max="3" value="2" step="1" class="slider" id="myRange">
  <label for="myRange"><span style="font-weight:bold"> 1 </span> </label>
  <p><span style="font-weight:bold">Sampling temperature:</span> <span id="demo"></span>
  </p>
</div>
<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">Sentence</th>
      <th class="tg-0pky">VocodedSpeech</th>
      <th class="tg-0pky">OF</th>
      <th class="tg-0pky">OFND</th>
      <th class="tg-0pky">GTTS</th>
      <th class="tg-0pky">NHMM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td nowrap class="tg-0pky">
        <span style="font-weight:bold">Sentence 1</span>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/VOC/SamplingTemperature/1.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="OF_a_1">
          <source id="OF_s_1" src="./audio/OF/SamplingTemperature/1_667.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="OFND_a_1">
          <source id="OFND_s_1" src="./audio/OFND/SamplingTemperature/1_667.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="GTTS_a_1">
          <source id="GTTS_s_1" src="./audio/GTTS/SamplingTemperature/1_667.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="NHMM_a_1">
          <source id="NHMM_s_1" src="./audio/NHMM/SamplingTemperature/1_667.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
    <tr>
      <td nowrap class="tg-0pky">
        <span style="font-weight:bold">Sentence 2</span>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/VOC/SamplingTemperature/2.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="OF_a_2">
          <source id="OF_s_2" src="./audio/OF/SamplingTemperature/2_667.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="OFND_a_2">
          <source id="OFND_s_2" src="./audio/OFND/SamplingTemperature/2_667.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="GTTS_a_2">
          <source id="GTTS_s_2" src="./audio/GTTS/SamplingTemperature/2_667.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="NHMM_a_2">
          <source id="NHMM_s_2" src="./audio/NHMM/SamplingTemperature/2_667.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
    <tr>
      <td nowrap class="tg-0pky">
        <span style="font-weight:bold">Sentence 3</span>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/VOC/SamplingTemperature/3.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="OF_a_3">
          <source id="OF_s_3" src="./audio/OF/SamplingTemperature/3_667.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="OFND_a_3">
          <source id="OFND_s_3" src="./audio/OFND/SamplingTemperature/3_667.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="GTTS_a_3">
          <source id="GTTS_s_3" src="./audio/GTTS/SamplingTemperature/3_667.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="NHMM_a_3">
          <source id="NHMM_s_3" src="./audio/NHMM/SamplingTemperature/3_667.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
    <tr>
      <td nowrap class="tg-0pky">
        <span style="font-weight:bold">Sentence 4</span>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/VOC/SamplingTemperature/4.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="OF_a_4">
          <source id="OF_s_4" src="./audio/OF/SamplingTemperature/4_667.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="OFND_a_4">
          <source id="OFND_s_4" src="./audio/OFND/SamplingTemperature/4_667.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="GTTS_a_4">
          <source id="GTTS_s_4" src="./audio/GTTS/SamplingTemperature/4_667.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="NHMM_a_4">
          <source id="NHMM_s_4" src="./audio/NHMM/SamplingTemperature/4_667.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
    <tr>
      <td nowrap class="tg-0pky">
        <span style="font-weight:bold">Sentence 5</span>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/VOC/SamplingTemperature/5.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="OF_a_5">
          <source id="OF_s_5" src="./audio/OF/SamplingTemperature/5_667.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="OFND_a_5">
          <source id="OFND_s_5" src="./audio/OFND/SamplingTemperature/5_667.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="GTTS_a_5">
          <source id="GTTS_s_5" src="./audio/GTTS/SamplingTemperature/5_667.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="NHMM_a_5">
          <source id="NHMM_s_5" src="./audio/NHMM/SamplingTemperature/5_667.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
    <tr>
      <td nowrap class="tg-0pky">
        <span style="font-weight:bold">Sentence 6</span>
      </td>
      <td class="tg-0pky">
        <audio controls>
          <source src="./audio/VOC/SamplingTemperature/6.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="OF_a_6">
          <source id="OF_s_6" src="./audio/OF/SamplingTemperature/6_667.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="OFND_a_6">
          <source id="OFND_s_6" src="./audio/OFND/SamplingTemperature/6_667.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="GTTS_a_6">
          <source id="GTTS_s_6" src="./audio/GTTS/SamplingTemperature/6_667.wav" type="audio/wav">
        </audio>
      </td>
      <td class="tg-0pky">
        <audio controls id="NHMM_a_6">
          <source id="NHMM_s_6" src="./audio/NHMM/SamplingTemperature/6_667.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
  </tbody>
</table>
<script>
  const values = [0, 0.334, 0.667, 1];
  const n_sent = 6;
  const file_names = ['0', '334', '667', '1'];
  var slider = document.getElementById("myRange");
  var output = document.getElementById("demo");
  const systems = ["OF", "OFND", "GTTS", "NHMM"];

  let audios = [];
  // initialize audios
  for(let i=0; i<systems.length; i++){
    let row = [];
    for(let j=0; j< n_sent; j++){
      src_audio = {
        'src': document.getElementById(`${systems[i]}_s_${j+1}`),
        'audio': document.getElementById(`${systems[i]}_a_${j+1}`)
      }

      row.push(src_audio);
    }
    audios.push(row);
  }

  output.innerHTML = values[slider.value];
  slider.oninput = function() {
    output.innerHTML = values[this.value];

    for (let i = 0; i < systems.length; i++){
      let number = this.value;

      for (let j=0; j< n_sent; j++){
        audios[i][j]['src'].src = `./audio/${systems[i]}/SamplingTemperature/${j+1}_${file_names[number]}.wav`;
        audios[i][j]['audio'].load();
      }
    }
  }
</script>


## Audio examples and code coming soon!
