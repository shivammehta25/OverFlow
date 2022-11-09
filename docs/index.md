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
</style>

## Summary

[Neural-HMMs][Neural_HMM_link] fuse classical HMM-based statistical speech synthesis and modern neural text-to-speech (TTS) retaining the best of both approaches. They require fewer data and fewer training updates to work while being less prone to gibberish output caused by neural attention failures. However, they could not model the highly non-Gaussian distribution of speech acoustics and despite being probabilistic we could not sample high-quality speech samples. To address these limitations, we propose a new approach **[OverFlow][github_link]** to improve the performance of neural HMMs by putting flows on top of them. This results in a powerful fully probabilistic model of durations and speech acoustics that can be trained using exact maximum likelihood giving more accurate pronunciations and better speech quality than comparable methods with or without sampling from the model.


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

3, 11, 19, 21, 29, 33, 37

## Stimuli from the listening tests

<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">Vocoded Speech</th>
      <th class="tg-0pky" colspan="3">Proposed OverFlow</th>
      <th class="tg-0pky">Tacotron 2</th>
      <th class="tg-0pky">GlowTTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th class="tg-fymr">VOC</th>
      <th class="tg-fymr">OF</th>
      <th class="tg-fymr">OFND (No Dropout)</th>
      <th class="tg-fymr">OFZT (Zero Temperature)</th>
      <th class="tg-fymr">T2</th>
      <th class="tg-fymr">GTTS</th>
    </tr>
    <tr>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/VOC/ListeningTest/1.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OF/ListeningTest/1.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OFND/ListeningTest/1.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OFZT/ListeningTest/1.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/T2/ListeningTest/1.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/GTTS/ListeningTest/1.wav" type="audio/wav">
          </audio>
        </td>
    </tr>
    <tr>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/VOC/ListeningTest/2.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OF/ListeningTest/2.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OFND/ListeningTest/2.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OFZT/ListeningTest/2.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/T2/ListeningTest/2.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/GTTS/ListeningTest/2.wav" type="audio/wav">
          </audio>
        </td>
    </tr>
    <tr>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/VOC/ListeningTest/3.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OF/ListeningTest/3.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OFND/ListeningTest/3.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OFZT/ListeningTest/3.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/T2/ListeningTest/3.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/GTTS/ListeningTest/3.wav" type="audio/wav">
          </audio>
        </td>
    </tr>
    <tr>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/VOC/ListeningTest/4.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OF/ListeningTest/4.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OFND/ListeningTest/4.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OFZT/ListeningTest/4.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/T2/ListeningTest/4.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/GTTS/ListeningTest/4.wav" type="audio/wav">
          </audio>
        </td>
    </tr>
    <tr>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/VOC/ListeningTest/5.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OF/ListeningTest/5.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OFND/ListeningTest/5.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OFZT/ListeningTest/5.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/T2/ListeningTest/5.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/GTTS/ListeningTest/5.wav" type="audio/wav">
          </audio>
        </td>
    </tr>
    <tr>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/VOC/ListeningTest/6.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OF/ListeningTest/6.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OFND/ListeningTest/6.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OFZT/ListeningTest/6.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/T2/ListeningTest/6.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/GTTS/ListeningTest/6.wav" type="audio/wav">
          </audio>
        </td>
    </tr>
    <tr>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/VOC/ListeningTest/7.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OF/ListeningTest/7.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OFND/ListeningTest/7.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/OFZT/ListeningTest/7.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/T2/ListeningTest/7.wav" type="audio/wav">
          </audio>
        </td>
        <td class="tg-0pky">
          <audio id="audio-small" controls  style="width: 110px;">
            <source src="./audio/GTTS/ListeningTest/7.wav" type="audio/wav">
          </audio>
        </td>
    </tr>
  </tbody>
</table>


## Audio examples and code coming soon!
