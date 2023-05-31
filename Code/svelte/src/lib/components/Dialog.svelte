<script lang="ts">
  import RadarChart from "./RadarChart.svelte"
  import type { DialogData } from "$lib/types";

  export let onClose: () => void = () => {}
  export let dialogData: DialogData;
</script>

<div class="dialog-mask">
  <div class="dialog">
    <div class="dialog-content">
      <div class="dialog-banner">
        <p>{dialogData.name}</p>
        <!-- svelte-ignore a11y-click-events-have-key-events -->
          <div class="interactive-btn-img"
               on:click={onClose}>
            <span class="material-icons">
            close
            </span>
          </div>
      </div>
      <div class="dialog-content-main upper-case">
        <div class="img-container">
          <img src={dialogData.url} alt="whatever" />
        </div>
          <div style="width: 0.5rem"></div>
        <div class="dialog-content-info">
          <div class="entry-header">
            <p>Class</p>
          </div>
          <div>
            <p>melanoma</p>
          </div>
          <div class="entry-header">
            <p>Model name</p>
          </div>
          <div>
            <p>vicorobot...</p>
          </div>
          <div class="entry-header">
            <p>Base model</p>
          </div>
          <div>
            <p>Information.... ...</p>
          </div>
          <div class="entry-header">
            <p>Shape</p>
          </div>
          <div>
            <p>60px, 60px</p>
          </div>
          <div class="entry-header">
            <p>Size</p>
          </div>
          <div>
            <p>60192929</p>
          </div>

            <div class="radar-container">
          <RadarChart inputs={dialogData.probs}></RadarChart>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<style>

  .radar-container {
    position: relative;
    flex: auto;
    background: transparent !important;
    display: flex;
    align-items: center !important;
    justify-content: center !important;
  }

  .radar-container > * {
    position: absolute;
  }

  .dialog-banner {
    height: 2rem;
    display: flex;
    padding-bottom: 0.5rem;
    align-items: center;
    justify-content: center;
  }

  .dialog-banner p {
    align-items: left;
    padding: 0;
    margin: 0;
  }

  .dialog-content {
    position: absolute;
    top: 0;
    left: 0;
    padding: 1rem;
    height: calc(100% - 2rem);
    width: calc(100% - 2rem);
    display: flex;
    flex-direction: column;
  }

  .dialog-content-main {
    display: flex;
    width: 100%;
    height: 100%;
  }

  .dialog-content-info {
    flex: 3;
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .dialog-content-info p {
    margin: 0;
    padding: 0;
  }

  .dialog-content-info > div {
    padding-left: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: left;
  }

  .dialog-content-info > div:nth-child(even) {
    background: #e4e4e4;
    height: 2rem;
  }

  .dialog-content-info > div:nth-child(odd) {
    height: 2rem;
    background: #1779ba;
    color: white;
  }

  .dialog {
    border: 1px solid #ccc;
    border-radius: 0.25rem;
    display: flex;
    flex-direction: column;
    z-index: 1000;
    height: 85vh;
    width: 90vw;
    transform: scale(1);
    position: relative;
    background: #eee;
    border: 1px solid #ccc;
  }

  /* Bigger than Phones(laptop / desktop) */
  @media only screen and (min-width: 1200px) {

    .dialog {
      height: 52rem;
      width: 80rem;
    }

  }

  .dialog-mask {
    z-index: 999;
    pointer-events: auto;
    background-color: rgba(0,0,0,0.4);
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    transition-property: background-color;
    transition-duration: 0.2s;
  }

  .interactive-btn-img {
    width: 21px;
    height: 21px;
    right: 1rem;
    z-index: 1;
    border-radius: 50%;
    line-height: 24px;
    z-index: 1;
    cursor: pointer;
    background-color: black;
    color: white;
    text-align: center;
    opacity: 0.6;
    position: absolute;
  }

  .interactive-btn-img span {
    font-size: 21px;
  }

  .upper-case {
    text-transform: uppercase;
  }

  .img-container {
    padding: 0 !important;
    width: unset !important;
    height: unset !important;
    flex: 4;
    position: relative;
    padding: 2px;
    background-color: transparent;
    border-radius: 2px;
    position: relative;
  }

  .img-container img {
    position: absolute;
    object-fit: cover;
    width: 100%;
    height: 100%;
  }

</style>
