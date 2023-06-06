<script lang="ts">
  import RadarChart from "./RadarChart.svelte"
  import type { DialogData } from "$lib/types";

  export let onClose: () => void = () => {}
  export let dialogData: DialogData;

  let metadata = [];

  $: {
    metadata = Object.entries(dialogData.metadata);
  }

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
        <div class="left-main-content">
          <img src={dialogData.url} alt="whatever" />
        </div>
        <div class="right-main-content">
          <div class="scrollable-info">

            {#each metadata as data}
              <div class="model-info">
                <div class="header">
                  <p>{data[0]}</p>
                </div>
                <div class="body">
                  <p>{data[1]}</p>
                </div>
              </div>
            {/each}

          </div>
          <div class="radar-container">
            <RadarChart inputs={dialogData.probabilities}></RadarChart>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<style>

  .radar-container {
    position: relative;
    flex: 3;
    background: transparent !important;
    display: flex;
    align-items: center !important;
    justify-content: center !important;
  }

  .dialog-banner {
    height: 2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
  }

  .dialog-banner p {
    align-items: left;
    padding: 0;
    margin: 0;
    position: relative;
  }

  .dialog-content {
    position: absolute;
    top: 0;
    left: 0;
    padding: 0.5rem;
    height: calc(100% - 1rem);
    width: calc(100% - 1rem);
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .dialog-content-main {
    display: flex;
    flex: 1;
    gap: 0.25rem;
    position: relative;
  }

  .dialog {
    border: 1px solid #ccc;
    border-radius: 0.25rem;
    display: flex;
    flex-direction: column;
    z-index: 1000;
    height: 40rem;
    width: 60rem;
    position: relative;
    background: #eee;
    border: 1px solid #ccc;
  }

  .dialog-mask {
    z-index: 999999;
    pointer-events: auto;
    background-color: rgba(0,0,0,0.8);
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
    right: 0;
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

  .left-main-content {
    position: relative;
    flex: 3;
    border-radius: 2px;
  }

  .left-main-content img {
    position: absolute;
    object-fit: cover;
    width: 100%;
    height: 100%;
  }

  .right-main-content {
    position: relative;
    flex: 2;
    border-radius: 2px;
    display: flex;
    flex-direction: column;
  }

  .scrollable-info {
    height: 13rem;
    overflow: auto;
    position: relative;
  }

  .scrollable-info .model-info:not(:last-child) {
    margin-bottom: 3px;
  }

  .model-info {
    width: 100%;
    min-height: 2rem;
    display: flex;
    flex-direction: column;
    text-align: right;
  }

  p {
    margin: 0;
    padding: 0;
    font-size: 0.8rem;
  }

  .model-info .header {
    color: white;
    height: 1rem;
    display: flex;
    align-items: center;
    justify-content: left;
    background: #5ba0ce;
    padding: 0 0.1rem;
  }

  .model-info .body {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: left;
    padding: 0 0.1rem;
  }

</style>
