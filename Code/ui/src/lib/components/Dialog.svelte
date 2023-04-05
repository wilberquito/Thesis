<script lang="ts">
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
      <div class="dialog-content-main">
        <div class="dialog-content-img">
          <img src={dialogData.url} alt="whatever" />
        </div>
        <div class="dialog-content-info">
          <div class="entry-header">
            <p>About image</p>
          </div>
          <div></div>
          <div class="entry-header">
            <p>About prediction</p>
          </div>
          <div></div>
          <div class="entry-header">
            <p>About model</p>
          </div>
          <div></div>
        </div>
      </div>
    </div>
  </div>
</div>

<style>

  .dialog-banner {
    height: 2rem;
    display: flex;
    align-items: center;
    padding-bottom: 0.5rem;
  }

  .dialog-banner p {
    width: 100%;
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

  .dialog-content-img {
    flex: 3;
    height: 100%;
    background-color: red;
    position: relative;
  }

  .dialog-content-img img {
    object-fit: cover;
    width: 100%;
    height: 100%;
    position: absolute;
  }

  .dialog-content-info {
    flex: 2;
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .dialog-content-info p {
    margin: 0;
    padding: 0;
  }

  .dialog-content-info > div {
    width: 100;

  }

  .entry-header {
    width: unset !important;
    padding-right: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: right;
  }

  .dialog-content-info > div:nth-child(even) {
    background: blue;
    flex: 1;
  }

  .dialog-content-info > div:nth-child(odd) {
    width: 100%;
    height: 2rem;
    background: red;
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
    top: 10px;
    right: 10px;
    z-index: 1;
    border-radius: 50%;
    line-height: 24px;
    z-index: 1;
    cursor: pointer;
    background-color: black;
    color: white;
    text-align: center;
    opacity: 0.6;
  }

  .interactive-btn-img span {
    font-size: 21px;
  }

</style>
