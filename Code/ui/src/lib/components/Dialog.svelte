<script lang="ts">
  import type { DialogData } from "$lib/types";

  export let onClose: () => void = () => {}
  export let dialogData: DialogData;

</script>

<div class="dialog-mask">
  <div class="dialog">
    <div class="dialog-content">
      <div class="dialog-content-main">
        <div class="dialog-content-img">
          <img src={dialogData.url} alt="whatever" />
        </div>
        <div class="dialog-content-info"></div>
      </div>
      <div class="model-banner">
        <p>{dialogData.model}</p>
      </div>
    </div>
  </div>
</div>

<style>

  .model-banner {
    height: 2rem;
    display: flex;
    align-items: center;
  }

  .model-banner p {
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
  }

  .dialog {
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
      height: 75vh;
      width: 65vw;
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

</style>
