<script lang="ts">

  export let onClose: () => void = () => {}
  let message="helllo"

</script>

<div class="dialog-mask">
  <div class="dialog">
    <div class="dialog-content">
      <div class="dialog-content-main">
        <div class="dialog-content-img"></div>
        <div class="dialog-content-info"></div>
      </div>
      <div class="model-banner">
        <p>Model name</p>
      </div>
    </div>
  </div>
</div>

<style>

  .model-banner {
    height: 2rem;
    background-color: whitesmoke;
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
  }

  .dialog-content-info {
    flex: 2;
    height: 100%;
  }

  .dialog {
    display: flex;
    flex-direction: column;
    z-index: 1000;
    height: 70vh;
    width: 70%;
    transform: scale(1);
    position: relative;
    background-color: white;
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
