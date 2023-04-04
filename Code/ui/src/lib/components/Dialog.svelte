<script lang="ts">

  export let onClose: () => void = () => {}
  let message="helllo"

</script>

<div class="dialog-mask">
  <div class="dialog">
    <div class="dialog-content">
      <div class="dialog-content-img"></div>
      <div class="dialog-content-info"></div>
    </div>
  </div>
</div>

<style>

  .dialog-content {
    flex: 1;
    height: 70vh;
    background-color: white;
    display: flex;
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
    position: relative;
    display: flex;
    z-index: 1000;
    max-height: 90%;
    width: 70%;
    transform: scale(1);
    position: relative;
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
