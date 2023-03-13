<script lang="ts">
  import type { UploadedImage } from "$lib/types";

  export let images: UploadedImage[] = [];
  export let closeHandler: (n: number) => void = (n) => {}

</script>

<div class="container">
  <div class="img-grid">
    {#each images as img, i}
      <div class="img-container">
        <img src={img.url} alt="whatever" />
        <!-- svelte-ignore a11y-click-events-have-key-events -->
        <div class="img-close" on:click={() => closeHandler(i)} />
      </div>
    {/each}
  </div>
</div>

<style>
  .container {
    background: #eee;
    border: 1px solid #ccc;
    border-radius: 0.25rem;
    padding: 0.75rem;
  }

  .img-container {
    width: 100%;
    height: 100%;
    position: relative;
  }

  /* make images fill their container*/
  img {
    object-fit: cover;
    width: 100%;
    height: 100%;
  }

  /* CSS Grid*/
  .img-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    grid-gap: 10px;
  }

  .img-close {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background-color: rgba(0, 0, 0, 0.5);
    position: absolute;
    top: 10px;
    right: 10px;
    text-align: center;
    line-height: 24px;
    z-index: 1;
    cursor: pointer;
  }

  .img-close:after {
    content: "✖";
    font-size: 14px;
    color: white;
  }
</style>
