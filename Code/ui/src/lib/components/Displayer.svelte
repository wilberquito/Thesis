<script lang="ts">
  import type { UploadedImage } from "$lib/types";

  export let images: UploadedImage[] = [];
  export let letClose: boolean = false;
  export let closeHandler: (n: number) => void = (_) => {}
  export let expandHandler: (n: number) => void = (_) => {}

</script>

<div class="container">
  <div class="img-grid">
    {#each images as img, i}
      <div id={img.name}
           class="img-container"
           class:cancer={img?.meta?.pred === 'Cancer'}
           class:not-cancer={img?.meta?.pred === 'NotCancer'}>
        <img src={img.url} alt="whatever" />
        <!-- svelte-ignore a11y-click-events-have-key-events -->
        {#if letClose}
          <div class="interactive-btn-img"
               on:click={() => closeHandler(i)}>
            <span class="material-icons">
            close
            </span>
          </div>
        {/if}
        <!-- svelte-ignore a11y-click-events-have-key-events -->
        {#if img.meta}
          <div class="interactive-btn-img"
               on:click={() => expandHandler(i)}>
            <span class="material-icons">
            fullscreen
            </span>
          </div>
        {/if}
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

  .cancer {
    background-color: #FF0000 !important;
  }

  .not-cancer {
    background-color: #00FF00 !important;
  }

  .img-container {
    position: relative;
    padding: 2px;
    background-color: transparent;
    border-radius: 2px;
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

  .interactive-btn-img {
    width: 21px;
    height: 21px;
    position: absolute;
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
