<script lang="ts">
    import type { UploadedImage } from "$lib/types";

    export let images: UploadedImage[] = [];
</script>

<div class="container">
    <div class="img-grid">
        {#each images as img}
            <div class="img-container">
                <img src={img.url} alt="whatever" />
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
  }

  /* make images fill their container*/
  img {
    object-fit: cover;
    width: 100%;
    height: 100%;
  }

  img:hover {
    opacity: 0.9;
    cursor: pointer;
  }

  /* CSS Grid*/
  .img-grid {
    display: grid;
    grid-template-columns: repeat(1, 1fr);
    grid-gap: 5px;
  }

  /* Media Query for changing grid on bigger screens*/

  /* Bigger than Phones(tablet) */
  @media only screen and (min-width: 768px) {
    .img-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  /* Bigger than Phones(laptop / desktop) */
  @media only screen and (min-width: 992px) {
    .img-grid {
      grid-template-columns: repeat(3, 1fr);
    }
  }

  /* Bigger than Phones(laptop / desktop) */
  @media only screen and (min-width: 1200px) {
    .img-grid {
      grid-template-columns: repeat(4, 1fr);
    }
  }

</style>