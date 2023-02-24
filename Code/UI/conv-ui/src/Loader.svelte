<script>
  import { onMount } from "svelte";

  let images = [];

  function handleFiles(event) {
    images = [];
    const files = event.target.files;

    for (let i = 0; i < files.length; i++) {
      const reader = new FileReader();

      reader.onload = (e) => {
        images = [...images, e.target.result];
      };

      reader.readAsDataURL(files[i]);
    }
  }
</script>

<div class="file-input-wrapper">
  <button class="file-input-button">Select Images</button>
  <input type="file" multiple on:change={handleFiles} />
</div>

<div class="container">
  <div class="img-grid">
    {#each images as image}
      <div class="img-container">
        <img src={image} alt="User uploaded image" />
      </div>
    {/each}
  </div>
</div>

<style>

  .container {
    margin: auto;
  }

  .img-container {
    width: 100%;
    height: 100%;
  }

  /* mak images fill their container*/
  img {
    object-fit: cover;
    width: 100%;
    height: 100%;
  }

  img:hover {
    opacity: 0.5;
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

  .file-input-wrapper {
    position: relative;
    display: inline-block;
    margin-top: 1rem;
    margin-bottom: 1rem;
  }

  .file-input-wrapper input[type="file"] {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    opacity: 0;
    cursor: pointer;
  }
  .file-input-button {
    display: inline-block;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    font-weight: 500;
    line-height: 1.5;
    color: #fff;
    background-color: #007bff;
    border-radius: 0.25rem;
    transition: background-color 0.15s ease-in-out,
      border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    cursor: pointer;
  }

  .file-input-button:hover {
    background-color: #0069d9;
  }
</style>
