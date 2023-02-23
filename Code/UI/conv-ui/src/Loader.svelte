<script>
  import { onMount } from 'svelte';

  let images = [];

  function handleFiles(event) {

    images = []
    const files = event.target.files;

    for (let i = 0; i < files.length; i++) {
      const reader = new FileReader();

      reader.onload = (e) => {
        images = [...images, e.target.result];
      };

      reader.readAsDataURL(files[i]);
    }
  }

  onMount(() => {
    const imageContainer = document.querySelector('.image-container');
    const imageWidth = imageContainer.clientWidth / 3;
    const numImages = images.length;

    const style = `
      .image-container {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        align-items: center;
        padding: 1rem;
        gap: 1rem;
        overflow-x: auto;
        scrollbar-width: none;
        -ms-overflow-style: none;
      }

      .image-container::-webkit-scrollbar {
        display: none;
      }

      .image-container img {
        width: ${imageWidth}px;
        height: ${imageWidth}px;
        object-fit: cover;
        border-radius: 0.25rem;
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
        transition: background-color 0.15s ease-in-out, border-color 0.15s ease-in-out,
          box-shadow 0.15s ease-in-out;
        cursor: pointer;
      }

      .file-input-button:hover {
        background-color: #0069d9;
      }
    `;

    const styleElement = document.createElement('style');
    styleElement.innerHTML = style;
    document.head.appendChild(styleElement);
  });
</script>

<div class="file-input-wrapper">
  <button class="file-input-button">Select Images</button>
  <input type="file" multiple on:change="{handleFiles}">
</div>

<div class="image-container">
  {#each images as image}
    <img src="{image}" alt="User uploaded image">
  {/each}
</div>
