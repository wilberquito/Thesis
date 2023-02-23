<script>
  import { onMount } from 'svelte';

  let images = [];

  function handleFiles(event) {
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
    `;

    const styleElement = document.createElement('style');
    styleElement.innerHTML = style;
    document.head.appendChild(styleElement);
  });
</script>

<input type="file" multiple on:change="{handleFiles}">
<div class="image-container">
  {#each images as image}
    <img src="{image}" alt="User uploaded image">
  {/each}
</div>

