<script lang="ts">
  import ky from "ky";
  import type { UploadedImage } from "$lib/types";
  import Displayer from "./Displayer.svelte";

  let uploadedImages: UploadedImage[] = [];

  async function handleFiles(event: any) {
    // Removes previous uploadedFiles
    uploadedImages = [];

    const files = event.target.files;

    for (const file of files) {
      const blob = new Blob([file], { type: file.type });
      const img: UploadedImage = {
        name: file.name,
        blob: blob,
        url: URL.createObjectURL(blob),
      };
      uploadedImages = [img, ...uploadedImages];
    }
    uploadedImages = uploadedImages.sort((a, b) => (a.name > b.name ? 1 : -1));
  }

  async function postImages(images: UploadedImage[]) {
    try {
      const formData = new FormData();
      const endpoint =
        images.length == 1
          ? "https://127.0.0.1:8080/predict"
          : "https://127.0.0.1:8080/predict_bulk";

      for (const img of images) formData.append("images[]", img.blob);

      const request = await ky.post(endpoint, {
        method: "POST",
        body: formData,
        headers: {
          "content-type": "application/json",
        },
        onDownloadProgress: (progress, chunk) => {
          // Example output:
          // `0% - 0 of 1271 bytes`
          // `100% - 1271 of 1271 bytes`
          console.log(
            `${progress.percent * 100}% - ${progress.transferredBytes} of ${
              progress.totalBytes
            } bytes`
          );
        },
      });

      const response = await request.json();
      console.log(response);
    } catch (error) {
      console.error(error);
    }
  }
</script>

<div class="layout">
  <form
    on:submit|preventDefault={() => postImages(uploadedImages)}
    class="file-input-wrapper"
  >
    <label class="btn">
      <p>Select Your Images+</p>
      <input
        type="file"
        class="upload selectable file-input-buttom"
        multiple
        on:change={handleFiles}
      />
    </label>
    <br />
    <label class="upload-btn btn">
      <p>Make prediction</p>
      <input
        type="submit"
        class="upload"
        disabled={uploadedImages.length < 1}
      />
    </label>
  </form>

  {#if uploadedImages.length >= 1}
    <Displayer images={uploadedImages} />
  {/if}
</div>

<style>
  p {
    margin: 0;
  }

  .layout {
    position: relative;
    padding: 0 1rem;
  }

  @media only screen and (min-width: 768px) {
    .layout {
      padding: 0 10vw;
    }
  }

  /* Bigger than Phones(laptop / desktop) */
  @media only screen and (min-width: 992px) {
    .layout {
      padding: 0 20vw;
    }
  }

  /* Bigger than Phones(laptop / desktop) */
  @media only screen and (min-width: 1200px) {
    .layout {
      padding: 0 25vw;
    }
  }

  .upload-btn {
    margin-top: 0.5rem;
    width: calc(100% - 14px);
    padding: 1.25rem 6px !important;
    background-color: #1779ba;
    color: white !important;
  }

  .upload-btn p {
    font-size: 1.2rem !important;
  }

  .upload-btn:has(> input[disabled]) {
    background-color: #cccccc;
    border-color: #4e7691;
    color: #666666;
  }

  .file-input-wrapper {
    position: relative;
    margin-top: 1rem;
    margin-bottom: 1rem;
  }

  .btn {
    position: relative;
    border-radius: 0.25rem;
    display: inline-block;
    font-weight: 600;
    text-align: center;
    min-width: 116px;
    padding: 1rem 6px;
    transition: all 0.3s ease;
    border: 1px solid #1779ba;
    font-size: 14px;
    color: #1779ba;
  }

  .selectable {
    cursor: pointer;
  }

  .btn p {
    font-size: 0.8rem;
  }

  .file-input-wrapper input {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    opacity: 0;
    cursor: pointer;
  }
</style>
