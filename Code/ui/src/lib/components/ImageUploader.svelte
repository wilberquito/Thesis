<script lang="ts">
  // import ky from "ky";
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
      console.log(img);
      uploadedImages = [img, ...uploadedImages];
    }
  }

  async function postImages(images: UploadedImage[]) {
    try {
      const formData = new FormData();

      // // Append each image to the FormData object
      // for (const image of images)
      //   formData.append(
      //     "images[]",
      //     new Blo;([image.buffer], { type: "image/jpeg" })
      //   );

      // Makes a post request with all files
      // const response = await ky.post("https://example.com/upload", {
      //   body: formData,
      // });

      // Handle the response here
      // console.log(await response.json());
    } catch (error) {
      console.error(error);
    }
  }
</script>

<form
  on:submit|preventDefault={() => postImages(uploadedImages)}
  class="file-input-wrapper"
>
  <label class="upload-btn">
    <p>Upload images</p>
    <input
      type="file"
      class="upload file-input-buttom"
      multiple
      on:change={handleFiles}
    />
  </label>
  <label class="upload-btn">
    <p>Make prediction</p>
    <input type="submit" class="upload file-input-buttom" />
  </label>
</form>

{#if uploadedImages.length >= 1}
  <Displayer images={uploadedImages}/>
{/if}

<style>

  .file-input-wrapper {
    position: relative;
    display: inline-block;
    margin-top: 1rem;
    margin-bottom: 1rem;
  }

  .upload-btn {
    position: relative;
    display: inline-block;
    font-weight: 600;
    color: #fff;
    text-align: center;
    min-width: 116px;
    padding: 5px;
    transition: all 0.3s ease;
    cursor: pointer;
    border: 2px solid;
    background-color: #4045ba;
    border-color: #4045ba;
    border-radius: 10px;
    line-height: 26px;
    font-size: 14px;
  }

  .upload {
    margin: 0;
  }

  .upload-btn:hover {
    opacity: 0.9;
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
