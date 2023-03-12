<script lang="ts">
  import ky from "ky";
  import type { UploadedImage, ImagePayload } from "$lib/types";
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
      const pack: ImagePayload[] = [];
      // Append each image to the FormData object
      for (const image of images) {
        const base64 = await blobToBase64(image.blob);
        if (base64) {
          pack.push({
            name: image.name,
            base64: base64,
          });
        }
      }

      const message = {
        pack: pack
      }

      const request = await ky.post("https://example.com/upload", {
        method: "POST",
        body: JSON.stringify(message),
        headers: {
		'content-type': 'application/json'
	}

      });

      const response = await request.json();
      console.log(response);
    } catch (error) {
      console.error(error);
    }

    function blobToBase64(blob: Blob): Promise<string | ArrayBuffer | null> {
      return new Promise(
        (resolve: (a: string | ArrayBuffer | null) => void, _) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result);
          reader.readAsDataURL(blob);
        }
      );
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
  <Displayer images={uploadedImages} />
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
